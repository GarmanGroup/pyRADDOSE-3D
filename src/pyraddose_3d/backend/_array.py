"""Array-library dispatch, so the numerical core runs on NumPy or PyTorch.

The port has two goals that pull against each other. Gradient-based
optimisation of a data-collection strategy wants PyTorch. Distribution inside
CCP4 does not: the torch stack measures ~570 MB installed against ~31 MB for
NumPy, pydantic and PyYAML together, and pulls CUDA toolkit, cuDNN, NCCL and
Triton on Linux.

Both are satisfied by writing the core against the subset of the array API
that NumPy and PyTorch already agree on, and dispatching the places where
they do not. That subset is wider than it looks -- ``torch`` accepts NumPy's
``axis=`` keyword on ``sum``, ``cumsum`` and ``concat``, and shares the
spelling of ``matmul``, ``sort``, ``where`` and ``reshape``. Of everything
RADDOSE-3D's MX path needs, :func:`erf` is the only divergence.

NumPy is the default and a required dependency. PyTorch is an extra::

    pip install pyRADDOSE-3D            # CCP4-sized; no gradients
    pip install pyRADDOSE-3D[torch]     # gradients

Four divergences have turned up so far, all handled here: :func:`erf`, which
NumPy does not have; :func:`sort`, which torch returns as a
``(values, indices)`` pair where NumPy returns just the values;
:func:`astype`, which is a NumPy array method with no torch equivalent; and
:func:`to_namespace`, because torch rejects an ndarray where NumPy would
have accepted a tensor, so a mixed expression has to be reconciled rather
than merely dispatched.

Nothing in this package imports :mod:`torch` at module scope. The import
happens only once an array that is *already* a tensor reaches
:func:`array_namespace`, so a torch-less install never pays for it and never
fails. ``tests/test_architecture.py`` enforces that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

__all__ = [
    "array_namespace",
    "astype",
    "erf",
    "is_torch",
    "promote",
    "sort",
    "to_namespace",
]


def is_torch(x: Any) -> bool:
    """Report whether ``x`` is a PyTorch tensor.

    Checks the defining module by name rather than with ``isinstance``, so
    that this is answerable without importing :mod:`torch`.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        True if ``x`` is a ``torch`` tensor.
    """
    return type(x).__module__.partition(".")[0] == "torch"


def array_namespace(*arrays: Any) -> ModuleType:
    """Return the array library to use for ``arrays``.

    Mirrors ``torch``'s own promotion rule: a tensor anywhere in the
    arguments makes the whole expression a tensor expression, because torch
    accepts NumPy scalars and arrays as operands but not the reverse.

    Parameters
    ----------
    *arrays : Any
        The operands of the expression about to be evaluated.

    Returns
    -------
    ModuleType
        :mod:`torch` if any operand is a tensor, otherwise :mod:`numpy`.
    """
    if any(is_torch(a) for a in arrays):
        import torch

        return torch
    return cast("ModuleType", np)


def erf(x: Any) -> Any:
    """Gauss error function, on either backend.

    The only array-API divergence the MX path hits. It is needed for the
    fraction of a Gaussian beam passing a collimation aperture, so it sits
    directly on the differentiable path and must carry a gradient under
    torch -- which ``torch.erf`` does.

    Parameters
    ----------
    x : Any
        Array or scalar.

    Returns
    -------
    Any
        ``erf(x)``, of the same array type as ``x``.
    """
    if is_torch(x):
        import torch

        return torch.erf(x)
    from scipy.special import erf as _erf

    return _erf(x)


def promote(*values: Any) -> tuple[ModuleType, tuple[Any, ...]]:
    """Return the namespace for ``values``, and ``values`` within it.

    NumPy functions accept Python floats wherever they accept arrays; torch
    functions do not, so ``torch.cos(90.0)`` raises where ``numpy.cos(90.0)``
    is fine. Any function that mixes traced arrays with plain-float defaults
    -- an angle defaulting to 90 degrees, say -- needs its arguments brought
    into one type first.

    Python numbers take the dtype of the first floating tensor among
    ``values``, so a float32 calculation is not silently widened; with none,
    float64. NumPy arrays keep their own dtype: a float64 mesh is never cut
    to float32 because a float32 tensor came first, as it once was. Every
    conversion lands on the first tensor's device.

    Parameters
    ----------
    *values : Any
        Arrays, tensors and scalars destined for the same expression.

    Returns
    -------
    tuple[ModuleType, tuple[Any, ...]]
        The namespace, and the values converted where needed. Under NumPy
        the values are returned untouched.
    """
    xp = array_namespace(*values)
    if xp is np:
        return xp, values

    import torch

    tensors = [v for v in values if is_torch(v)]
    device = tensors[0].device
    number = next((t.dtype for t in tensors if t.is_floating_point()), torch.float64)

    def convert(v: Any) -> Any:
        if is_torch(v):
            return v
        if isinstance(v, np.ndarray | np.generic):
            return torch.as_tensor(v, device=device)
        return torch.as_tensor(v, dtype=number, device=device)

    return xp, tuple(convert(v) for v in values)


def sort(x: Any, axis: int = -1) -> Any:
    """Sort along an axis, returning only the values.

    ``torch.sort`` returns a ``(values, indices)`` named tuple where
    ``numpy.sort`` returns the array, so the two cannot be called the same
    way. Only the values are ever wanted here.

    Parameters
    ----------
    x : Any
        Array or tensor to sort.
    axis : int, optional
        Axis to sort along. Default the last.

    Returns
    -------
    Any
        The sorted values, ascending.
    """
    if is_torch(x):
        import torch

        return torch.sort(x, dim=axis).values
    return np.sort(x, axis=axis)


def astype(x: Any, dtype: Any) -> Any:
    """Convert to another dtype.

    ``numpy`` spells this as an array method and ``torch`` as ``Tensor.to``,
    so neither can be called through the other's name.

    Parameters
    ----------
    x : Any
        Array or tensor.
    dtype : Any
        Target dtype, in whichever library ``x`` belongs to.

    Returns
    -------
    Any
        ``x`` as ``dtype``.
    """
    if is_torch(x):
        return x.to(dtype)
    return x.astype(dtype)


def to_namespace(xp: ModuleType, x: Any, like: Any = None) -> Any:
    """Bring ``x`` into ``xp``, leaving it alone if it is already there.

    The fourth divergence, and the one that bites hardest. Choosing a
    namespace from *some* of the operands and then using it on *all* of them
    only works if every operand belongs to it: ``torch.zeros_like`` rejects
    an ndarray outright, where NumPy would have accepted a tensor.

    That mixture is the normal case rather than an exotic one. A crystal mesh
    is fixed geometry with no gradient worth having -- its occupancy test is
    a boolean parity count -- so it stays a NumPy array, while the beam and
    wedge settings a strategy optimiser tunes are tensors. Something has to
    reconcile them at the boundary.

    Parameters
    ----------
    xp : ModuleType
        Target namespace, from :func:`array_namespace`.
    x : Any
        Array, tensor or scalar.
    like : Any, optional
        A tensor whose device the result should be on. Without it, a new
        tensor is on the CPU.

    Returns
    -------
    Any
        ``x`` as one of ``xp``'s arrays. Converting a NumPy array to a
        tensor shares its memory rather than copying, and a tensor that is
        already a tensor is returned unchanged, gradient history intact,
        unless it must move to ``like``'s device. Python numbers become what
        NumPy would make them: float64, int64 or bool. ``torch.as_tensor``
        made Python floats float32, torch's default.
    """
    if xp is np:
        return x.detach().cpu().numpy() if is_torch(x) else np.asarray(x)
    device = like.device if is_torch(like) else None
    if is_torch(x):
        return x if device is None or x.device == device else x.to(device)
    if isinstance(x, (list, tuple)) and any(is_torch(v) for v in x):
        # torch.as_tensor refuses a sequence containing tensors, and would
        # drop their gradient history even if it did not.
        return xp.stack([to_namespace(xp, v, like) for v in x])
    return xp.as_tensor(np.asarray(x), device=device)
