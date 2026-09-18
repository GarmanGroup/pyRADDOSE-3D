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

__all__ = ["array_namespace", "erf", "is_torch", "promote"]


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

    Scalars take the dtype of the first tensor among ``values``, so a float32
    calculation is not silently widened.

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

    dtype = next((v.dtype for v in values if is_torch(v)), None)
    return xp, tuple(
        v if is_torch(v) else torch.as_tensor(v, dtype=dtype) for v in values
    )
