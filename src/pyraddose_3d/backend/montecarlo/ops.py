"""The Monte Carlo's arrays, on NumPy or on a torch device.

The Monte Carlo, GOS and XFEL subprograms are written once, in NumPy's
names, and run on whichever library their arrays belong to. Each function
takes ``xp`` from :func:`namespace`: for NumPy arrays NumPy itself, so that
the NumPy path calls exactly what it always has, and for tensors a
:class:`TorchNamespace`, which gives NumPy's functions of the same names on
the tensors' device, in float64, with NumPy's dtypes and argument
conventions.

A run on a device starts from :class:`TorchRandom`, whose draws are torch's
own and are made on the device. Its tables, meshes and coefficients are
copied there once (:func:`on`), and its tallies come back to NumPy at the
end, to be processed as NumPy's are (:func:`to_numpy`).

The two paths agree to the last bit or so in each operation, not in a run:
torch's draws are not NumPy's, its transcendental functions round
differently, and on a GPU its tallies add in no fixed order. A run on a
device is compared with RADDOSE-3D's statistically, as NumPy's is.
:class:`MirroredRandom` puts NumPy's draws on the device, so that a run
there can be followed alongside one on NumPy.

Nothing here imports :mod:`torch` until a device is asked for.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend._array import is_torch

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "MirroredRandom",
    "TorchNamespace",
    "TorchRandom",
    "float64_by_default",
    "namespace",
    "ndtri",
    "on",
    "random_namespace",
    "to_numpy",
    "torch_namespace",
]


def namespace(*arrays: Any) -> Any:
    """NumPy, or the namespace of the device the first tensor is on."""
    for array in arrays:
        if is_torch(array):
            return torch_namespace(str(array.device))
    return np


def random_namespace(rng: Any) -> Any:
    """The namespace a generator's draws belong to: NumPy's has none."""
    return getattr(rng, "xp", np)


@functools.cache
def torch_namespace(device: str) -> TorchNamespace:
    """The namespace for ``device``, one for each: ``"cuda"`` is the current one."""
    import torch

    place = torch.device(device)
    if place.type == "cuda" and place.index is None:
        place = torch.device("cuda", torch.cuda.current_device())
    if str(place) != device:
        return torch_namespace(str(place))
    return TorchNamespace(place)


def to_numpy(x: Any) -> Any:
    """A tensor as a NumPy array, anything else as it is."""
    if is_torch(x):
        return x.detach().cpu().numpy()
    return x


def ndtri(x: Any) -> Any:
    """The standard normal's quantile function, on either library."""
    if is_torch(x):
        import torch

        return torch.special.ndtri(x)
    from scipy.special import ndtri as _ndtri

    return _ndtri(x)


@contextlib.contextmanager
def float64_by_default(xp: Any) -> Iterator[None]:
    """Run in torch's inference mode, with float64 its default dtype.

    torch makes float32 of an integer tensor's quotient, of an integer
    tensor times a float, and of a ``where`` between two floats; NumPy makes
    float64, and so must the Monte Carlo. Nothing on NumPy changes.
    """
    if xp is np:
        yield
        return
    torch = xp.torch
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.inference_mode():
            yield
    finally:
        torch.set_default_dtype(before)


#: Copies on devices: (id, device) -> (the original, its copy). Keeping the
#: original keeps its id from being reused while its copy is held.
_ON: dict[tuple[int, str], tuple[Any, Any]] = {}
#: Copies kept, the oldest dropped first: tables, meshes, a run's media.
_ON_KEPT = 64


def on(xp: Any, value: Any, *, keep: bool = True) -> Any:
    """``value`` on ``xp``'s device: itself on NumPy.

    An array becomes a tensor. A dataclass gets a copy with its arrays moved,
    and every dataclass in it whose fields are all arrays, numbers or
    ``None`` moved too -- a table, a mesh, a set of coefficients -- but not
    one that holds anything else, such as a material, which stays as it is.
    Copies are kept, so that asking again for the same object is free,
    unless ``keep`` is false: for something asked for once.
    """
    if xp is np or value is None:
        return value
    if not keep:
        return _move(xp, value)
    key = (id(value), str(xp.device))
    held = _ON.get(key)
    if held is not None and held[0] is value:
        return held[1]
    moved = _move(xp, value)
    if len(_ON) >= _ON_KEPT:
        del _ON[next(iter(_ON))]
    _ON[key] = (value, moved)
    return moved


def _plain(value: Any) -> bool:
    """A dataclass of arrays, numbers and ``None`` only."""
    return dataclasses.is_dataclass(value) and all(
        isinstance(getattr(value, f.name), np.ndarray | int | float | bool | None)
        or is_torch(getattr(value, f.name))
        or _plain(getattr(value, f.name))
        for f in dataclasses.fields(value)
    )


def _move(xp: Any, value: Any) -> Any:
    if isinstance(value, np.ndarray) or is_torch(value):
        return xp.asarray(value)
    if isinstance(value, tuple):
        return tuple(_move(xp, item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {}
        for f in dataclasses.fields(value):
            item = getattr(value, f.name)
            if isinstance(item, np.ndarray) or is_torch(item):
                changes[f.name] = xp.asarray(item)
            elif _plain(item):
                changes[f.name] = _move(xp, item)
        return dataclasses.replace(value, **changes)
    raise TypeError(f"cannot move a {type(value).__name__} to a device")


class _TorchAdd:
    """``numpy.add``'s ``at`` and ``accumulate``."""

    def __init__(self, xp: TorchNamespace) -> None:
        self._xp = xp

    def at(self, target: Any, index: Any, values: Any) -> None:
        """Add ``values`` at ``index``, as many times as an index repeats."""
        xp = self._xp
        index = index if isinstance(index, tuple) else (index,)
        index = tuple(xp.asarray(i) for i in index)
        target.index_put_(
            index, xp.asarray(values, dtype=target.dtype), accumulate=True
        )

    def accumulate(self, a: Any, axis: int = 0) -> Any:
        """The running sums along ``axis``."""
        return self._xp.torch.cumsum(a, dim=axis)


class _TorchLinalg:
    def __init__(self, xp: TorchNamespace) -> None:
        self._xp = xp

    def norm(self, a: Any, axis: int | None = None) -> Any:
        return self._xp.torch.linalg.norm(self._xp.asarray(a), dim=axis)


class TorchNamespace:
    """NumPy's functions, as the Monte Carlo calls them, on one torch device.

    Arrays are made on the device; floats are float64 and integers int64, as
    NumPy makes them; NumPy arrays and Python numbers passed in are brought
    to the device. Only what the Monte Carlo calls is here.
    """

    def __init__(self, device: Any) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.int64 = torch.int64
        self.float64 = torch.float64
        self.bool_ = torch.bool
        self.inf = math.inf
        self.nan = math.nan
        self.pi = math.pi
        self.add = _TorchAdd(self)
        self.linalg = _TorchLinalg(self)

    def __repr__(self) -> str:
        return f"TorchNamespace({str(self.device)!r})"

    # --- dtypes and conversion ------------------------------------------

    def _dtype(self, dtype: Any) -> Any:
        """A NumPy or Python dtype as torch's; torch's as it is."""
        if dtype is None or isinstance(dtype, self.torch.dtype):
            return dtype
        return self.torch.from_numpy(np.empty(0, dtype=np.dtype(dtype))).dtype

    def _numpy_dtype(self, dtype: Any) -> Any:
        if isinstance(dtype, self.torch.dtype):
            return self.torch.empty(0, dtype=dtype).numpy().dtype
        return dtype

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """On the device, as a tensor: NumPy's dtype for anything not one."""
        if is_torch(x):
            return x.to(device=self.device, dtype=self._dtype(dtype) or x.dtype)
        if isinstance(x, list | tuple) and any(is_torch(item) for item in x):
            return self.stack([self.asarray(item, dtype) for item in x])
        array = np.asarray(x, dtype=self._numpy_dtype(dtype))
        return self.torch.tensor(array, device=self.device)

    array = asarray

    def astype(self, x: Any, dtype: Any) -> Any:
        return self.asarray(x).to(self._dtype(dtype))

    def copy(self, x: Any) -> Any:
        return self.asarray(x).clone()

    def _operand(self, x: Any) -> Any:
        """An operand torch takes: a tensor, or a Python number to promote.

        A float becomes a float64 tensor where torch would make it float32;
        anything else becomes a tensor.
        """
        if is_torch(x) or isinstance(x, int | bool):
            return x
        if isinstance(x, float):
            if self.torch.get_default_dtype() == self.torch.float64:
                return x
            return self.torch.tensor(x, dtype=self.torch.float64, device=self.device)
        return self.asarray(x)

    # --- making arrays ------------------------------------------------

    def zeros(self, shape: Any, dtype: Any = float) -> Any:
        return self.torch.zeros(shape, dtype=self._dtype(dtype), device=self.device)

    def ones(self, shape: Any, dtype: Any = float) -> Any:
        return self.torch.ones(shape, dtype=self._dtype(dtype), device=self.device)

    def empty(self, shape: Any, dtype: Any = float) -> Any:
        return self.torch.empty(shape, dtype=self._dtype(dtype), device=self.device)

    def full(self, shape: Any, value: Any, dtype: Any = None) -> Any:
        if dtype is None:
            dtype = np.asarray(value).dtype
        return self.torch.full(
            (shape,) if isinstance(shape, int) else shape,
            value,
            dtype=self._dtype(dtype),
            device=self.device,
        )

    def zeros_like(self, x: Any) -> Any:
        return self.torch.zeros_like(x)

    def empty_like(self, x: Any) -> Any:
        return self.torch.empty_like(x)

    def arange(self, *args: Any, dtype: Any = None) -> Any:
        if dtype is None:
            dtype = np.int64 if all(isinstance(a, int) for a in args) else float
        return self.torch.arange(*args, dtype=self._dtype(dtype), device=self.device)

    def broadcast_to(self, x: Any, shape: Any) -> Any:
        return self.torch.broadcast_to(self.asarray(x), shape)

    # --- elementwise ----------------------------------------------------

    def where(self, condition: Any, a: Any, b: Any) -> Any:
        return self.torch.where(
            self.asarray(condition), self._operand(a), self._operand(b)
        )

    def power(self, a: Any, b: Any) -> Any:
        return self.torch.pow(self._operand(a), self._operand(b))

    def maximum(self, a: Any, b: Any) -> Any:
        if isinstance(b, int | float):
            return self.torch.clamp(self.asarray(a), min=b)
        if isinstance(a, int | float):
            return self.torch.clamp(self.asarray(b), min=a)
        return self.torch.maximum(self.asarray(a), self.asarray(b))

    def minimum(self, a: Any, b: Any) -> Any:
        if isinstance(b, int | float):
            return self.torch.clamp(self.asarray(a), max=b)
        if isinstance(a, int | float):
            return self.torch.clamp(self.asarray(b), max=a)
        return self.torch.minimum(self.asarray(a), self.asarray(b))

    def clip(self, a: Any, low: Any, high: Any) -> Any:
        if isinstance(low, int | float) and isinstance(high, int | float):
            return self.torch.clamp(self.asarray(a), low, high)
        return self.torch.clamp(self.asarray(a), self.asarray(low), self.asarray(high))

    def sqrt(self, x: Any) -> Any:
        return self.torch.sqrt(self.asarray(x))

    def log(self, x: Any) -> Any:
        return self.torch.log(self.asarray(x))

    def exp(self, x: Any) -> Any:
        return self.torch.exp(self.asarray(x))

    def cos(self, x: Any) -> Any:
        return self.torch.cos(self.asarray(x))

    def sin(self, x: Any) -> Any:
        return self.torch.sin(self.asarray(x))

    def tan(self, x: Any) -> Any:
        return self.torch.tan(self.asarray(x))

    def arccos(self, x: Any) -> Any:
        return self.torch.acos(self.asarray(x))

    def arctan(self, x: Any) -> Any:
        return self.torch.atan(self.asarray(x))

    def abs(self, x: Any) -> Any:
        return self.torch.abs(self.asarray(x))

    def sign(self, x: Any) -> Any:
        return self.torch.sign(self.asarray(x))

    def floor(self, x: Any) -> Any:
        return self.torch.floor(self.asarray(x))

    def trunc(self, x: Any) -> Any:
        return self.torch.trunc(self.asarray(x))

    def isnan(self, x: Any) -> Any:
        return self.torch.isnan(self.asarray(x))

    def isfinite(self, x: Any) -> Any:
        return self.torch.isfinite(self.asarray(x))

    def nan_to_num(self, x: Any) -> Any:
        return self.torch.nan_to_num(self.asarray(x))

    def radians(self, x: Any) -> Any:
        return self.torch.deg2rad(self.asarray(x))

    # --- reductions and searches -----------------------------------------

    def sum(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return self.torch.sum(self.asarray(a), dim=axis, keepdim=keepdims)

    def max(self, a: Any, axis: Any = None, initial: Any = None) -> Any:
        a = self.asarray(a)
        if initial is not None and a.numel() == 0:
            return self.asarray(initial)
        found = self.torch.amax(a, dim=() if axis is None else axis)
        return found if initial is None else self.torch.clamp(found, min=initial)

    def min(self, a: Any, axis: Any = None) -> Any:
        return self.torch.amin(self.asarray(a), dim=() if axis is None else axis)

    def cumsum(self, a: Any, axis: int) -> Any:
        return self.torch.cumsum(self.asarray(a), dim=axis)

    def argmax(self, a: Any, axis: Any = None) -> Any:
        a = self.asarray(a)
        if a.dtype == self.torch.bool:
            a = a.to(self.torch.uint8)
        return self.torch.argmax(a, dim=axis)

    def any(self, a: Any, axis: Any = None) -> Any:
        a = self.asarray(a)
        return a.any() if axis is None else a.any(dim=axis)

    def flatnonzero(self, a: Any) -> Any:
        return self.torch.nonzero(self.asarray(a).reshape(-1), as_tuple=True)[0]

    def searchsorted(self, a: Any, v: Any, side: str = "left") -> Any:
        return self.torch.searchsorted(
            self.asarray(a).contiguous(), self.asarray(v).contiguous(), side=side
        )

    def unique(self, a: Any, return_inverse: bool = False) -> Any:
        return self.torch.unique(
            self.asarray(a), sorted=True, return_inverse=return_inverse
        )

    def isin(self, a: Any, values: Any) -> Any:
        return self.torch.isin(self.asarray(a), self.asarray(values))

    def sort(self, a: Any, axis: int = -1) -> Any:
        return self.torch.sort(self.asarray(a), dim=axis).values

    def array_equal(self, a: Any, b: Any) -> bool:
        a, b = self.asarray(a), self.asarray(b)
        return a.shape == b.shape and bool(self.torch.equal(a, b))

    # --- joining and products ---------------------------------------------

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return self.torch.cat([self.asarray(a) for a in arrays], dim=axis)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return self.torch.stack([self.asarray(a) for a in arrays], dim=axis)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return self.torch.einsum(subscripts, *(self.asarray(o) for o in operands))

    def cross(self, a: Any, b: Any) -> Any:
        """NumPy's: each component a product less a product, unfused."""
        a, b = self.asarray(a), self.asarray(b)
        return self.torch.stack(
            [
                a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
            ],
            dim=-1,
        )

    def errstate(self, **_: Any) -> contextlib.nullcontext[None]:
        """Torch neither warns nor raises on a division by zero or a NaN."""
        return contextlib.nullcontext()


class TorchRandom:
    """Uniform draws and permutations from torch's generator, on a device.

    Seeded as NumPy's would be from the experiment's seed; the draws are not
    NumPy's, but the same seed on the same device gives the same run -- on a
    GPU, to the last bits of its tallies, which it adds in no fixed order.
    """

    #: A cascade's transitions are drawn for all its vacancies at once, not
    #: group by group as NumPy's are: one draw, not one a group.
    draws_in_groups = False

    def __init__(self, seed: int | None, device: str) -> None:
        import torch

        self.xp = torch_namespace(device)
        self.generator = torch.Generator(device=self.xp.device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

    def random(self, n: int) -> Any:
        """``n`` uniform draws in [0, 1), float64."""
        torch = self.xp.torch
        return torch.rand(
            n, generator=self.generator, device=self.xp.device, dtype=torch.float64
        )

    def permutation(self, n: int) -> Any:
        """A random order of ``range(n)``."""
        return self.xp.torch.randperm(
            n, generator=self.generator, device=self.xp.device
        )


class MirroredRandom:
    """NumPy's draws, on a device: a run there that follows one on NumPy."""

    def __init__(self, rng: np.random.Generator, device: str) -> None:
        self.rng = rng
        self.xp = torch_namespace(device)

    def random(self, n: int) -> Any:
        return self.xp.asarray(self.rng.random(n))

    def permutation(self, n: int) -> Any:
        return self.xp.asarray(self.rng.permutation(n))
