"""Where photons and electrons are, relative to the crystal.

RADDOSE-3D's Monte Carlo rotates the crystal for each photon -- by the
angle its place in the run gives it, about y, after translating it by the
wedge's start offset and its translation so far -- and asks of points in the
lab frame:

- whether they are inside (``isMicrocrystalAt``: ray parity against the
  rotated mesh);
- where a photon's line along +z first enters it (``getStartingZ``);
- how far an electron is from leaving it (``getIntersectionDistance``);
- which voxel they are in (``convertToPixelCoordinates``).

Here every point carries its own angle and offset, and every question is
answered in the crystal's own frame, which gives the same answers away from
points exactly on a face. RADDOSE-3D answered the last two wrongly for a
rotated crystal until its #44.

Lengths are in nm, as the Java tracks them; the mesh is in um. Points,
placements and the mesh are NumPy arrays, or tensors on one device
(:mod:`.ops`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyraddose_3d.backend.montecarlo.ops import namespace, on
from pyraddose_3d.backend.voxels import occupancy

if TYPE_CHECKING:
    from pyraddose_3d.backend.geometry import Mesh

__all__ = [
    "Placement",
    "entry_depth",
    "exit_distance",
    "inside",
    "voxel_indices",
]

#: Points per batch in the (points x triangles) tests, to bound memory.
_BATCH = 200_000


@dataclass(frozen=True)
class Placement:
    """How the crystal stands for each photon, and for its electrons.

    Attributes
    ----------
    angle : numpy.ndarray
        Rotation about y, radians.
    offset : numpy.ndarray
        ``(N, 3)``: the wedge's start position plus its translation at that
        angle, um.
    """

    angle: np.ndarray
    offset: np.ndarray

    def take(self, index: np.ndarray) -> Placement:
        """The placements of the chosen points."""
        return Placement(self.angle[index], self.offset[index])


def _to_crystal(points_nm: np.ndarray, placement: Placement) -> np.ndarray:
    """Lab-frame points (nm) into the unrotated crystal's frame (um).

    The inverse of ``setUpRotatedVertices``, which translates the mesh and
    then rotates it.
    """
    xp = namespace(points_nm)
    p = points_nm / 1000
    cos, sin = xp.cos(placement.angle), xp.sin(placement.angle)
    x = p[:, 0] * cos - p[:, 2] * sin
    z = p[:, 0] * sin + p[:, 2] * cos
    rotated = xp.stack([x, p[:, 1], z], axis=1)
    return rotated - placement.offset


def inside(mesh: Mesh, points_nm: np.ndarray, placement: Placement) -> np.ndarray:
    """``isMicrocrystalAt``: which points are inside the placed crystal."""
    xp = namespace(points_nm)
    mesh = on(xp, mesh)
    local = _to_crystal(points_nm, placement)
    result = xp.empty(len(local), dtype=bool)
    for start in range(0, len(local), _BATCH):
        chunk = local[start : start + _BATCH]
        result[start : start + _BATCH] = xp.asarray(occupancy(mesh, chunk))
    return result


def entry_depth(
    mesh: Mesh,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    placement: Placement,
    crossing: int = 0,
) -> np.ndarray:
    """``getStartingZ``: where each photon's line along +z meets it, nm.

    The line through ``(x, y, 0)``, in both directions, and its crossings in
    order: the first, the front, by default, or ``crossing=1``, the back.
    NaN where the line misses: since #48 RADDOSE-3D's is too, not 0.0,
    which a face at z = 0 also gives.
    """
    xp = namespace(x_nm)
    origin = _to_crystal(xp.stack([x_nm, y_nm, xp.zeros_like(x_nm)], axis=1), placement)
    direction = xp.stack(
        [-xp.sin(placement.angle), xp.zeros_like(x_nm), xp.cos(placement.angle)],
        axis=1,
    )
    corners = xp.asarray(on(xp, mesh).corners())
    depth = xp.zeros(len(origin))
    for start in range(0, len(origin), _BATCH):
        o = origin[start : start + _BATCH]
        d = direction[start : start + _BATCH]
        t, hit = _line_triangle(corners, o, d)
        t = xp.sort(xp.where(hit, t, xp.inf), axis=1)[:, crossing]
        depth[start : start + _BATCH] = xp.where(xp.isfinite(t), t * 1000, xp.nan)
    return depth


def _line_triangle(
    corners: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Moller-Trumbore for a line, both ways: its parameter at each triangle."""
    xp = namespace(origin)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    edge1, edge2 = b - a, c - a
    p = xp.cross(direction[:, None, :], edge2[None, :, :])
    det = _dot3(xp, "tk,ntk->nt", edge1, p)
    parallel = det == 0
    inverse = 1 / xp.where(parallel, 1.0, det)
    to_origin = origin[:, None, :] - a[None, :, :]
    u = _dot3(xp, "ntk,ntk->nt", to_origin, p) * inverse
    q = xp.cross(to_origin, edge1[None, :, :])
    v = _dot3(xp, "nk,ntk->nt", direction, q) * inverse
    t = _dot3(xp, "tk,ntk->nt", edge2, q) * inverse
    hit = ~parallel & (u >= 0) & (v >= 0) & (u + v <= 1)
    return t, hit


def _dot3(xp: object, subscripts: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``einsum``'s dot products of 3-vectors, on the last axis.

    NumPy's own ``einsum``, as it always was. On torch, the components are
    multiplied and added in order: its ``einsum`` goes by matrix products,
    which fuse and reorder them.
    """
    if xp is np:
        return np.asarray(np.einsum(subscripts, a, b))
    left, out = subscripts.split("->")[0].split(",")[0], subscripts.split("->")[1]
    # Broadcast ``a`` to ``b``'s axes: each of its letters where ``b`` has it.
    a = a[tuple(slice(None) if letter in left else None for letter in out + "k")]
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def exit_distance(
    mesh: Mesh, points_nm: np.ndarray, direction: np.ndarray, placement: Placement
) -> np.ndarray:
    """``getIntersectionDistance``: nm along ``direction`` to the crystal's surface.

    The first face of the crystal as it stands that the ray crosses: from
    inside, where it leaves; from outside, where it enters. 0 where it
    crosses none.
    """
    xp = namespace(points_nm)
    origin = _to_crystal(points_nm, placement)
    cos, sin = xp.cos(placement.angle), xp.sin(placement.angle)
    turned = xp.stack(
        [
            direction[:, 0] * cos - direction[:, 2] * sin,
            direction[:, 1],
            direction[:, 0] * sin + direction[:, 2] * cos,
        ],
        axis=1,
    )
    corners = xp.asarray(on(xp, mesh).corners())
    distance = xp.zeros(len(origin))
    for start in range(0, len(origin), _BATCH):
        t, hit = _line_triangle(
            corners, origin[start : start + _BATCH], turned[start : start + _BATCH]
        )
        t = xp.min(xp.where(hit & (t > 0), t, xp.inf), axis=1)
        distance[start : start + _BATCH] = xp.where(xp.isfinite(t), t * 1000, 0.0)
    return distance


def voxel_indices(
    points_nm: np.ndarray,
    placement: Placement,
    minimum_um: np.ndarray,
    pixels_per_micron: float,
) -> np.ndarray:
    """``convertToPixelCoordinates``: ``(N, 3)`` voxel indices.

    In the crystal's own frame; rounded half up, as ``StrictMath.round``;
    negative indices raised to 0.
    """
    xp = namespace(points_nm)
    local = _to_crystal(points_nm, placement)
    index = xp.floor((local - on(xp, minimum_um)) * pixels_per_micron + 0.5)
    return xp.astype(xp.maximum(index, 0), xp.int64)
