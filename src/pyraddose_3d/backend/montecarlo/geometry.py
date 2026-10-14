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

Here every point carries its own angle and offset. The first three are
answered in the crystal's own frame, which gives the same answers away from
points exactly on a face. The Java's two quirks are kept: the distance to
leaving uses the planes of the *unrotated* crystal in the lab frame, and the
voxel lookup translates before rotating where the rotation of the mesh
translates first and rotates after. Both are listed in ``UPSTREAM_FIXES.md``.

Lengths are in nm, as the Java tracks them; the mesh is in um.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyraddose_3d.backend.geometry import Mesh, triangle_planes
from pyraddose_3d.backend.voxels import occupancy

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
    p = points_nm / 1000
    cos, sin = np.cos(placement.angle), np.sin(placement.angle)
    x = p[:, 0] * cos - p[:, 2] * sin
    z = p[:, 0] * sin + p[:, 2] * cos
    rotated = np.stack([x, p[:, 1], z], axis=1)
    return rotated - placement.offset


def inside(mesh: Mesh, points_nm: np.ndarray, placement: Placement) -> np.ndarray:
    """``isMicrocrystalAt``: which points are inside the placed crystal."""
    local = _to_crystal(points_nm, placement)
    result = np.empty(len(local), dtype=bool)
    for start in range(0, len(local), _BATCH):
        chunk = local[start : start + _BATCH]
        result[start : start + _BATCH] = np.asarray(occupancy(mesh, chunk))
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
    0.0 where the line misses, as in the Java, which takes 0.0 to mean a
    miss.
    """
    origin = _to_crystal(np.stack([x_nm, y_nm, np.zeros_like(x_nm)], axis=1), placement)
    direction = np.stack(
        [-np.sin(placement.angle), np.zeros_like(x_nm), np.cos(placement.angle)],
        axis=1,
    )
    corners = np.asarray(mesh.corners())
    depth = np.zeros(len(origin))
    for start in range(0, len(origin), _BATCH):
        o = origin[start : start + _BATCH]
        d = direction[start : start + _BATCH]
        t, hit = _line_triangle(corners, o, d)
        t = np.sort(np.where(hit, t, np.inf), axis=1)[:, crossing]
        depth[start : start + _BATCH] = np.where(np.isfinite(t), t * 1000, 0.0)
    return depth


def _line_triangle(
    corners: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Moller-Trumbore for a line, both ways: its parameter at each triangle."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    edge1, edge2 = b - a, c - a
    p = np.cross(direction[:, None, :], edge2[None, :, :])
    det = np.einsum("tk,ntk->nt", edge1, p)
    parallel = det == 0
    inverse = 1 / np.where(parallel, 1.0, det)
    to_origin = origin[:, None, :] - a[None, :, :]
    u = np.einsum("ntk,ntk->nt", to_origin, p) * inverse
    q = np.cross(to_origin, edge1[None, :, :])
    v = np.einsum("nk,ntk->nt", direction, q) * inverse
    t = np.einsum("tk,ntk->nt", edge2, q) * inverse
    hit = ~parallel & (u >= 0) & (v >= 0) & (u + v <= 1)
    return t, hit


def exit_distance(
    mesh: Mesh, points_nm: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """``getIntersectionDistance``: nm along ``direction`` to the nearest plane.

    The planes of the unrotated crystal, in the lab frame, whether or not
    the crossing lies within its triangle, and nothing at a distance of 0 or
    behind. 0 where no plane is ahead.
    """
    normals, offsets = (np.asarray(v) for v in triangle_planes(mesh))
    origin = points_nm / 1000
    along = direction @ normals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(origin @ normals.T + offsets) / along
    usable = np.isfinite(t) & (t > 0)
    nearest = np.where(usable, t, np.inf).min(axis=1)
    return np.where(np.isfinite(nearest), nearest, 0.0) * 1000


def voxel_indices(
    points_nm: np.ndarray,
    placement: Placement,
    minimum_um: np.ndarray,
    pixels_per_micron: float,
) -> np.ndarray:
    """``convertToPixelCoordinates``: ``(N, 3)`` voxel indices.

    Translated back, then rotated by ``2 pi - angle``; rounded half up, as
    ``StrictMath.round``; negative indices raised to 0.
    """
    shifted = points_nm - 1000 * placement.offset
    angle = 2 * np.pi - placement.angle
    cos, sin = np.cos(angle), np.sin(angle)
    x = shifted[:, 0] * cos + shifted[:, 2] * sin
    z = -1 * shifted[:, 0] * sin + shifted[:, 2] * cos
    local = np.stack([x, shifted[:, 1], z], axis=1)
    index = np.floor(((local / 1000) - minimum_um) * pixels_per_micron + 0.5)
    return np.maximum(index, 0).astype(np.int64)
