"""The voxel grid a crystal is sampled on, and which voxels are inside it.

Ports the grid construction in ``CrystalPolyhedron``'s constructor and
``calculateCrystalOccupancy``.

The grid spans the axis-aligned bounding box of the mesh, at a spacing of one
over ``pixels_per_micron``, with a voxel at each corner of the box -- so a
40 um edge at 0.5 voxels per micron gets 21 voxels, not 20. Voxel positions
are the grid points themselves, not cell centres.

No rotation is applied here. ``AngleP`` and ``AngleL`` are already baked into
the mesh, and this box spans that rotated mesh, so the coordinates are
already in the lab frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyraddose_3d.backend._array import array_namespace, astype
from pyraddose_3d.backend.geometry import triangle_planes

if TYPE_CHECKING:
    from pyraddose_3d.backend.geometry import Mesh

__all__ = ["VoxelGrid", "occupancy", "voxel_grid"]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""


@dataclass(frozen=True)
class VoxelGrid:
    """A regular grid of sample points spanning a crystal.

    Attributes
    ----------
    origin : Array
        ``(3,)`` position of voxel ``(0, 0, 0)``, the bounding box corner.
    shape : tuple[int, int, int]
        Voxel counts along x, y and z.
    pixels_per_micron : float
        Grid points per micrometre along each axis.
    """

    origin: Array
    shape: tuple[int, int, int]
    pixels_per_micron: float

    @property
    def count(self) -> int:
        """Total number of voxels."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def spacing(self) -> float:
        """Distance between adjacent grid points, in micrometres."""
        return 1.0 / self.pixels_per_micron

    @property
    def volume(self) -> float:
        """Volume of one voxel, in cubic micrometres."""
        return self.spacing**3

    def coordinates(self) -> Array:
        """Positions of every voxel.

        Returns
        -------
        Array
            ``(N, 3)`` in micrometres, ordered so that z varies fastest --
            matching the ``i, j, k`` loop nesting in the Java, so that a
            flattened result can be compared index for index.
        """
        xp = array_namespace(self.origin)
        # The dtype is spelled out because torch.arange yields int64, and
        # dividing an integer tensor by a Python float promotes to torch's
        # *default* dtype -- float32 -- not to the float64 of whatever it is
        # then added to. The grid would silently be single precision, and
        # every dose with it, at about 1e-7 relative.
        axes = [
            self.origin[axis]
            + xp.arange(self.shape[axis], dtype=self.origin.dtype)
            / self.pixels_per_micron
            for axis in range(3)
        ]
        grid = xp.meshgrid(*axes, indexing="ij")
        return xp.stack([g.reshape(-1) for g in grid], axis=-1)


def voxel_grid(mesh: Mesh, pixels_per_micron: float) -> VoxelGrid:
    """Build the grid that samples a crystal.

    Parameters
    ----------
    mesh : Mesh
        The crystal, with any ``AngleP`` and ``AngleL`` already applied.
    pixels_per_micron : float
        Grid points per micrometre.

    Returns
    -------
    VoxelGrid
        A grid spanning the mesh's bounding box.

    Notes
    -----
    The voxel count rounds half away from zero, which is what Java's
    ``StrictMath.round`` does. Python's built-in ``round`` rounds half to
    even, so a crystal whose extent lands exactly on a half voxel would come
    out one voxel smaller here -- a 21-voxel edge becoming 20, and every
    dose with it.
    """
    xp = array_namespace(mesh.vertices)
    lowest = xp.min(mesh.vertices, axis=0)
    highest = xp.max(mesh.vertices, axis=0)
    extent = highest - lowest

    shape = tuple(
        math.floor(float(extent[axis]) * pixels_per_micron + 0.5) + 1
        for axis in range(3)
    )
    return VoxelGrid(
        origin=lowest,
        shape=(shape[0], shape[1], shape[2]),
        pixels_per_micron=pixels_per_micron,
    )


def occupancy(mesh: Mesh, points: Array) -> Array:
    """Which points lie inside a closed mesh.

    Ports ``calculateCrystalOccupancy``: cast a ray along +z and count how
    many triangles it crosses. An odd count means inside.

    Parameters
    ----------
    mesh : Mesh
        A closed triangle mesh.
    points : Array
        ``(N, 3)`` positions.

    Returns
    -------
    Array
        ``(N,)`` boolean.

    Notes
    -----
    Deliberately not the same test as
    :func:`pyraddose_3d.backend.geometry.find_depth`, which traces the other
    way, excludes a zero-distance hit rather than keeping it, and collapses
    duplicate distances from shared edges. Those differences are in the Java
    and are kept: occupancy decides which voxels exist at all, so making it
    agree with ``find_depth`` would change which voxels are counted, not
    just their depths.
    """
    from pyraddose_3d.backend.geometry import _crosses_triangle

    xp = array_namespace(mesh.vertices, points)
    normals, offsets = triangle_planes(mesh)

    # Forward along +z, and the true ray parameter rather than its negation.
    towards_detector = normals[:, 2]
    projected = (
        points[:, :1] * normals[None, :, 0]
        + points[:, 1:2] * normals[None, :, 1]
        + points[:, 2:3] * normals[None, :, 2]
    )
    parallel = towards_detector == 0
    distances = -(projected + offsets) / xp.where(parallel, 1.0, towards_detector)

    # `>= 0` here against `> 0` in find_depth: the Java skips on `< 0`, so a
    # point exactly on a face counts as a crossing.
    reachable = ~parallel & (distances >= 0) & xp.isfinite(distances)
    hits = reachable & _crosses_triangle(mesh.corners(), points)

    # Parity, with no de-duplication: unlike find_depth, the Java does not
    # collapse equal distances here.
    crossings = xp.sum(astype(hits, xp.int64), axis=1)
    return (crossings % 2) == 1
