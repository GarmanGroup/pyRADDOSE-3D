"""Crystal geometry: how much crystal a beam crossed to reach a voxel.

Ports ``CrystalPolyhedron.findDepth`` and the mesh handling it rests on. Every
crystal shape in RADDOSE-3D is a triangle mesh -- a cuboid is twelve
triangles, a cylinder a many-sided prism -- so one implementation covers all
of them.

What ``depth`` means
--------------------
The beam travels along +z. :func:`find_depth` returns the path length through
crystal *behind* a voxel: the material the beam had to cross to arrive there,
which is what attenuates it. A voxel on the entry face has depth zero; one on
the far face has the full thickness.

The Java reaches this by computing each ray-plane parameter and then negating
it, which reads as though it were tracing forwards. It is not, and the sign is
worth stating plainly rather than leaving in a double negative.

Vectorisation
-------------
The Java loops over triangles per voxel, collecting hits into a list, sorting,
de-duplicating and walking the sorted list in pairs. None of that survives
here: every step is a whole-array operation over (voxels x triangles), which
is what makes the exposure loop affordable and what lets gradients flow.

Two observations make it work. First, the ray direction is the z axis, so the
intersection point shares the voxel's x and y -- the polygon test does not
depend on the distance at all, and can be evaluated once per voxel/triangle
pair. Second, the Java's "first crossing, then every other gap" walk

.. code-block:: text

    depth = d0 + (d2 - d1) + (d4 - d3) + ...

is an alternating sum, ``d0 - d1 + d2 - d3 + ...``, so it needs no loop and no
compaction: a sign from each hit's rank among the hits is enough.

Differentiability
-----------------
Differentiable in the vertex positions, the rotation angle and the voxel
coordinates. It is piecewise: a voxel's depth jumps as a ray crosses a
silhouette edge, and as the parity of the crossing count flips. Those are
genuine discontinuities in the model, not artefacts, and the gradient either
side of them is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyraddose_3d.backend._array import array_namespace, astype, promote, sort

__all__ = ["Mesh", "find_depth", "triangle_planes"]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""


@dataclass(frozen=True)
class Mesh:
    """A closed triangle mesh.

    Attributes
    ----------
    vertices : Array
        ``(V, 3)`` vertex positions in micrometres.
    triangles : Array
        ``(T, 3)`` integer indices into ``vertices``. **Zero-based**, unlike
        the one-based indices in RADDOSE-3D's shape definitions; converting
        is the reader's job, not this module's.
    """

    vertices: Array
    triangles: Array

    def corners(self) -> Array:
        """Vertex positions gathered per triangle.

        Returns
        -------
        Array
            ``(T, 3, 3)``: triangle, corner, coordinate.
        """
        return self.vertices[self.triangles]

    def rotated(
        self,
        angle: Array,
        translation: Array | None = None,
    ) -> Mesh:
        """Translate, then rotate about the y axis.

        The order matters and matches ``setupDepthFinding``: the goniometer
        translation is applied in the crystal's own frame and the whole thing
        is then turned.

        Parameters
        ----------
        angle : Array
            Rotation in radians.
        translation : Array, optional
            ``(3,)`` displacement in micrometres, applied before the
            rotation. This is the wedge's start position plus whatever its
            per-degree translation has accumulated.

        Returns
        -------
        Mesh
            A new mesh. The triangle indices are shared, not copied.
        """
        xp, (vertices, angle) = promote(self.vertices, angle)
        if translation is not None:
            _, (vertices, shift) = promote(vertices, translation)
            vertices = vertices + shift

        cos = xp.cos(angle)
        sin = xp.sin(angle)
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        return Mesh(
            vertices=xp.stack([x * cos + z * sin, y, -x * sin + z * cos], axis=-1),
            triangles=self.triangles,
        )


def triangle_planes(mesh: Mesh) -> tuple[Array, Array]:
    """Unit normal and origin offset of each triangle's plane.

    Parameters
    ----------
    mesh : Mesh
        The mesh.

    Returns
    -------
    tuple[Array, Array]
        ``(T, 3)`` unit normals, and ``(T,)`` offsets ``d`` such that a point
        ``p`` lies on triangle ``t``'s plane when ``dot(n, p) + d == 0``.

    Notes
    -----
    The normal is ``(p2 - p1) x (p3 - p1)`` normalised, so its direction
    follows the winding order of the mesh. :func:`find_depth` does not care
    which way it points -- the sign cancels -- but anything that does should
    know it is winding-dependent.
    """
    xp = array_namespace(mesh.vertices)
    corners = mesh.corners()
    first, second, third = corners[:, 0], corners[:, 1], corners[:, 2]

    normals = xp.linalg.cross(second - first, third - first)
    lengths = xp.sqrt(xp.sum(normals**2, axis=-1, keepdims=True))
    normals = normals / lengths

    offsets = -xp.sum(normals * first, axis=-1)
    return normals, offsets


def _crosses_triangle(corners: Array, points: Array) -> Array:
    """Whether each point lies inside each triangle, projected onto xy.

    The crossing-number test of Franklin's PNPOLY, evaluated for all three
    edges at once. A point is inside when an odd number of edges lie above
    it and to its right.

    Parameters
    ----------
    corners : Array
        ``(T, 3, 3)`` triangle corners.
    points : Array
        ``(N, 3)`` points. Only x and y are read: the ray runs along z, so
        every intersection shares the voxel's x and y.

    Returns
    -------
    Array
        ``(N, T)`` boolean.
    """
    xp = array_namespace(corners, points)
    point_x = points[:, 0][:, None]
    point_y = points[:, 1][:, None]

    inside = None
    for index in range(3):
        # j is the previous corner, wrapping round, as PNPOLY walks edges.
        current, previous = corners[:, index], corners[:, index - 1]
        current_x, current_y = current[:, 0], current[:, 1]
        previous_x, previous_y = previous[:, 0], previous[:, 1]

        straddles = (current_y > point_y) != (previous_y > point_y)
        # Only evaluated where it straddles, but computed everywhere, so the
        # denominator has to be kept away from zero: where it would be zero
        # the edge is horizontal and cannot straddle.
        height = xp.where(previous_y - current_y == 0, 1.0, previous_y - current_y)
        boundary = (previous_x - current_x) * (point_y - current_y) / height + current_x

        crossing = straddles & (point_x < boundary)
        inside = crossing if inside is None else (inside ^ crossing)

    return inside


def find_depth(mesh: Mesh, voxels: Array) -> Array:
    """Path length through the crystal behind each voxel.

    Parameters
    ----------
    mesh : Mesh
        The crystal, already rotated into the frame of the exposure.
    voxels : Array
        ``(N, 3)`` voxel positions in micrometres.

    Returns
    -------
    Array
        ``(N,)`` depths in micrometres. Zero for a voxel outside the crystal.

    Notes
    -----
    A voxel is outside when the ray meets the surface an even number of times
    -- including zero. The Java treats an even count as "outside" and returns
    zero, which is also how a numerically awkward case fails, and that is
    reproduced here.

    Hits at exactly equal distances are collapsed to one. A ray through a
    shared triangle edge registers against both triangles, and counting it
    twice would flip the parity and wrongly report the voxel as outside.
    """
    xp, (vertices, voxels) = promote(mesh.vertices, voxels)
    mesh = Mesh(vertices=vertices, triangles=mesh.triangles)

    normals, offsets = triangle_planes(mesh)

    # Distance from the voxel back along the beam to each triangle's plane.
    # Positive means behind the voxel, which is where the beam came from.
    #
    # Summed component by component rather than with a matmul. The numerator
    # is a difference of two nearly equal numbers whenever a voxel sits on a
    # face, and the denominator is near zero whenever a face is nearly
    # parallel to the beam -- so the quotient amplifies the last bits of the
    # numerator enormously. A matmul is free to reassociate, and BLAS picks
    # different orders for different batch shapes, which made a voxel's depth
    # depend on which other voxels happened to be passed alongside it. This
    # matches the order of the Java's dotProduct, and is the same for one
    # voxel as for a million.
    towards_source = normals[:, 2]
    projected = (
        voxels[:, :1] * normals[None, :, 0]
        + voxels[:, 1:2] * normals[None, :, 1]
        + voxels[:, 2:3] * normals[None, :, 2]
    )

    # A face exactly parallel to the beam is never hit. Dividing by its zero
    # z component says so, but an infinity here comes back through the
    # gradient as a NaN even after it has been masked out of the result --
    # the mask multiplies it by zero, and zero times infinity is not zero.
    # So the denominator is held away from zero and the face excluded
    # explicitly, which gives the same answer without ever forming one.
    parallel = towards_source == 0
    distances = (projected + offsets) / xp.where(parallel, 1.0, towards_source)

    reachable = ~parallel & (distances > 0) & xp.isfinite(distances)
    hits = reachable & _crosses_triangle(mesh.corners(), voxels)

    # Sort the hits to the front and everything else behind. Infinity is the
    # sentinel, so that "is this a hit" can be read back off the sorted array
    # rather than sorted separately and assumed to line up.
    infinity = xp.asarray(float("inf"), dtype=distances.dtype)
    ordered = sort(xp.where(hits, distances, infinity), axis=1)
    counted = xp.isfinite(ordered)
    # Nothing downstream may touch the sentinel, or an infinity reaches the
    # gradient as a NaN even where it is masked out of the result.
    ordered = xp.where(counted, ordered, 0.0)

    # A ray through a shared edge hits both triangles at the same distance.
    same_as_previous = ordered[:, 1:] == ordered[:, :-1]
    keep = xp.concatenate([counted[:, :1], counted[:, 1:] & ~same_as_previous], axis=1)

    # depth = d0 - d1 + d2 - d3 + ..., which is the Java's
    # "first crossing, then every other gap" written without the loop.
    rank = xp.cumsum(astype(keep, ordered.dtype), axis=1) - 1
    sign = xp.where(rank % 2 == 0, 1.0, -1.0)
    depth = xp.sum(xp.where(keep, sign * ordered, 0.0), axis=1)

    crossings = xp.sum(astype(keep, ordered.dtype), axis=1)
    inside = (crossings % 2) == 1
    return xp.where(inside, depth, 0.0)
