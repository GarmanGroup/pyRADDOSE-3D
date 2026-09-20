"""Crystal meshes shared by the geometry and exposure tests."""

from __future__ import annotations

import numpy as np

from pyraddose_3d.backend.geometry import Mesh

#: A cuboid as ``CrystalCuboid`` builds one: eight corners and twelve
#: triangles, with RADDOSE-3D's one-based indices made zero-based.
CUBOID_TRIANGLES = (
    np.array(
        [
            [1, 3, 2], [4, 3, 1], [3, 6, 2], [7, 6, 3],
            [2, 5, 1], [2, 6, 5], [4, 8, 3], [8, 7, 3],
            [4, 1, 8], [1, 5, 8], [8, 5, 7], [7, 5, 6],
        ]
    )
    - 1
)


def cuboid(x: float = 40.0, y: float = 30.0, z: float = 20.0) -> Mesh:
    """A cuboid mesh centred on the origin, in micrometres."""
    half_x, half_y, half_z = x / 2, y / 2, z / 2
    vertices = np.array(
        [
            [-half_x, -half_y, half_z], [-half_x, -half_y, -half_z],
            [-half_x, half_y, -half_z], [-half_x, half_y, half_z],
            [half_x, -half_y, half_z], [half_x, -half_y, -half_z],
            [half_x, half_y, -half_z], [half_x, half_y, half_z],
        ]
    )
    return Mesh(vertices, CUBOID_TRIANGLES)
