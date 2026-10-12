"""Reading a crystal's shape from a Wavefront ``.obj`` file.

Ports ``ImportWireframeObj``: vertices from ``v`` lines, triangles from ``f``
lines, everything else ignored. The mesh is used as given -- not centred,
not scaled -- so its coordinates are micrometres in the crystal's own frame.

Where the Java fails on a file, this reads it; where the Java silently reads
a different shape, this refuses:

- A blank line, a one-character line or a tab between numbers crashes the
  Java; here they are ordinary whitespace.
- ``f 1/2/3`` (vertex, texture and normal indices) crashes the Java, which
  only strips ``//``; here everything after the first ``/`` is dropped.
- A face with more than three vertices is silently cut to its first three
  by the Java, which leaves holes in the solid and a wrong dose. Refused:
  export the mesh triangulated.
- A negative, relative index is refused, as is one past the last vertex.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pyraddose_3d.backend.geometry import Mesh

if TYPE_CHECKING:
    import os

__all__ = ["ObjError", "read_obj"]


class ObjError(ValueError):
    """An ``.obj`` file this cannot turn into a closed triangle mesh."""


def _index(token: str, line_number: int) -> int:
    text = token.split("/", 1)[0]
    try:
        index = int(text)
    except ValueError:
        raise ObjError(f"line {line_number}: {token!r} is not a vertex index") from None
    if index < 1:
        raise ObjError(
            f"line {line_number}: vertex index {index}; only positive, absolute "
            "indices are supported, as in RADDOSE-3D"
        )
    return index


def read_obj(path: str | os.PathLike[str]) -> Mesh:
    """Read a triangulated ``.obj`` file.

    Parameters
    ----------
    path : str | os.PathLike[str]
        The file.

    Returns
    -------
    Mesh
        Vertices in micrometres as written, and zero-based triangles in the
        file's order and winding.

    Raises
    ------
    ObjError
        If a face is not a triangle, an index is out of range, a number does
        not parse, or there are no faces.
    """
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "v":
            try:
                vertices.append([float(value) for value in tokens[1:4]])
            except ValueError:
                raise ObjError(f"line {line_number}: {line.strip()!r}") from None
            if len(vertices[-1]) < 3:
                # The Java skips a vertex line with fewer than three numbers.
                vertices.pop()
        elif tokens[0] == "f":
            corners = tokens[1:]
            if len(corners) < 3:
                continue
            if len(corners) > 3:
                raise ObjError(
                    f"line {line_number}: a face with {len(corners)} vertices. "
                    "RADDOSE-3D keeps only the first three of each face, which "
                    "leaves holes in the crystal; export the mesh triangulated."
                )
            triangles.append([_index(corner, line_number) for corner in corners])

    if not triangles:
        raise ObjError(f"{path}: no faces")
    faces = np.array(triangles, dtype=np.int64) - 1
    if faces.max() >= len(vertices):
        raise ObjError(
            f"{path}: a face uses vertex {faces.max() + 1}, but there are only "
            f"{len(vertices)}"
        )
    return Mesh(np.array(vertices, dtype=np.float64), faces)
