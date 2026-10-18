"""Dose contours: RADDOSE-3D's ``DoseState.R``, drawn with matplotlib.

RADDOSE-3D writes an R script that draws the final dose as nested surfaces --
everything above 0.1 MGy, above 20 and above 30 -- in a box the size of the
crystal, and saves five views of them; it needs R's ``rgl`` and ``misc3d``.
This draws the same surfaces, in the same colours, with matplotlib::

    pip install "pyRADDOSE-3D[plot]"

from a :class:`~pyraddose_3d.solve.Simulation`, or from a ``DoseState.csv``
written earlier, so a finished run can be drawn again without running it.

Matplotlib is imported only by the functions that draw, so the rest of the
package never needs it.

The surfaces
------------
:func:`isosurface` finds them by marching tetrahedra: each cube of eight
neighbouring voxel centres is cut into six tetrahedra, and the dose is
interpolated linearly along each edge that crosses the level. Neighbouring
cubes cut their shared face along the same diagonal, so the surface has no
cracks. The voxels outside the crystal are dose zero, and so is a layer
added round the grid, so every surface closes, as the R script's do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.outputs import final_dose_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from pyraddose_3d.solve import Simulation

__all__ = [
    "CONTOUR_COLOURS",
    "CONTOUR_LEVELS",
    "CONTOUR_OPACITIES",
    "VIEWS",
    "DoseField",
    "isosurface",
    "plot_dose_contours",
    "save_dose_contours",
]

#: ``DoseState.R``'s contour levels, in MGy.
CONTOUR_LEVELS = (0.1, 20.0, 30.0)
#: ``DoseState.R``'s colour for each level.
CONTOUR_COLOURS = ("lightblue", "darkblue", "red")
#: ``DoseState.R``'s opacity for each level.
CONTOUR_OPACITIES = (0.2, 0.5, 1.0)
#: ``DoseState.R``'s five views, ``(theta, phi)`` in degrees: theta round the
#: rotation axis, from looking along the beam, and phi up from the horizontal.
VIEWS = ((0, 0), (0, 45), (45, 45), (90, 45), (90, 90))

#: Corner ``c`` of a cube is ``(c & 1, c >> 1 & 1, c >> 2 & 1)`` voxels along.
_CORNERS = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)])
#: Freudenthal's six tetrahedra: each a path from corner 0 to corner 7, one
#: step along each axis, in each of the six orders.
_TETRAHEDRA = np.array(
    [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]]
)


@dataclass(frozen=True)
class DoseField:
    """The final dose on the voxel grid.

    Attributes
    ----------
    dose : np.ndarray
        ``(nx, ny, nz)``, in MGy; zero outside the crystal.
    x, y, z : np.ndarray
        The voxel centres along each axis, in micrometres: ``z`` along the
        beam, ``y`` the rotation axis.
    """

    dose: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @classmethod
    def from_simulation(cls, simulation: Simulation) -> DoseField:
        """The dose at the end of a run, on the run's whole grid.

        Parameters
        ----------
        simulation : Simulation
            From :func:`pyraddose_3d.solve.simulate`.

        Returns
        -------
        DoseField
            The dose ``DoseState.csv`` lists, and zero outside the crystal.
        """
        state = final_dose_state(simulation)
        grid = simulation.grid
        # float() reads a torch tensor's elements as well as NumPy's.
        origin = np.array([float(value) for value in grid.origin])
        axes = [origin[a] + grid.spacing * np.arange(grid.shape[a]) for a in range(3)]
        field = cls(np.zeros(grid.shape), *axes)
        indices = np.rint((state["coordinates"] - origin) / grid.spacing).astype(int)
        field.dose[tuple(indices.T)] = state["dose"]
        return field

    @classmethod
    def read_dose_state(cls, path: str | Path) -> DoseField:
        """Read a ``DoseState.csv``: ``x,y,z,dose,fluence,elastic`` per voxel.

        Parameters
        ----------
        path : str | Path
            A ``DoseState.csv``, from :func:`pyraddose_3d.outputs.write_outputs`
            or ``programs/dose/run_dose.py``.

        Returns
        -------
        DoseField
            The voxels it lists, on their grid.
        """
        table = np.loadtxt(path, delimiter=",", ndmin=2)
        return cls.from_voxels(table[:, :3], table[:, 3])

    @classmethod
    def from_voxels(cls, coordinates: np.ndarray, dose: np.ndarray) -> DoseField:
        """Put voxels listed by position, as ``DoseState.csv`` lists them, on a grid.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(N, 3)`` voxel centres in micrometres, on a regular grid.
        dose : np.ndarray
            ``(N,)``, in MGy.

        Returns
        -------
        DoseField
            The smallest grid holding them; a voxel not listed is zero.
        """
        coordinates = np.asarray(coordinates, dtype=float)
        axes, indices = [], []
        for axis in range(3):
            values = coordinates[:, axis]
            low = values.min()
            steps = np.diff(np.unique(values))
            # A written file holds single-precision positions, so the step is
            # taken from the gaps and each position rounded to it.
            step = steps.min() if steps.size else 1.0
            index = np.rint((values - low) / step).astype(int)
            axes.append(low + step * np.arange(index.max() + 1))
            indices.append(index)
        field = np.zeros([len(axis) for axis in axes])
        field[tuple(indices)] = dose
        return cls(field, *axes)


def isosurface(field: DoseField, level: float) -> np.ndarray:
    """The surface where the dose crosses ``level``.

    Parameters
    ----------
    field : DoseField
        The dose.
    level : float
        In MGy, above zero: everything outside the crystal is zero.

    Returns
    -------
    np.ndarray
        ``(T, 3, 3)`` triangles: corner, then ``x, y, z`` in micrometres. Each
        is wound so that its normal points from the higher dose to the lower.
        Empty if no voxel reaches the level.
    """
    if not level > 0:
        raise ValueError(f"a contour level must be above zero, not {level}")
    values = np.pad(field.dose, 1)
    axes = [_padded(field.x), _padded(field.y), _padded(field.z)]
    shape = np.array(values.shape) - 1
    corner_values = np.stack(
        [
            values[dx : dx + shape[0], dy : dy + shape[1], dz : dz + shape[2]]
            for dx, dy, dz in _CORNERS
        ],
        axis=-1,
    )
    above = corner_values >= level
    crossing = above.any(axis=-1) & ~above.all(axis=-1)
    cubes = np.argwhere(crossing)
    cube_values = corner_values[crossing]
    # (cubes, 8, 3): each corner's position.
    cube_points = np.stack(
        [axis[cubes[:, None, a] + _CORNERS[None, :, a]] for a, axis in enumerate(axes)],
        axis=-1,
    )

    triangles = [np.empty((0, 3, 3))]
    for tetrahedron in _TETRAHEDRA:
        triangles += _cut(
            cube_values[:, tetrahedron], cube_points[:, tetrahedron], level
        )
    return np.concatenate(triangles)


def _padded(axis: np.ndarray) -> np.ndarray:
    """An axis's positions with one more step at each end."""
    step = axis[1] - axis[0] if len(axis) > 1 else 1.0
    return np.concatenate([[axis[0] - step], axis, [axis[-1] + step]])


def _cut(values: np.ndarray, points: np.ndarray, level: float) -> list[np.ndarray]:
    """The triangles where ``level`` cuts some tetrahedra.

    ``values`` is ``(n, 4)`` and ``points`` ``(n, 4, 3)``. Sorted so the
    corners below the level come first, a tetrahedron with one corner above
    it is cut in a triangle round that corner, one with three in a triangle
    round the corner below, and one with two in a quadrilateral, drawn as
    two triangles.
    """
    above = values >= level
    count = above.sum(axis=1)
    order = np.argsort(above, axis=1, kind="stable")
    values = np.take_along_axis(values, order, axis=1)
    points = np.take_along_axis(points, order[:, :, None], axis=1)

    def edge(rows: np.ndarray, a: int, b: int) -> np.ndarray:
        fraction = (level - values[rows, a]) / (values[rows, b] - values[rows, a])
        return points[rows, a] + fraction[:, None] * (points[rows, b] - points[rows, a])

    out = []
    for below, rows in ((3, count == 1), (1, count == 3)):
        if not rows.any():
            continue
        lone = 3 if below == 3 else 0
        others = [c for c in range(4) if c != lone]
        triangle = np.stack([edge(rows, lone, other) for other in others], axis=1)
        out.append(_wound(triangle, points[rows], below))
    rows = count == 2
    if rows.any():
        quad = [edge(rows, 2, 0), edge(rows, 2, 1), edge(rows, 3, 1), edge(rows, 3, 0)]
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            triangle = np.stack([quad[a], quad[b], quad[c]], axis=1)
            out.append(_wound(triangle, points[rows], 2))
    return out


def _wound(triangles: np.ndarray, corners: np.ndarray, below: int) -> np.ndarray:
    """Wind each triangle so its normal points to its tetrahedron's lower dose.

    ``corners`` are the tetrahedron's, sorted so the first ``below`` are
    below the level.
    """
    downhill = corners[:, :below].mean(axis=1) - corners[:, below:].mean(axis=1)
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    backwards = np.einsum("ij,ij->i", normal, downhill) < 0
    triangles[backwards] = triangles[backwards][:, ::-1]
    return triangles


def plot_dose_contours(
    source: Simulation | DoseField | str | Path,
    levels: Sequence[float] = CONTOUR_LEVELS,
    colours: Sequence[Any] = CONTOUR_COLOURS,
    opacities: Sequence[float] = CONTOUR_OPACITIES,
    view: tuple[float, float] = VIEWS[2],
    ax: Any = None,
) -> Axes:
    """Draw the dose as ``DoseState.R`` does: a surface at each level.

    Parameters
    ----------
    source : Simulation | DoseField | str | Path
        A run, its dose, or the path of a ``DoseState.csv``.
    levels : Sequence[float]
        Doses in MGy, each above zero. ``DoseState.R``'s by default.
    colours : Sequence
        A matplotlib colour for each level.
    opacities : Sequence[float]
        Each level's opacity, from 0 to 1.
    view : tuple[float, float]
        ``(theta, phi)`` in degrees, as in :data:`VIEWS`.
    ax : Axes3D, optional
        A 3D axes to draw in. A new figure's, by default.

    Returns
    -------
    Axes
        The axes drawn in. The rotation axis, y, points up, and the beam
        runs along z. A box marks the crystal's extent, as the R script's
        does.
    """
    from matplotlib.colors import LightSource, to_rgba
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if not len(levels) == len(colours) == len(opacities):
        raise ValueError("give a colour and an opacity for each level")
    field = _field(source)
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.figure(figsize=(6, 5)).add_subplot(projection="3d")

    surfaces, faces, legend = [], [], []
    for level, colour, opacity in zip(levels, colours, opacities, strict=True):
        triangles = isosurface(field, level)
        surfaces.append(triangles)
        faces.append(np.tile(to_rgba(colour, opacity), (len(triangles), 1)))
        legend.append(Patch(color=to_rgba(colour, opacity), label=f"{level:g} MGy"))
    triangles = np.concatenate(surfaces)
    if len(triangles):
        # One collection, so that matplotlib sorts every level's triangles by
        # depth together and a surface inside another is drawn behind it.
        ax.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolors=np.concatenate(faces),
                edgecolors=np.concatenate(faces),
                linewidths=0,
                shade=True,
                lightsource=LightSource(azdeg=315, altdeg=45),
            )
        )
    _draw_box(ax, field)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_aspect("equal")
    theta, phi = view
    ax.view_init(elev=phi, azim=theta, vertical_axis="y")
    ax.legend(handles=legend, title="dose above", loc="upper left", fontsize="small")
    return ax


def _field(source: Simulation | DoseField | str | Path) -> DoseField:
    if isinstance(source, DoseField):
        return source
    if isinstance(source, str | Path):
        return DoseField.read_dose_state(source)
    return DoseField.from_simulation(source)


def _draw_box(ax: Any, field: DoseField) -> None:
    """The twelve edges of the box round the voxel centres, in grey."""
    low = [field.x[0], field.y[0], field.z[0]]
    high = [field.x[-1], field.y[-1], field.z[-1]]
    corners = np.array(
        [[(low, high)[c >> a & 1][a] for a in range(3)] for c in range(8)]
    )
    for a in range(3):
        for c in range(8):
            if not c >> a & 1:
                ax.plot(*corners[[c, c | 1 << a]].T, color="grey", linewidth=0.8)
    for a, axis in enumerate((field.x, field.y, field.z)):
        # Room for the outermost surface, half a voxel outside the crystal.
        margin = (axis[1] - axis[0]) if len(axis) > 1 else 1.0
        getattr(ax, f"set_{'xyz'[a]}lim")(low[a] - margin, high[a] + margin)


def save_dose_contours(
    source: Simulation | DoseField | str | Path,
    prefix: str = "output-",
    views: Sequence[tuple[float, float]] = VIEWS,
    **options: Any,
) -> list[Path]:
    """Save a picture of the contours from each view, as ``DoseState.R`` does.

    Parameters
    ----------
    source : Simulation | DoseField | str | Path
        A run, its dose, or the path of a ``DoseState.csv``.
    prefix : str
        Prepended to each file name, as ``write_outputs``'s is.
    views : Sequence[tuple[float, float]]
        ``(theta, phi)`` for each picture; ``DoseState.R``'s five by default.
    **options
        ``levels``, ``colours`` and ``opacities``, as for
        :func:`plot_dose_contours`.

    Returns
    -------
    list[Path]
        ``DoseContours_<theta>_<phi>.png`` for each view -- the R script's
        ``plot_<theta>_<phi>.png`` -- after the prefix.
    """
    from matplotlib.figure import Figure

    field = _field(source)
    paths = []
    for theta, phi in views:
        figure = Figure(figsize=(6, 5))
        ax = figure.add_subplot(projection="3d")
        plot_dose_contours(field, view=(theta, phi), ax=ax, **options)
        path = Path(f"{prefix}DoseContours_{theta:g}_{phi:g}.png")
        figure.savefig(path, dpi=150)
        paths.append(path)
    return paths
