"""Dose contours: the surfaces, the field they are cut from, and the pictures."""

from __future__ import annotations

import collections
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import dose_state_csv, final_dose_state
from pyraddose_3d.plotting import (
    CONTOUR_LEVELS,
    VIEWS,
    DoseField,
    isosurface,
    plot_dose_contours,
    save_dose_contours,
)
from pyraddose_3d.solve import Simulation, simulate

INSULIN = Path(__file__).parent.parent / "docs" / "examples" / "inputs" / "insulin.txt"


@functools.cache
def insulin() -> Simulation:
    return simulate(Experiment.from_file(INSULIN))


def _sphere(radius: float, step: float = 1.0) -> DoseField:
    """Dose falling linearly from 30 MGy at the centre to zero at ``radius``."""
    axis = np.arange(-radius - 2, radius + 2 + step / 2, step)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    return DoseField(np.clip(30 * (1 - r / radius), 0, None), axis, axis, axis)


def _volume(triangles: np.ndarray) -> float:
    """Signed volume enclosed: positive when the normals point outwards."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def _area(triangles: np.ndarray) -> float:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)


def _open_edges(triangles: np.ndarray) -> int:
    """Edges not shared by exactly two triangles: none, on a closed surface."""
    corners = [tuple(point) for point in np.round(triangles.reshape(-1, 3), 9)]
    edges = collections.Counter(
        frozenset((corners[3 * t + a], corners[3 * t + (a + 1) % 3]))
        for t in range(len(triangles))
        for a in range(3)
    )
    return sum(1 for count in edges.values() if count != 2)


# --- the surfaces --------------------------------------------------------


def test_a_sphere_s_contour_is_the_sphere() -> None:
    """Dose 30 (1 - r/10) crosses 14 MGy at r = 5.33. On a 1 um grid every
    corner lies on or just inside it, as a chord does, within 1.1%; the area
    comes to 0.9% short, and the volume 1.8%."""
    radius = 10 * (1 - 14 / 30)
    triangles = isosurface(_sphere(10.0), 14.0)
    radii = np.linalg.norm(triangles.reshape(-1, 3), axis=1)
    assert np.all((radii > 0.985 * radius) & (radii < radius * (1 + 1e-12)))
    assert _area(triangles) == pytest.approx(4 * np.pi * radius**2, rel=0.015)
    assert _volume(triangles) == pytest.approx(4 / 3 * np.pi * radius**3, rel=0.025)


def test_a_surface_has_no_cracks_and_faces_out() -> None:
    """Every edge is shared by two triangles, and every triangle's normal
    points from the higher dose to the lower: the signed volume is positive."""
    for level in (1.0, 15.0, 25.0):
        triangles = isosurface(_sphere(7.3, step=0.7), level)
        assert len(triangles) > 0
        assert _open_edges(triangles) == 0
        assert _volume(triangles) > 0


def test_a_surface_at_the_edge_of_the_grid_closes() -> None:
    """Outside the grid is dose zero, as outside the crystal is: a dose filling
    the grid is closed, half a voxel beyond the outermost centres. (Its
    corners are cut off: a straight line between the corner voxel and the
    zero beyond crosses the level there.)"""
    x, y, z = np.arange(5.0), np.arange(6.0), np.arange(7.0)
    triangles = isosurface(DoseField(np.full((5, 6, 7), 10.0), x, y, z), 5.0)
    assert _open_edges(triangles) == 0
    assert _volume(triangles) > 0
    corners = triangles.reshape(-1, 3)
    assert np.array_equal(corners.min(axis=0), [-0.5, -0.5, -0.5])
    assert np.array_equal(corners.max(axis=0), [4.5, 5.5, 6.5])


def test_a_level_above_the_highest_dose_has_no_surface() -> None:
    assert isosurface(_sphere(5.0), 31.0).shape == (0, 3, 3)


@pytest.mark.parametrize("level", [0.0, -1.0])
def test_a_level_of_zero_or_less_is_refused(level: float) -> None:
    """Everything outside the crystal is zero, so it would wrap the grid."""
    with pytest.raises(ValueError, match="above zero"):
        isosurface(_sphere(5.0), level)


# --- the field -----------------------------------------------------------


def test_the_field_is_the_simulation_s_final_dose() -> None:
    simulation = insulin()
    field = DoseField.from_simulation(simulation)
    state = final_dose_state(simulation)
    assert field.dose.shape == simulation.grid.shape
    assert np.count_nonzero(field.dose) == np.count_nonzero(state["dose"])
    assert field.dose.sum() == pytest.approx(state["dose"].sum())
    assert field.dose.max() == pytest.approx(state["dose"].max())


def test_a_dose_state_csv_reads_back_as_the_dose(tmp_path: Path) -> None:
    """The file lists only the voxels inside the crystal, in single precision."""
    simulation = insulin()
    path = tmp_path / "DoseState.csv"
    path.write_text(dose_state_csv(simulation), encoding="utf-8")
    state = final_dose_state(simulation)
    expected = DoseField.from_voxels(state["coordinates"], state["dose"])

    read = DoseField.read_dose_state(path)

    assert read.dose.shape == expected.dose.shape
    assert np.allclose(read.dose, expected.dose, rtol=1e-6)
    for got, want in zip(
        (read.x, read.y, read.z), (expected.x, expected.y, expected.z), strict=True
    ):
        assert np.allclose(got, want, atol=1e-4)


# --- the pictures --------------------------------------------------------


@pytest.fixture
def matplotlib() -> None:
    """The drawing tests need the ``plot`` extra; none of them opens a window."""
    pytest.importorskip("matplotlib")


def test_importing_plotting_does_not_import_matplotlib() -> None:
    """Matplotlib is the ``plot`` extra: only drawing needs it."""
    code = (
        "import json, sys\nimport pyraddose_3d.plotting\n"
        "print(json.dumps('matplotlib' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) is False


def test_each_level_is_drawn_in_one_collection(matplotlib: None) -> None:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    field = DoseField.from_simulation(insulin())
    ax = Figure().add_subplot(projection="3d")
    plot_dose_contours(field, ax=ax)

    (surfaces,) = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    expected = sum(len(isosurface(field, level)) for level in CONTOUR_LEVELS)
    assert len(surfaces.get_facecolor()) == expected
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["0.1 MGy", "20 MGy", "30 MGy"]
    assert len(ax.lines) == 12  # the box


def test_a_level_needs_a_colour_and_an_opacity(matplotlib: None) -> None:
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(projection="3d")
    with pytest.raises(ValueError, match="colour and an opacity"):
        plot_dose_contours(_sphere(5.0), levels=(1.0, 2.0), ax=ax)


def test_the_five_views_are_saved(matplotlib: None, tmp_path: Path) -> None:
    paths = save_dose_contours(_sphere(5.0), prefix=str(tmp_path / "out-"))
    assert [path.name for path in paths] == [
        f"out-DoseContours_{theta}_{phi}.png" for theta, phi in VIEWS
    ]
    for path in paths:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
