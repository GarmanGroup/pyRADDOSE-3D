"""Reading crystal shapes from .obj files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyraddose_3d.backend.wireframe import ObjError, read_obj
from pyraddose_3d.models.crystal import CuboidCrystal, PolyhedronCrystal
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import crystal_mesh, pixels_per_micron, solve

PRISM = Path(__file__).parent.parent / "data" / "hexagonal_prism.obj"
AVERAGE = {"method": "average"}


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "shape.obj"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_the_prism() -> None:
    mesh = read_obj(PRISM)
    assert mesh.vertices.shape == (12, 3)
    assert mesh.triangles.shape == (20, 3)
    assert mesh.triangles.min() == 0
    assert mesh.triangles.max() == 11
    np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), [30.0, 24.0, 20.784610])


def test_a_cuboid_from_a_file_is_the_built_in_cuboid(tmp_path: Path) -> None:
    """Same vertices and triangles in the same order, so the same dose exactly."""
    cuboid = CuboidCrystal(dimensions=(20.0, 16.0, 12.0), composition=AVERAGE)
    mesh = crystal_mesh(cuboid)
    lines = [f"v {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in mesh.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.triangles]
    path = _write(tmp_path, "\n".join(lines) + "\n")
    polyhedron = PolyhedronCrystal(model_file=str(path), composition=AVERAGE)

    def run(crystal: object) -> list[dict[str, float]]:
        experiment = Experiment.model_validate(
            {
                "crystal": {**crystal.model_dump(), "pixels_per_micron": 0.5},
                "beam": {
                    "type": "gaussian",
                    "flux": 2e12,
                    "energy": 12.1,
                    "fwhm": [20, 20],
                },
                "wedges": [{"start": 0, "end": 90, "exposure_time": 10}],
            }
        )
        return [summary.as_dict() for summary in solve(experiment)]

    assert run(polyhedron) == run(cuboid)


def test_is_more_tolerant_than_the_java_where_that_is_safe(tmp_path: Path) -> None:
    """Blank lines, tabs and v/vt/vn faces crash the Java's reader."""
    path = _write(
        tmp_path,
        "o thing\n\nv 0 0 0\nv\t1 0 0\nv 0 1 0\nv 0 0 1\n"
        "vn 0 0 1\nf 1/1/1 3/1/1 2/1/1\nf 1//1 2//1 4//1\nf 1 4 3\nf 2 3 4\n",
    )
    mesh = read_obj(path)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.triangles.tolist() == [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]


def test_a_vertex_with_too_few_numbers_is_skipped_as_the_java_skips_it(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "v 1 2\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert read_obj(path).vertices.shape == (3, 3)


@pytest.mark.parametrize(
    ("faces", "message"),
    [
        ("f 1 2 3 4\n", "export the mesh triangulated"),
        ("f -1 -2 -3\n", "absolute indices"),
        ("f 1 2 9\n", "only 4"),
        ("", "no faces"),
    ],
)
def test_refuses_what_the_java_would_misread(
    tmp_path: Path, faces: str, message: str
) -> None:
    path = _write(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n" + faces)
    with pytest.raises(ObjError, match=message):
        read_obj(path)


# --- the resolution ------------------------------------------------------------


def test_a_resolution_given_is_used() -> None:
    crystal = CuboidCrystal(
        dimensions=(4.0, 400.0, 400.0), pixels_per_micron=3.0, composition=AVERAGE
    )
    assert pixels_per_micron(crystal, crystal_mesh(crystal)) == 3.0


def test_the_default_is_ten_voxels_across_the_first_dimension() -> None:
    crystal = CuboidCrystal(dimensions=(25.0, 16.0, 12.0), composition=AVERAGE)
    assert pixels_per_micron(crystal, crystal_mesh(crystal)) == pytest.approx(0.4)


def test_the_default_is_capped_at_a_million_voxels() -> None:
    """Checked against RADDOSE-3D, which reports 0.86 um per voxel for this.

    Ten across 4 um would be 2.5 per um, and 10 x 1000 x 1000 voxels.
    """
    crystal = CuboidCrystal(dimensions=(4.0, 400.0, 400.0), composition=AVERAGE)
    resolution = pixels_per_micron(crystal, crystal_mesh(crystal))
    assert resolution == pytest.approx((1e6 / (4 * 400 * 400)) ** (1 / 3))
    assert f"{1 / resolution:.2f}" == "0.86"


def test_a_polyhedron_defaults_to_half_a_voxel_per_micron() -> None:
    crystal = PolyhedronCrystal(model_file=str(PRISM), composition=AVERAGE)
    assert pixels_per_micron(crystal, crystal_mesh(crystal)) == 0.5


def test_a_polyhedrons_placeholder_dimension_sets_its_default() -> None:
    crystal = PolyhedronCrystal(
        model_file=str(PRISM), dimensions=(16.0,), composition=AVERAGE
    )
    assert pixels_per_micron(crystal, crystal_mesh(crystal)) == pytest.approx(0.625)
