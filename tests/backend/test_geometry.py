"""Tests for crystal geometry and depth finding.

``find_depth`` is the hardest thing in the port to get right: the Java loops
over triangles per voxel, and the vectorised form has to agree with it exactly
while computing a whole voxel grid at once. Most of what is checked here
exists because getting that equivalence wrong is easy and quiet.
"""

import csv
import gzip
import math
import pathlib
from collections import defaultdict

import numpy as np
import pytest

from pyraddose_3d.backend.geometry import (
    Mesh,
    _crosses_triangle,
    find_depth,
    triangle_planes,
)

REFERENCE = pathlib.Path(__file__).parent / "data" / "find_depth.tsv.gz"

#: A 40 x 30 x 20 um cuboid, as CrystalCuboid builds one: eight corners and
#: twelve triangles, with RADDOSE-3D's one-based indices made zero-based.
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
    """A cuboid mesh centred on the origin."""
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


def _load_reference() -> dict:
    cases: dict = defaultdict(
        lambda: {"verts": {}, "tris": {}, "wedge": None, "depths": []}
    )
    with gzip.open(REFERENCE, "rt", encoding="ascii") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            tag, case_id = row[0], row[1]
            case = cases[case_id]
            if tag == "VERT":
                case["verts"][int(row[2])] = [float(v) for v in row[3:6]]
            elif tag == "TRI":
                case["tris"][int(row[2])] = [int(v) - 1 for v in row[3:6]]
            elif tag == "WEDGE":
                case["wedge"] = [float(v) for v in row[2:5]]
            elif tag == "DEPTH":
                case["depths"].append([float(v) for v in row[2:7]])
    return dict(cases)


REFERENCE_CASES = _load_reference()
CASE_IDS = sorted(REFERENCE_CASES)


def _shares_an_edge(mesh: Mesh, points: np.ndarray) -> np.ndarray:
    """Which of these rays meet two triangles at the very same distance.

    Used to point the de-duplication test at cases that actually exercise
    it. Whether a given ray does depends on rounding -- the two distances
    have to come out bitwise equal -- so the points cannot simply be written
    down, they have to be found.
    """
    normals, offsets = triangle_planes(mesh)
    parallel = normals[:, 2] == 0
    distances = (points @ normals.T + offsets) / np.where(
        parallel, 1.0, normals[:, 2]
    )
    hits = (
        ~parallel
        & (distances > 0)
        & np.isfinite(distances)
        & _crosses_triangle(mesh.corners(), points)
    )
    ordered = np.sort(np.where(hits, distances, np.inf), axis=1)
    finite = np.isfinite(ordered)
    duplicated = finite[:, 1:] & (ordered[:, 1:] == ordered[:, :-1])
    return np.any(duplicated, axis=1)


def _mesh_for(case: dict) -> Mesh:
    return Mesh(
        np.array([case["verts"][i] for i in sorted(case["verts"])]),
        np.array([case["tris"][i] for i in sorted(case["tris"])]),
    )


# --- conformance against the Java ---------------------------------------


def test_reference_covers_four_crystals() -> None:
    assert set(CASE_IDS) == {
        "cuboid", "cuboid-rotated", "cuboid-offset", "cylinder"
    }
    assert sum(len(c["depths"]) for c in REFERENCE_CASES.values()) == 5488


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_depths_match_java(case_id: str) -> None:
    """Every probe point, at four rotations, for four crystal shapes.

    Asserted at 1e-9. The measured worst case is 2.7e-15 and 98% of points
    come out bit-identical, so this has six orders of headroom -- which is
    the right margin for a calculation whose inputs go through a rotation
    and a normalisation before anything else happens.
    """
    case = REFERENCE_CASES[case_id]
    mesh = _mesh_for(case)
    translation = np.array(case["wedge"])

    by_angle: dict[float, list] = defaultdict(list)
    for angle, x, y, z, want in case["depths"]:
        by_angle[angle].append((x, y, z, want))

    worst = 0.0
    for angle, rows in by_angle.items():
        rotated = mesh.rotated(angle, translation)
        points = np.array([[x, y, z] for x, y, z, _ in rows])
        expected = np.array([want for *_, want in rows])
        got = np.asarray(find_depth(rotated, points))
        worst = max(worst, float(np.max(np.abs(got - expected))))

    assert worst < 1e-9, f"worst absolute deviation {worst:.3e} um"


# --- the bug this module is most likely to grow again -------------------


def test_depth_does_not_depend_on_the_rest_of_the_batch() -> None:
    """A voxel's depth is a property of that voxel. Nothing else.

    This failed during development, and quietly. The ray-plane numerator is
    a difference of two nearly equal numbers whenever a voxel lies on a face,
    and the denominator is near zero whenever a face is nearly parallel to
    the beam, so the quotient magnifies the last bits of the numerator
    enormously. Computing it with a matmul let BLAS reassociate, and it
    chooses different orders for different batch shapes -- so a voxel's depth
    changed depending on which other voxels were passed alongside it.

    A 90 degree rotation is the case that exposes it: cos is 6e-17 rather
    than 0, so faces end up very nearly, but not exactly, parallel to the
    beam.
    """
    mesh = cuboid().rotated(math.radians(90.0), np.zeros(3))
    points = np.array(
        [
            [-10.0, -10.0, 10.0],  # exactly on a rotated face
            [10.0, 10.0, 10.0],  # exactly on the opposite one
            [0.0, 0.0, 0.0],  # comfortably inside
            [0.0, 0.0, 500.0],  # comfortably outside
        ]
    )
    together = np.asarray(find_depth(mesh, points))
    separately = np.array(
        [float(find_depth(mesh, p[None, :])[0]) for p in points]
    )

    np.testing.assert_array_equal(
        together,
        separately,
        err_msg="depth must not depend on which voxels share the batch",
    )


def test_batch_order_does_not_matter() -> None:
    """The same points, shuffled, give the same depths in the same places."""
    rng = np.random.default_rng(0)
    mesh = cuboid().rotated(math.radians(37.0), np.zeros(3))
    points = rng.uniform(-25, 25, size=(200, 3))

    straight = np.asarray(find_depth(mesh, points))
    order = rng.permutation(len(points))
    shuffled = np.asarray(find_depth(mesh, points[order]))

    np.testing.assert_array_equal(straight[order], shuffled)


# --- geometry that must hold whatever the implementation ----------------


def test_a_voxel_outside_the_crystal_has_no_depth() -> None:
    mesh = cuboid()
    outside = np.array(
        [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0],
         [-100.0, -100.0, -100.0]]
    )
    np.testing.assert_array_equal(np.asarray(find_depth(mesh, outside)), 0.0)


def test_depth_at_the_entry_face_is_zero_and_at_the_exit_face_is_full() -> None:
    """The beam runs along +z, so a voxel at low z has just entered."""
    mesh = cuboid(z=20.0)
    just_inside_entry = np.array([[0.0, 0.0, -10.0 + 1e-9]])
    just_inside_exit = np.array([[0.0, 0.0, 10.0 - 1e-9]])

    assert float(find_depth(mesh, just_inside_entry)[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(find_depth(mesh, just_inside_exit)[0]) == pytest.approx(20.0, abs=1e-6)


def test_depth_grows_along_the_beam() -> None:
    """Monotone in z through a convex crystal: each step further in has more
    material behind it.
    """
    mesh = cuboid(z=20.0)
    z = np.linspace(-9.9, 9.9, 40)
    points = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    depths = np.asarray(find_depth(mesh, points))

    assert np.all(np.diff(depths) > 0)
    np.testing.assert_allclose(depths, z + 10.0, atol=1e-9)


def test_depth_never_exceeds_the_crystal_thickness() -> None:
    rng = np.random.default_rng(1)
    mesh = cuboid(z=20.0)
    points = rng.uniform(-30, 30, size=(2000, 3))
    depths = np.asarray(find_depth(mesh, points))

    assert np.all(depths >= 0)
    assert np.all(depths <= 20.0 + 1e-9)


def test_rotation_preserves_the_path_through_the_centre() -> None:
    """A cube turned about y presents the same thickness to the beam at 0
    and 90 degrees, and more at 45.
    """
    cube = cuboid(20.0, 20.0, 20.0)
    centre = np.array([[0.0, 0.0, 0.0]])

    upright = float(find_depth(cube.rotated(0.0, np.zeros(3)), centre)[0])
    quarter = float(
        find_depth(cube.rotated(math.radians(90.0), np.zeros(3)), centre)[0]
    )
    diagonal = float(
        find_depth(cube.rotated(math.radians(45.0), np.zeros(3)), centre)[0]
    )

    assert upright == pytest.approx(10.0)
    assert quarter == pytest.approx(10.0, abs=1e-9)
    assert diagonal == pytest.approx(10.0 * math.sqrt(2), abs=1e-9)


def test_translation_moves_the_crystal_not_the_voxel() -> None:
    """Shifting the crystal by +5 in z moves the depth profile with it."""
    mesh = cuboid(z=20.0)
    point = np.array([[0.0, 0.0, 0.0]])

    unshifted = float(find_depth(mesh.rotated(0.0, np.zeros(3)), point)[0])
    shifted = float(find_depth(mesh.rotated(0.0, np.array([0.0, 0.0, 5.0])), point)[0])

    assert unshifted == pytest.approx(10.0)
    assert shifted == pytest.approx(5.0)


def test_a_ray_through_a_shared_edge_is_counted_once() -> None:
    """Every internal edge of the mesh is shared by two triangles, so a ray
    through one registers twice at the same distance. Counting both flips
    the parity and reports an interior voxel as outside -- which is how this
    shows up: not as a small error, but as a depth of zero.

    The diagonal splitting each cuboid face is such an edge; for this
    triangulation it runs along ``x == -y``. Whether a particular ray along
    it produces two *bitwise equal* distances depends on rounding, so the
    points are found rather than written down -- of 400 along the diagonal,
    18 qualify. Asserting that some do is what stops this test quietly
    becoming vacuous.
    """
    mesh = cuboid(20.0, 20.0, 20.0)
    along_diagonal = np.array(
        [[-t, t, 0.0] for t in np.linspace(-9.5, 9.5, 400)]
    )
    shared = _shares_an_edge(mesh, along_diagonal)

    assert shared.any(), (
        "no ray here hits a shared edge, so this test proves nothing -- the "
        "mesh or its triangulation must have changed"
    )

    depths = np.asarray(find_depth(mesh, along_diagonal[shared]))
    assert np.all(depths > 0), (
        "a voxel under a shared triangle edge is still inside the crystal. "
        "Counting the edge twice flips the crossing parity and reports it as "
        f"outside, which is a depth of zero, not a small error; got {depths}"
    )
    np.testing.assert_allclose(depths, 10.0, atol=1e-9)


def test_triangle_planes_are_unit_normals_through_their_corners() -> None:
    mesh = cuboid()
    normals, offsets = triangle_planes(mesh)

    np.testing.assert_allclose(np.sum(normals**2, axis=1), 1.0, atol=1e-12)
    # Every corner of a triangle satisfies its own plane equation.
    corners = mesh.corners()
    for corner in range(3):
        residual = np.sum(normals * corners[:, corner], axis=1) + offsets
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)


# --- differentiability --------------------------------------------------


def test_gradient_flows_to_the_rotation_angle() -> None:
    torch = pytest.importorskip("torch")
    mesh = Mesh(
        torch.tensor(cuboid().vertices, dtype=torch.float64), CUBOID_TRIANGLES
    )
    angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    point = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)

    find_depth(mesh.rotated(angle), point).sum().backward()

    assert angle.grad is not None
    assert torch.isfinite(angle.grad)
    # Turning a 40 x 30 x 20 cuboid away from upright presents more material
    # along the beam, so the depth at the centre increases.
    assert angle.grad.item() > 0


def test_gradient_flows_to_the_vertices() -> None:
    torch = pytest.importorskip("torch")
    vertices = torch.tensor(
        cuboid().vertices, dtype=torch.float64, requires_grad=True
    )
    point = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)

    find_depth(Mesh(vertices, CUBOID_TRIANGLES), point).sum().backward()

    assert vertices.grad is not None
    assert torch.all(torch.isfinite(vertices.grad))
    assert torch.any(vertices.grad != 0), "some vertex must affect the depth"


def test_gradient_is_numerically_correct_away_from_edges() -> None:
    """``gradcheck`` in the interior only.

    Depth is piecewise: it steps as a ray crosses a silhouette edge or the
    crossing parity flips. Those are real discontinuities in the model, so a
    finite-difference check is only meaningful well inside a face.
    """
    torch = pytest.importorskip("torch")
    vertices = torch.tensor(cuboid().vertices, dtype=torch.float64)
    point = torch.tensor([[1.3, -2.1, 0.7]], dtype=torch.float64)

    def depth(angle: object) -> object:
        return find_depth(Mesh(vertices, CUBOID_TRIANGLES).rotated(angle), point)

    angle = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(depth, (angle,), eps=1e-6, atol=1e-7)


def test_both_backends_agree() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    points = rng.uniform(-25, 25, size=(500, 3))

    numpy_mesh = cuboid().rotated(0.7, np.zeros(3))
    torch_mesh = Mesh(
        torch.tensor(cuboid().vertices, dtype=torch.float64), CUBOID_TRIANGLES
    ).rotated(torch.tensor(0.7, dtype=torch.float64))

    got_numpy = np.asarray(find_depth(numpy_mesh, points))
    got_torch = find_depth(
        torch_mesh, torch.tensor(points, dtype=torch.float64)
    ).numpy()

    np.testing.assert_allclose(got_numpy, got_torch, atol=1e-12)
