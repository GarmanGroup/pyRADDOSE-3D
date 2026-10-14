"""The simple Monte Carlo's parts, one at a time.

Its whole runs are compared with RADDOSE-3D's statistically, in
``tests/conformance/test_montecarlo_files.py``. These pin the parts that can
be pinned exactly: the electron physics against MicroED's, the angle tables'
indexing, the geometry, and the bookkeeping.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from pyraddose_3d.backend import microed
from pyraddose_3d.backend.montecarlo import angles, geometry, material, photons
from pyraddose_3d.backend.montecarlo.simple import simulate_simple
from pyraddose_3d.backend.montecarlo.transport import Media, new_directions, track
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import monte_carlo_setup

CASE = "tests/conformance/cases/140-montecarlo-baseline/config.yaml"


@pytest.fixture(scope="module")
def setup():  # type: ignore[no-untyped-def]
    return monte_carlo_setup(Experiment.from_yaml(CASE))


ENERGIES = np.array([0.05, 0.075, 0.5, 5.0, 12.1, 20.0, 20.5, 150.0, 300.0, 350.0])


def test_the_stopping_power_is_microeds(setup) -> None:  # type: ignore[no-untyped-def]
    vectorised = material.stopping_power(setup.material, ENERGIES)
    for energy, value in zip(ENERGIES, vectorised, strict=True):
        assert value == pytest.approx(
            microed.stopping_power(setup.material, float(energy)), rel=1e-13
        )


def test_the_elastic_mean_free_path_is_microeds(setup) -> None:  # type: ignore[no-untyped-def]
    sigma = material.elastic_cross_sections(setup.material, ENERGIES)
    vectorised = material.elastic_mean_free_path(setup.material, sigma)
    for energy, value in zip(ENERGIES, vectorised, strict=True):
        assert value == pytest.approx(
            microed.elastic_mean_free_path(setup.material, float(energy)), rel=1e-13
        )


def test_the_elastic_element_shares_run_to_one(setup) -> None:  # type: ignore[no-untyped-def]
    sigma = material.elastic_cross_sections(setup.material, ENERGIES)
    cdf = material.elastic_element_cdf(setup.material, sigma)
    assert np.all(np.diff(cdf, axis=0) >= 0)
    np.testing.assert_allclose(cdf[-1], 1.0, rtol=1e-14)


def test_angles_by_column_as_the_java_counts_them() -> None:
    """Up to 20 keV the column is the angle; above, returnDeflectionAngle's ranges."""
    low = np.degrees(angles.index_angles(False, 181))
    np.testing.assert_allclose(low, np.arange(181.0), atol=1e-12)
    high = np.degrees(angles.index_angles(True, 606))
    assert high[0] == 0.0
    assert high[1] == pytest.approx(1e-4)
    assert high[2] == pytest.approx(1.25e-4)
    assert high[37] == pytest.approx(1e-3)
    assert high[146] == pytest.approx(1.0)
    assert high[236] == pytest.approx(10.0)
    assert high[297] == pytest.approx(25.5)
    assert high[605] == pytest.approx(179.5)


def test_the_header_and_the_formula_part_above_one_degree() -> None:
    """The Java's angle lags the file's by a column from 1 degree on."""
    table = angles.angular_table(6, high_energy=True)
    formula = np.degrees(angles.index_angles(True, 606))
    np.testing.assert_allclose(formula[:146], table.angles_deg[:146], rtol=1e-9)
    assert formula[146] == 1.0
    assert table.angles_deg[146] == pytest.approx(1.1)


def test_a_draw_takes_the_first_angle_reaching_it() -> None:
    """At the nearest tabulated energy: 0.075 keV is a tie, and takes 0.05."""
    table = angles.angular_table(6, high_energy=False)
    row = table.probabilities[0]
    cumulative = np.cumsum(row) / row.sum()
    for column in (0, 1, 30, 179):
        draw = np.array([cumulative[column] - 1e-12])
        got = angles.deflection_angles(np.array([6]), np.array([0.075]), draw)
        assert math.degrees(got[0]) == pytest.approx(column)


def test_the_cuboids_geometry() -> None:
    from pyraddose_3d.backend.geometry import Mesh

    half = np.array([10.0, 8.0, 6.0])
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float
    )
    faces = [
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ]  # fmt: skip
    mesh = Mesh(signs * half, np.array(faces))
    still = geometry.Placement(np.zeros(3), np.zeros((3, 3)))
    points = np.array([[0.0, 0.0, 0.0], [9000.0, 7000.0, 5000.0], [0.0, 0.0, 7000.0]])
    assert geometry.inside(mesh, points, still).tolist() == [True, True, False]
    depth = geometry.entry_depth(
        mesh, np.array([0.0, 2000.0, 20000.0]), np.zeros(3), still
    )
    assert depth.tolist() == pytest.approx([-6000.0, -6000.0, 0.0])
    turned = geometry.Placement(np.full(1, math.pi / 2), np.zeros((1, 3)))
    assert geometry.entry_depth(mesh, np.zeros(1), np.zeros(1), turned)[0] == (
        pytest.approx(-10000.0)
    )
    along_x = geometry.exit_distance(
        mesh, np.zeros((1, 3)), np.array([[1.0, 0, 0]]), still.take(np.arange(1))
    )
    assert along_x[0] == pytest.approx(10000.0)
    ijk = geometry.voxel_indices(points[:2], still.take(np.arange(2)), -half, 0.5)
    assert ijk.tolist() == [[5, 4, 3], [10, 8, 6]]


def test_directions_stay_unit_vectors() -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(size=(1000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    turned = new_directions(
        d, rng.random(1000) * math.pi, rng.random(1000) * 2 * math.pi
    )
    np.testing.assert_allclose(np.linalg.norm(turned, axis=1), 1.0, rtol=1e-12)


def test_an_electron_that_stops_inside_deposits_its_energy(setup) -> None:  # type: ignore[no-untyped-def]
    """Including the negative remainder of an overshooting last step."""
    from pyraddose_3d.backend.geometry import Mesh

    big = Mesh(np.asarray(setup.mesh.vertices) * 100, np.asarray(setup.mesh.triangles))
    rng = np.random.default_rng(3)
    n = 200
    deposited = np.zeros(n)

    def deposit(points, which, energy, starting):  # type: ignore[no-untyped-def]
        np.add.at(deposited, which, energy)

    energy = rng.uniform(0.01, 12.0, n)
    direction = np.tile([0.0, 0.6, 0.8], (n, 1))
    still = geometry.Placement(np.zeros(n), np.zeros((n, 3)))
    track(
        rng,
        Media(setup.material, big),
        np.zeros((n, 3)),
        direction,
        energy,
        np.zeros(n, dtype=bool),
        still,
        np.zeros(n, dtype=int),
        deposit,
    )
    np.testing.assert_allclose(deposited, energy, rtol=1e-12, atol=1e-12)


def test_beams_stay_within_their_apertures() -> None:
    rng = np.random.default_rng(4)
    for kind, circular in (("gaussian", False), ("tophat", False)):
        beam = photons.BeamShape(kind, (40.0, 30.0), circular, (8.0, 8.0))
        x, y = photons.beam_positions(rng, beam, 10000)
        assert beam.exposes(x, y).all()


def test_a_circular_gaussian_reaches_the_x_aperture_in_every_direction() -> None:
    """With equal widths its radius is drawn within the x aperture alone, so
    an elliptical aperture narrower in y is overrun; reproduced."""
    rng = np.random.default_rng(4)
    beam = photons.BeamShape("gaussian", (40.0, 30.0), True, (20.0, 20.0))
    x, y = photons.beam_positions(rng, beam, 10000)
    radius = np.hypot(x, y)
    assert radius.max() <= 20000
    assert np.abs(y).max() > 15000


def test_a_circular_tophat_fills_a_bow_tie() -> None:
    """Its y is bounded by the line from the centre to the aperture's corner,
    not by the ellipse; reproduced."""
    rng = np.random.default_rng(4)
    beam = photons.BeamShape("tophat", (40.0, 30.0), True)
    x, y = photons.beam_positions(rng, beam, 10000)
    assert np.all(np.abs(y) / 15000 <= np.abs(x) / 20000 + 1e-12)
    assert not beam.exposes(x, y).all()


def test_compton_electrons_stay_within_the_compton_edge() -> None:
    rng = np.random.default_rng(5)
    energy, direction = photons.compton_electrons(rng, np.full(10000, 12.1))
    mc2 = 510.99895
    assert energy.max() <= 2 * 12.1**2 / (mc2 + 2 * 12.1) * (1 + 1e-9)
    assert energy.min() >= 0
    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, rtol=1e-12)


def test_runs_repeat_from_a_seed(setup) -> None:  # type: ignore[no-untyped-def]
    small = dataclasses.replace(setup, photons=20000, runs=2)
    first = simulate_simple(small, np.random.default_rng(11))
    second = simulate_simple(small, np.random.default_rng(11))
    assert [r.mc_adwc for r in first] == [r.mc_adwc for r in second]
    assert first[0].mc_adwc != first[1].mc_adwc


def test_below_the_k_edge_iodine_never_takes_the_k_shell() -> None:
    """RADDOSE-3D's fix: a draw past every shell ionises an outer one."""
    cdf = photons.shell_cdf(53, np.full(5, 12.1))
    assert np.all(cdf[:, 0] == 0)
    modelled = cdf[0, 3]
    assert modelled == pytest.approx(0.78, abs=0.01)
    draws = np.array([0.0, modelled / 2, modelled, 0.9, 0.999999])
    shells = photons.ionised_shells(cdf, draws)
    assert (
        shells.tolist()
        == [1, shells[1], photons.OUTER_SHELL] + [photons.OUTER_SHELL] * 2
    )
    assert 0 not in shells.tolist()


def test_most_elements_have_a_shell_for_every_draw() -> None:
    """Where the last shell modelled has an edge ratio of 0, it takes the rest."""
    for z in (1, 6, 7, 8, 15, 16, 30):
        assert photons.shell_cdf(z, np.array([12.1])).max() == pytest.approx(1.0)


def test_an_outer_shell_leaves_nothing_at_the_atom() -> None:
    from pyraddose_3d.backend.montecarlo.simple import _relaxation

    z = np.array([53, 53, 6])
    shell = np.array([photons.OUTER_SHELL, 1, 0])
    binding = np.array([0.0, 5.19, 0.28])
    rng = np.random.default_rng(0)
    left, resolved = _relaxation(z, shell, binding, np.zeros(3), rng)
    # Iodine does not relax: its L1 binding energy goes to both tallies.
    assert left.tolist() == [0.0, 5.19, 0.28]
    assert resolved.tolist() == [0.0, 5.19, 0.0]


def test_the_surrounding_pushes_each_vertex_out() -> None:
    from pyraddose_3d.backend.geometry import Mesh
    from pyraddose_3d.backend.montecarlo.simple import _box

    mesh = Mesh(np.array([[-1.0, -2.0, 0.0], [1.0, 2.0, 3.0]]), np.zeros((0, 3), int))
    box = _box(mesh, (500.0, 1000.0, 2000.0))
    assert np.asarray(box.vertices).tolist() == [[-1.5, -3.0, 2.0], [1.5, 3.0, 5.0]]


def test_the_surroundings_thickness_is_a_running_cap(setup) -> None:  # type: ignore[no-untyped-def]
    """Capped photon by photon, in order, by a photoelectron's reach."""
    from pyraddose_3d.backend.montecarlo.material import stopping_power
    from pyraddose_3d.backend.montecarlo.simple import Surrounding, _boxes

    surrounding = Surrounding(
        setup.composition, setup.density, setup.material, (2.0, 0.0, 1e6)
    )
    energy = np.array([12.0, 10.0, 11.0])
    carry = np.array([[2000.0, np.inf, 1e9]])
    meshes: list = []
    index = _boxes(setup, surrounding, energy, np.zeros(3, int), carry, {}, meshes)
    reach = energy / stopping_power(setup.material, energy)
    assert len(meshes) == 2 and index[1] == index[2] != index[0]
    assert carry[0].tolist() == [2000.0, reach[1], reach[1]]


def test_the_ions_file_iterates_as_a_hash_map_does() -> None:
    """Element.hashCode is the atomic number; the table has 16 buckets."""
    from pyraddose_3d.outputs import _hash_map_order

    assert _hash_map_order([1, 6, 7, 8, 15, 16, 30]) == [16, 1, 6, 7, 8, 30, 15]


def test_the_files_are_laid_out_as_raddose_3ds() -> None:
    from pyraddose_3d.backend.montecarlo.simple import SimpleRun
    from pyraddose_3d.outputs import monte_carlo_csv, monte_carlo_ions_csv

    run = SimpleRun(rd3d_adwc=1.5, mc_adwc=float("nan"), ions_per_element={8: 0.25})
    lines = monte_carlo_csv([run, run]).splitlines()
    assert lines[1] == " 1, 1.500000, 0.000000, NaN, 0.000000, 0.000000"
    assert lines[2].startswith(" 2, ")
    assert monte_carlo_ions_csv(run) == (
        "Atomic number,ionisationsPerAtom\n 8, 0.250000\n"
    )


def _cuboid(x: float, y: float, z: float):  # type: ignore[no-untyped-def]
    from pyraddose_3d.models.crystal import CuboidCrystal
    from pyraddose_3d.solve import crystal_mesh

    crystal = CuboidCrystal(
        dimensions=(x, y, z),
        composition={"method": "unit_cell", "unit_cell": (78.0, 78.0, 78.0),
                     "num_monomers": 24, "num_residues": 51},
    )  # fmt: skip
    return crystal_mesh(crystal)


def test_escape_distances_are_to_the_rotated_crystal() -> None:
    """RADDOSE-3D #44: from the centre of a 4 x 6 x 8 um crystal rotated 90
    degrees, along x, 4 um -- half its z, now along x -- not 2."""
    mesh = _cuboid(4.0, 6.0, 8.0)
    turned = geometry.Placement(np.full(1, math.pi / 2), np.zeros((1, 3)))
    distance = geometry.exit_distance(
        mesh, np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), turned
    )
    assert distance[0] == pytest.approx(4000.0)


def test_entry_distances_are_to_the_face_crossed() -> None:
    """From outside, past a corner: not the nearest plane ahead, y = 3."""
    mesh = _cuboid(4.0, 6.0, 8.0)
    still = geometry.Placement(np.zeros(1), np.zeros((1, 3)))
    norm = math.sqrt(1.25)
    direction = np.array([[1 / norm, -0.5 / norm, 0.0]])
    distance = geometry.exit_distance(
        mesh, np.array([[-5000.0, 4000.0, 0.0]]), direction, still
    )
    assert distance[0] == pytest.approx(3000 * norm)


def test_an_offset_rotated_crystals_points_find_their_voxels() -> None:
    """RADDOSE-3D #44: the inverse of translating and then rotating."""
    angle = math.pi / 2
    place = geometry.Placement(np.full(1, angle), np.array([[0.5, 0.0, 0.0]]))
    corner = np.array([1.99 + 0.5, 2.99, 3.99])
    x = corner[0] * math.cos(angle) + corner[2] * math.sin(angle)
    z = -corner[0] * math.sin(angle) + corner[2] * math.cos(angle)
    point = np.array([[x * 1000, corner[1] * 1000, z * 1000]])
    ijk = geometry.voxel_indices(point, place, np.array([-2.0, -3.0, -4.0]), 2.0)
    assert ijk.tolist() == [[8, 12, 16]]
