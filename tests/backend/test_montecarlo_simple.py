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
from pyraddose_3d.backend.montecarlo import angles, geometry, material, photons, tables
from pyraddose_3d.backend.montecarlo.angles import column_weights
from pyraddose_3d.backend.montecarlo.simple import (
    _placements,
    _relaxation,
    simulate_simple,
)
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


def test_elastic_angles_are_weighted_by_sin_theta() -> None:
    """RADDOSE-3D #46: a flat cross-section per solid angle puts a quarter
    of deflections under 60 degrees, not 45."""
    degrees = np.arange(181.0)
    running = np.cumsum(column_weights(degrees))
    quarter = degrees[np.searchsorted(running / running[-1], 0.25)]
    assert quarter == pytest.approx(60, abs=1)


def test_elastic_angles_are_the_tables_own() -> None:
    """RADDOSE-3D #46: above 20 keV, column 146 is the header's 1.1 degrees."""
    sampler = angles._sampler(6, True)
    assert math.degrees(sampler.angles[146]) == pytest.approx(1.1)


def test_a_draw_takes_the_first_angle_reaching_it() -> None:
    """At the nearest tabulated energy: 0.075 keV is a tie, and takes 0.05."""
    table = angles.angular_table(6, high_energy=False)
    weighted = table.probabilities[0] * column_weights(table.angles_deg)
    cumulative = np.cumsum(weighted) / weighted.sum()
    for column in (1, 30, 179):
        draw = np.array([cumulative[column] - 1e-12])
        got = angles.deflection_angles(np.array([6]), np.array([0.075]), draw)
        assert math.degrees(got[0]) == pytest.approx(column)


def test_a_materials_block_holds_each_elements_table() -> None:
    """Gathered for a material: each element's own running chances, and
    none for mercury above 20 keV, which ELSEPA does not tabulate."""
    block = angles.angular_block((1, 6, 80), True)
    assert block.slot[80] == -1
    for z in (1, 6):
        np.testing.assert_array_equal(
            block.cumulative[block.slot[z]], angles._sampler(z, True).cumulative
        )
    with pytest.raises(ValueError, match="Z = 80"):
        angles.deflection_angles(
            np.array([6, 80]), np.array([30.0, 30.0]), np.array([0.5, 0.5]), (1, 6, 80)
        )


@pytest.mark.parametrize("high_energy", [False, True])
def test_the_search_takes_the_first_column_reaching_the_draw(high_energy: bool) -> None:
    """A fixed number of passes finds what a search to the end would."""
    block = angles.angular_block((1, 6, 7, 8, 16), high_energy)
    rng = np.random.default_rng(4)
    n = 2000
    slot = rng.integers(0, 5, n)
    rows = rng.integers(0, len(block.energies_kev), n)
    uniform = rng.random(n)
    expected = [
        np.searchsorted(block.cumulative[s, r], u, side="left")
        for s, r, u in zip(slot, rows, uniform, strict=True)
    ]
    np.testing.assert_array_equal(
        angles._first_reaching(block.cumulative, slot, rows, uniform), expected
    )


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
    assert depth[:2].tolist() == pytest.approx([-6000.0, -6000.0])
    assert np.isnan(depth[2])
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


def test_deflections_turn_through_their_angle() -> None:
    """Through theta, at phi around the old direction: opposite sides of the
    cone sum along it. Directions along z, and in the xz plane, included."""
    rng = np.random.default_rng(1)
    d = rng.normal(size=(1000, 3))
    d[:3] = [[0, 0, 1], [0, 0, -1], [0.6, 0, 0.8]]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    theta = rng.random(1000) * math.pi
    phi = rng.random(1000) * 2 * math.pi
    one = new_directions(d, theta, phi)
    other = new_directions(d, theta, phi + math.pi)
    np.testing.assert_allclose(np.sum(one * d, axis=1), np.cos(theta), atol=1e-9)
    np.testing.assert_allclose(one + other, 2 * np.cos(theta)[:, None] * d, atol=1e-9)


def test_isotropic_directions_are_uniform_over_the_sphere() -> None:
    d = photons.isotropic_directions(np.random.default_rng(3), 200_000)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(d.mean(axis=0), 0, atol=0.005)
    np.testing.assert_allclose((d * d).mean(axis=0), 1 / 3, atol=0.005)


def test_polarised_directions_are_uniform_around_the_axis() -> None:
    rng = np.random.default_rng(5)
    for axis in (0, 1):
        d = photons.about_axis(rng, np.full(200_000, 0.6), axis)
        assert np.all(d[:, axis] == 0.6)
        others = [k for k in range(3) if k != axis]
        np.testing.assert_allclose((d[:, others] ** 2).mean(axis=0), 0.32, atol=0.003)


def test_photoelectron_angles_follow_the_dipole_distribution() -> None:
    """``cos^2`` per solid angle: the cosine's square averages 3/5.

    Every K-shell photoelectron follows it, as since RADDOSE-3D #59; three
    in four did, the rest isotropic.
    """
    assert photons._EMISSION[-1] == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(7)
    cos = photons.cos_angle_to_x(rng, 200_000)
    assert np.mean(cos**2) == pytest.approx(0.6, abs=0.005)
    k_shell = np.ones(200_000, dtype=bool)
    for rotated, axis in ((True, 0), (False, 1)):
        d = photons.photoelectron_directions(rng, k_shell, rotated)
        assert np.mean(d[:, axis] ** 2) == pytest.approx(0.6, abs=0.005)


def test_compton_angles_follow_klein_and_nishina() -> None:
    """Thomson's ``1 + cos^2`` at low energy; forward at high."""
    rng = np.random.default_rng(9)
    mu = photons.klein_nishina_cosines(rng, np.full(200_000, 1e-9))
    assert np.mean(mu**2) == pytest.approx(0.4, abs=0.004)
    grid = np.linspace(-1, 1, 20001)
    p = 1 / (1 + (1 - grid))
    f = p * p * (p + 1 / p - (1 - grid * grid))
    expected = np.trapezoid(grid * f, grid) / np.trapezoid(f, grid)
    mu = photons.klein_nishina_cosines(rng, np.full(200_000, 1.0))
    assert np.mean(mu) == pytest.approx(expected, abs=0.005)


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
    for kind in ("gaussian", "tophat"):
        for circular in (False, True):
            beam = photons.BeamShape(kind, (40.0, 30.0), circular, (8.0, 8.0))
            x, y = photons.beam_positions(rng, beam, 10000)
            assert beam.exposes(x, y).all()


def test_a_circular_tophat_fills_its_circle() -> None:
    """RADDOSE-3D #45: half the photons have |y| > |x|. It filled a bow tie."""
    rng = np.random.default_rng(4)
    beam = photons.BeamShape("tophat", (5.0, 5.0), True)
    x, y = photons.beam_positions(rng, beam, 20000)
    assert np.mean(np.abs(y) > np.abs(x)) == pytest.approx(0.5, abs=0.02)


def test_a_circular_gaussians_radii_are_two_dimensional() -> None:
    """RADDOSE-3D #45: 1 - exp(-1/2) within one sigma, not 68%."""
    rng = np.random.default_rng(4)
    sigma = 20 / (2 * math.sqrt(2 * math.log(2)))
    beam = photons.BeamShape("gaussian", (200.0, 200.0), True, (sigma, sigma))
    x, y = photons.beam_positions(rng, beam, 20000)
    within = np.mean(np.hypot(x, y) / 1000 < sigma)
    assert within == pytest.approx(1 - math.exp(-0.5), abs=0.015)


def test_a_circular_exposed_area_is_the_ellipses() -> None:
    beam = photons.BeamShape("tophat", (5.0, 4.0), True)
    assert beam.exposed_area_nm2() == pytest.approx(math.pi * 2500 * 2000)


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


def test_mixed_elements_shells_are_each_ones_own() -> None:
    """``shell_cdfs`` over photons of many elements is ``shell_cdf``'s for
    each, to the last bit."""
    elements = (1, 6, 16, 30, 53, 78)
    rng = np.random.default_rng(6)
    element = rng.integers(0, len(elements), 3000)
    energy = rng.uniform(0.5, 100.0, 3000)
    table = photons.stacked_shells(elements)
    z = np.array(elements)[element]
    mixed = photons.shell_cdfs(
        z, table.edges[element], table.ionisation[element], energy
    )
    for i, one in enumerate(elements):
        mine = element == i
        np.testing.assert_array_equal(mixed[mine], photons.shell_cdf(one, energy[mine]))


def test_a_spectrums_coefficients_are_each_batchs(setup) -> None:  # type: ignore[no-untyped-def]
    """Computed once for a pink beam's every energy, they are what each
    batch's photons would have had computed for them alone."""
    rng = np.random.default_rng(7)
    spectrum = np.sort(rng.uniform(11.0, 13.0, 20_000))
    whole = photons.photon_coefficients(setup.composition, spectrum)
    batch = rng.integers(0, len(spectrum), 5000)
    alone = photons.photon_coefficients(setup.composition, spectrum[batch])
    for name in (
        "absorption",
        "compton",
        "elastic",
        "photoelectric_cdf",
        "compton_cdf",
    ):
        np.testing.assert_array_equal(
            getattr(whole.take(batch), name), getattr(alone, name)
        )


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


def test_the_surrounding_is_the_crystal_scaled_about_its_centre() -> None:
    """RADDOSE-3D #49: about its bounding box's centre, so a crystal off the
    origin is enclosed. Pushed away from the origin, this one's z ran from 2."""
    from pyraddose_3d.backend.geometry import Mesh
    from pyraddose_3d.backend.montecarlo.simple import _box

    mesh = Mesh(np.array([[-1.0, -2.0, 0.0], [1.0, 2.0, 3.0]]), np.zeros((0, 3), int))
    box = _box(mesh, (500.0, 1000.0, 2000.0))
    assert np.asarray(box.vertices).tolist() == [[-1.5, -3.0, -2.0], [1.5, 3.0, 5.0]]


def test_a_cylinders_surrounding_is_as_thick_all_round() -> None:
    from pyraddose_3d.backend.montecarlo.simple import _box

    cylinder = monte_carlo_setup(
        Experiment.from_yaml(
            "tests/conformance/cases/146-montecarlo-cylinder/config.yaml"
        )
    ).mesh
    v = np.asarray(cylinder.vertices)
    w = np.asarray(_box(cylinder, (1000.0, 1000.0, 1000.0)).vertices)
    np.testing.assert_allclose(
        np.hypot(w[:, 0], w[:, 2]), np.hypot(v[:, 0], v[:, 2]) * 3.5 / 2.5
    )


def test_the_surrounding_is_capped_by_the_most_energetic_photon(setup) -> None:  # type: ignore[no-untyped-def]
    """RADDOSE-3D #49: once, by the reach of the most energetic photon's
    photoelectron; 0 takes the cap. It was a running cap, photon by photon."""
    from pyraddose_3d.backend.montecarlo.material import stopping_power
    from pyraddose_3d.backend.montecarlo.simple import Surrounding, _thickness

    surrounding = Surrounding(
        setup.composition, setup.density, setup.material, (2.0, 0.0, 1e6)
    )
    pink = dataclasses.replace(setup, energies=np.array([10.0, 11.0, 12.0]))
    reach = 12.0 / stopping_power(setup.material, np.array([12.0]))[0]
    assert 2000 < reach < 1e9
    assert _thickness(pink, surrounding) == pytest.approx((2000.0, reach, reach))


def _crossing(setup, start_z: float, direction_z: float, energy: float) -> float:  # type: ignore[no-untyped-def]
    """What one electron at the crystal's +z face deposits, in a surrounding."""
    from pyraddose_3d.backend.montecarlo.simple import _box

    box = _box(setup.mesh, (1000.0, 1000.0, 1000.0))
    deposited = []

    def deposit(points, which, energy, starting):  # type: ignore[no-untyped-def]
        deposited.extend(energy.tolist())

    track(
        np.random.default_rng(5),
        Media(setup.material, setup.mesh, setup.material, [box]),
        np.array([[0.0, 0.0, start_z]]),
        np.array([[0.0, 0.0, direction_z]]),
        np.array([energy]),
        np.array([start_z > 6000.0]),
        geometry.Placement(np.zeros(1), np.zeros((1, 3))),
        np.zeros(1, dtype=int),
        deposit,
    )
    return float(np.sum(deposited))


def test_an_electron_coming_back_with_under_50_ev_leaves_it(setup) -> None:  # type: ignore[no-untyped-def]
    """RADDOSE-3D #49: where it comes in. It was lost."""
    assert np.asarray(setup.mesh.vertices)[:, 2].max() == 6.0
    energy = 0.05 + 1e-9
    assert _crossing(setup, 6000.0 + 1e-6, -1.0, energy) == pytest.approx(
        0.05, abs=1e-7
    )


def test_an_electron_leaving_with_under_50_ev_leaves_it(setup) -> None:  # type: ignore[no-untyped-def]
    """RADDOSE-3D #49: where it left from, as with no surrounding: all of it
    is deposited. Only the way out was."""
    energy = 0.05 + 1e-9
    assert _crossing(setup, 6000.0 - 1e-6, 1.0, energy) == pytest.approx(
        energy, abs=1e-12
    )


def test_the_ions_file_lists_elements_by_atomic_number() -> None:
    """As since RADDOSE-3D #56; it listed them in a HashMap's bucket order."""
    from pyraddose_3d.backend.montecarlo.simple import SimpleRun
    from pyraddose_3d.outputs import monte_carlo_ions_csv

    run = SimpleRun(ions_per_element={16: 0.5, 1: 0.1, 30: 0.2, 6: 0.3, 15: 0.4})
    lines = monte_carlo_ions_csv(run).splitlines()[1:]
    assert [int(line.split(",")[0]) for line in lines] == [1, 6, 15, 16, 30]


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


def test_a_face_at_zero_is_not_a_miss() -> None:
    """RADDOSE-3D #48: a crystal whose front face is at z = 0 is crossed there."""
    mesh = _cuboid(4.0, 6.0, 8.0)
    shifted = type(mesh)(
        np.asarray(mesh.vertices) + np.array([0, 0, 4.0]), mesh.triangles
    )
    still = geometry.Placement(np.zeros(2), np.zeros((2, 3)))
    front = geometry.entry_depth(shifted, np.array([0.0, 5000.0]), np.zeros(2), still)
    assert front[0] == pytest.approx(0.0)
    assert np.isnan(front[1])


def test_wedges_turn_from_their_start_to_their_end(setup) -> None:  # type: ignore[no-untyped-def]
    """RADDOSE-3D #48: a still at its start, a rotation from start to end."""
    index = np.array([0, 250, 500])
    still = dataclasses.replace(
        setup, start_angle=math.radians(30), end_angle=math.radians(30), photons=1000
    )
    assert np.degrees(_placements(still, index).angle) == pytest.approx([30] * 3)
    down = dataclasses.replace(
        setup, start_angle=math.radians(90), end_angle=0.0, photons=1000
    )
    assert np.degrees(_placements(down, index).angle) == pytest.approx([90, 67.5, 45])


def test_the_elastic_chance_is_of_every_interaction() -> None:
    """RADDOSE-3D #48: elastic, photoelectric and Compton 2 : 1 : 1, a half."""
    one = np.ones(1)
    coefficients = photons.PhotonCoefficients(
        one, one, 2 * one, one[:, None], one[:, None]
    )
    assert coefficients.elastic_chance[0] == 0.5


def test_phosphorus_relaxes() -> None:
    """RADDOSE-3D #48: some of its K holes fluoresce, and their photons leave."""
    n = 4000
    binding = np.full(n, 2.149)
    left, resolved = _relaxation(
        np.full(n, 15), np.zeros(n, dtype=np.int64), binding, np.full(n, 0.064),
        np.random.default_rng(13),
    )  # fmt: skip
    assert not resolved.any()
    assert 0.9 * 2.149 < left.mean() < 0.97 * 2.149


def test_fluorescence_transitions_are_the_fluorescence_tables() -> None:
    """RADDOSE-3D #48: drawn by the fluorescence table, not the Auger one."""
    n = 20000
    left, _ = _relaxation(
        np.full(n, 16), np.zeros(n, dtype=np.int64), np.full(n, 2.472), np.ones(n),
        np.random.default_rng(17),
    )  # fmt: skip
    table = tables.fluorescence_transitions()[(16, 0)]
    chance = np.diff(np.maximum.accumulate(np.concatenate([[0.0], table.cumulative])))
    chance = np.maximum(chance, 0)
    for k in np.flatnonzero(chance):
        share = np.mean(np.isclose(left, 2.472 - table.energy_kev[k]))
        assert share == pytest.approx(chance[k], abs=0.01), k


def test_every_heavy_element_has_its_shells_binding_energies() -> None:
    """Xenon and heavier, but gadolinium, had no L1 binding energy: in
    RADDOSE-3D a photoelectron from platinum's L1 shell stopped the run."""
    heavy = [z for z in sorted(microed._shell_table()) if z >= 35]
    for z in heavy:
        binding = photons.shells(z).binding
        assert np.isfinite(binding).all() and (binding > 0).all(), z
