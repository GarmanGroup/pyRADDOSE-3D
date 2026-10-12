"""Tests for the voxel grid and the exposure loop.

This is the calculation the whole port exists to reproduce, so the
conformance check runs the entire chain -- mesh, grid, occupancy, beam,
depth, attenuation, dose -- against RADDOSE-3D and compares per voxel.

On the tolerances
-----------------
Interior voxels agree to machine precision. Voxels lying **exactly on a
crystal face** do not, and cannot: they sit on the knife edge of the
inside/outside test, and Java's ``Math.cos`` and NumPy's differ in the last
bit or two. Neither is wrong -- only ``StrictMath`` is required to be
reproducible, which is why RADDOSE-3D already varies against itself across
JVMs.

:func:`test_on_face_voxels_are_intrinsically_unstable` measures how much
that matters rather than asserting it: nudging the crystal by a single ulp
moves the average dose by 3e-4, which is **larger** than the disagreement
with Java. The port therefore tracks RADDOSE-3D more closely than the
calculation tracks itself under the smallest perturbation expressible in
double precision.
"""

import csv
import math
import pathlib
from collections import defaultdict

import numpy as np
import pytest
from _geometry_helpers import CUBOID_TRIANGLES, cuboid

from pyraddose_3d.backend.beam import GaussianBeam, TophatBeam
from pyraddose_3d.backend.coefficients import MaterialCoefficients
from pyraddose_3d.backend.exposure import (
    STATIC_EXPOSURE_STEPS,
    expose,
    exposure_angles,
)
from pyraddose_3d.backend.geometry import Mesh
from pyraddose_3d.backend.voxels import occupancy, voxel_grid

REFERENCE = pathlib.Path(__file__).parent / "data" / "exposure.tsv"

#: Crystal dimensions every reference case uses, in micrometres.
CRYSTAL = (40.0, 30.0, 20.0)

#: The AngleP / AngleL each case was built with, in degrees.
CRYSTAL_ANGLES = {"rotated-crystal": (45.0, 20.0)}


def apply_crystal_angles(
    vertices: np.ndarray, angle_p: float, angle_l: float
) -> np.ndarray:
    """Apply ``AngleP`` then ``AngleL``, as the Java constructor does.

    P turns the crystal about z, L about x. Both are given in degrees in the
    input file -- the constructor converts -- which is worth stating because
    passing radians looks right and silently gives a 57-fold smaller turn.
    """
    turn_p = math.radians(angle_p)
    turn_l = math.radians(angle_l)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    x2 = x * math.cos(turn_p) + y * math.sin(turn_p)
    y2 = -x * math.sin(turn_p) + y * math.cos(turn_p)
    return np.stack(
        [
            x2,
            y2 * math.cos(turn_l) + z * math.sin(turn_l),
            -y2 * math.sin(turn_l) + z * math.cos(turn_l),
        ],
        axis=-1,
    )


def _load_reference() -> dict:
    cases: dict = defaultdict(lambda: {"voxels": []})
    with REFERENCE.open(encoding="ascii") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            tag, case_id = row[0], row[1]
            case = cases[case_id]
            if tag == "SETUP":
                case["coefficients"] = MaterialCoefficients(
                    absorption=float(row[2]),
                    attenuation=float(row[3]),
                    elastic=float(row[4]),
                    inelastic=float(row[5]),
                )
                case["density"] = float(row[6])
                case["pixels_per_micron"] = float(row[7])
            elif tag == "WEDGE":
                case["wedge"] = [float(v) for v in row[2:9]]
            elif tag == "BEAM":
                case["beam"] = (row[2], row[3])
            elif tag == "GRID":
                case["grid"] = tuple(int(v) for v in row[2:5])
            elif tag == "SUMMARY":
                case["summary"] = (int(row[2]), float(row[3]), float(row[4]))
            elif tag == "VOX":
                case["voxels"].append(
                    (
                        int(row[2]),
                        int(row[3]),
                        int(row[4]),
                        row[5] == "true",
                        float(row[6]),
                        float(row[7]),
                        float(row[8]),
                    )
                )
    return dict(cases)


REFERENCE_CASES = _load_reference()
CASE_IDS = sorted(REFERENCE_CASES)


def _run(case_id: str, **options: object) -> tuple:
    """Reproduce one reference case end to end; ``options`` go to ``expose``."""
    case = REFERENCE_CASES[case_id]
    angle_p, angle_l = CRYSTAL_ANGLES.get(case_id, (0.0, 0.0))
    mesh = Mesh(
        apply_crystal_angles(cuboid(*CRYSTAL).vertices, angle_p, angle_l),
        CUBOID_TRIANGLES,
    )
    grid = voxel_grid(mesh, case["pixels_per_micron"])
    coordinates = grid.coordinates()
    occupied = occupancy(mesh, coordinates)

    kind, collimation = case["beam"]
    if kind == "tophat":
        beam = TophatBeam(2.0e12, 12.1, 100.0, 100.0)
    else:
        aperture = None if collimation == "none" else float(collimation)
        beam = GaussianBeam(2.0e12, 12.1, 80.0, 60.0, aperture, aperture)

    start, end, resolution, seconds, *position = case["wedge"]
    angles = exposure_angles(start, end, resolution)
    translations = np.tile(np.array(position), (len(angles), 1))

    result = expose(
        mesh,
        grid,
        occupied,
        beam,
        case["coefficients"],
        case["density"],
        angles,
        seconds,
        translations=translations,
        **options,
    )
    return case, grid, coordinates, occupied, result


def _on_a_face(coordinates: np.ndarray, case_id: str = "") -> np.ndarray:
    """Which voxels lie exactly on a face of an axis-aligned crystal.

    Only meaningful for the unrotated cases; a rotated crystal has no
    axis-aligned faces, so nothing is flagged and every voxel is compared
    tightly.
    """
    if case_id in CRYSTAL_ANGLES:
        return np.zeros(len(coordinates), dtype=bool)
    half = np.array(CRYSTAL) / 2
    return np.any(np.isclose(np.abs(coordinates), half, atol=1e-9), axis=1)


# --- conformance against the Java ---------------------------------------


def test_reference_covers_five_exposures() -> None:
    assert set(CASE_IDS) == {
        "still-tophat",
        "rotation-gauss",
        "collimated",
        "rotated-crystal",
        "offset",
    }


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_grid_and_occupancy_match_java(case_id: str) -> None:
    """Exact, not approximate. These are counts, and a crystal one voxel
    bigger is a different crystal.
    """
    case, grid, _, occupied, _ = _run(case_id)

    assert grid.shape == case["grid"]
    assert int(np.sum(occupied)) == case["summary"][0]


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_summary_dose_matches_java(case_id: str) -> None:
    """Average and maximum dose over the whole crystal, to 1e-4.

    The measured worst case is 1.6e-5 on the average, and that residual is
    entirely on-face voxels -- see the module docstring.
    """
    case, _, _, _, result = _run(case_id)
    _, average, peak = case["summary"]

    assert float(result.average_dose) == pytest.approx(average, rel=1e-4)
    assert float(result.max_dose) == pytest.approx(peak, rel=1e-4)


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_interior_voxels_match_java_to_machine_precision(case_id: str) -> None:
    """Every sampled voxel not sitting on a face, at 1e-9.

    Splitting the comparison is the point. Averaging the on-face voxels into
    a single loose tolerance would hide a real regression in the interior,
    where the calculation is perfectly well conditioned.
    """
    case, grid, coordinates, occupied, result = _run(case_id)
    on_face = _on_a_face(coordinates, case_id)
    _, ny, nz = grid.shape

    compared = 0
    for i, j, k, inside, dose, fluence, elastic in case["voxels"]:
        index = (i * ny + j) * nz + k
        assert bool(occupied[index]) == inside, f"occupancy at ({i},{j},{k})"
        if on_face[index]:
            continue
        compared += 1
        assert float(result.dose[index]) == pytest.approx(dose, rel=1e-9, abs=1e-12)
        assert float(result.fluence[index]) == pytest.approx(
            fluence, rel=1e-9, abs=1e-30
        )
        assert float(result.elastic[index]) == pytest.approx(
            elastic, rel=1e-9, abs=1e-30
        )

    assert compared > 0, "every sampled voxel was skipped -- nothing was checked"


def test_on_face_voxels_are_intrinsically_unstable() -> None:
    """Measures why the summary tolerance is 1e-4 rather than 1e-12.

    A voxel exactly on a crystal face sits on the knife edge of the
    inside/outside parity test. Moving the crystal by one ulp -- the
    smallest change a double can express, and far below any physical scale
    -- flips some of them.

    The result: the average dose moves by more than this port disagrees with
    RADDOSE-3D. The disagreement is therefore a property of the calculation,
    not of the translation, and no amount of care in the port would remove
    it. It is also why RADDOSE-3D already varies against itself across JVMs.
    """
    case, grid, coordinates, occupied, result = _run("rotation-gauss")
    mesh = Mesh(cuboid(*CRYSTAL).vertices, CUBOID_TRIANGLES)

    start, end, resolution, seconds, *position = case["wedge"]
    angles = exposure_angles(start, end, resolution)
    beam = GaussianBeam(2.0e12, 12.1, 80.0, 60.0)
    common = {
        "coefficients": case["coefficients"],
        "density": case["density"],
        "angles": angles,
        "exposure_seconds": seconds,
        "translations": np.tile(np.array(position), (len(angles), 1)),
    }

    nudged = expose(
        Mesh(np.nextafter(mesh.vertices, mesh.vertices + 1), CUBOID_TRIANGLES),
        grid,
        occupied,
        beam,
        **common,
    )

    before = np.asarray(result.dose)
    after = np.asarray(nudged.dose)
    moved = np.abs(after - before) / np.maximum(np.abs(before), 1e-30) > 1e-9
    on_face = _on_a_face(coordinates, "rotation-gauss")

    assert moved.any(), "a one-ulp nudge should move something"
    assert np.all(on_face[moved]), (
        "only voxels exactly on a face should be sensitive to a one-ulp "
        "change; an interior voxel moving would mean the calculation is "
        "unstable where it has no right to be"
    )

    shift = abs(float(nudged.average_dose) - float(result.average_dose)) / float(
        result.average_dose
    )
    assert shift > 1e-5, (
        "if a one-ulp nudge no longer moves the average dose by more than "
        "the tolerance this suite allows against Java, that tolerance can "
        f"be tightened; it moved by {shift:.2e}"
    )


# --- the voxel grid -----------------------------------------------------


def test_images_are_not_recorded_unless_asked() -> None:
    """A field per image is A times the memory of the totals; opt in only."""
    *_, result = _run("rotation-gauss")
    assert result.images is None


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_recorded_images_agree_with_the_totals(case_id: str) -> None:
    """The per-image record is the same calculation, kept rather than summed.

    So the last image's dose is the final dose, the per-image fluences add
    up to the wedge's fluence, and recording changes nothing it records.
    """
    *_, plain = _run(case_id)
    *_, recorded = _run(case_id, record_images=True)
    images = recorded.images

    assert images is not None
    assert images.dose.shape == images.fluence.shape == images.decay.shape
    assert images.dose.shape[0] == len(plain.image_dwd)
    np.testing.assert_array_equal(recorded.dose, plain.dose)
    np.testing.assert_array_equal(images.dose[-1], plain.dose)
    np.testing.assert_allclose(images.fluence.sum(axis=0), plain.fluence, rtol=1e-12)
    # With no decay model every voxel diffracts at full efficiency.
    assert np.all(images.decay == 1.0)


def test_grid_spans_the_bounding_box_inclusively() -> None:
    """A 40 um edge at 0.5 voxels per micron gets 21 voxels, not 20: there
    is a voxel on each face, not just between them.
    """
    grid = voxel_grid(cuboid(40.0, 30.0, 20.0), 0.5)
    assert grid.shape == (21, 16, 11)

    coordinates = np.asarray(grid.coordinates())
    np.testing.assert_allclose(coordinates[0], [-20.0, -15.0, -10.0])
    np.testing.assert_allclose(coordinates[-1], [20.0, 15.0, 10.0])


def test_voxel_count_rounds_half_away_from_zero() -> None:
    """Java's ``StrictMath.round`` rounds half up; Python's ``round`` rounds
    half to even. A crystal whose extent lands on exactly half a voxel would
    otherwise come out a voxel short, and every dose with it.
    """
    # 45 um at 0.1 per um is 4.5 voxels: half away from zero gives 5, plus
    # the inclusive end makes 6. Banker's rounding would give 5.
    assert voxel_grid(cuboid(45.0, 45.0, 45.0), 0.1).shape == (6, 6, 6)


def test_the_grid_keeps_the_precision_it_was_given() -> None:
    """Guards a silent single-precision downgrade on the torch backend.

    ``torch.arange`` yields int64, and dividing an integer tensor by a
    Python float promotes to torch's *default* dtype -- float32 -- not to
    the float64 of whatever it is added to afterwards. The whole voxel grid
    was therefore single precision, and every dose with it, at about 1e-7
    relative: far too small to look wrong and far too large for a project
    whose conformance tolerances are 1e-9.
    """
    torch = pytest.importorskip("torch")

    plain = cuboid(40.0, 30.0, 20.0)
    grid = voxel_grid(plain, 0.5)
    assert np.asarray(grid.coordinates()).dtype == np.float64

    torch_grid = type(grid)(
        origin=torch.tensor(np.asarray(grid.origin), dtype=torch.float64),
        shape=grid.shape,
        pixels_per_micron=grid.pixels_per_micron,
    )
    assert torch_grid.coordinates().dtype == torch.float64

    np.testing.assert_array_equal(
        np.asarray(grid.coordinates()), torch_grid.coordinates().numpy()
    )


def test_occupancy_finds_the_interior() -> None:
    mesh = cuboid(40.0, 30.0, 20.0)
    inside = occupancy(mesh, np.array([[0.0, 0.0, 0.0], [19.0, 14.0, 9.0]]))
    outside = occupancy(mesh, np.array([[100.0, 0.0, 0.0], [0.0, 0.0, -50.0]]))

    assert np.all(inside)
    assert not np.any(outside)


# --- the angle schedule -------------------------------------------------


def test_a_still_exposure_repeats_one_angle() -> None:
    """A wedge narrower than one step is a still. It is sampled many times
    rather than once, because the exposure time is divided between the
    samples -- so a still and a rotation of equal duration deposit the same
    energy.
    """
    angles = exposure_angles(0.0, 0.0, math.radians(2))
    assert len(angles) == STATIC_EXPOSURE_STEPS
    assert set(angles) == {0.0}


def test_a_rotation_steps_by_the_resolution() -> None:
    angles = exposure_angles(0.0, math.radians(90), math.radians(2))
    assert len(angles) == 46
    assert angles[0] == pytest.approx(0.0)
    assert angles[-1] == pytest.approx(math.radians(90))


def test_a_reversed_wedge_counts_backwards() -> None:
    """A reversed wedge gets *44* steps where the forward one gets 46.

    ``count = sign * int((end - start) / resolution + 1)``. Forward that is
    ``int(46.0)``; reversed it is ``int(-44.0)``, because the ``+ 1`` moves
    toward zero rather than away from it and ``int`` truncates. So a wedge
    swept backwards is sampled at two fewer angles than the same wedge swept
    forwards, and ends two steps short.

    Pinned rather than corrected: it changes results, and the exposure time
    is divided by the step count, so it is not a simple truncation.
    """
    forward = exposure_angles(0.0, math.radians(90), math.radians(2))
    reverse = exposure_angles(math.radians(90), 0.0, math.radians(2))

    assert len(forward) == 46
    assert len(reverse) == 44
    assert reverse[0] > reverse[-1]
    assert reverse[-1] == pytest.approx(math.radians(90 - 43 * 2))


def test_whole_turns_are_removed_from_both_ends() -> None:
    """A wedge from 370 to 460 degrees is the same exposure as 10 to 100:
    only the complete revolutions already in the start angle come off.
    """
    plain = exposure_angles(math.radians(10), math.radians(100), math.radians(5))
    wound = exposure_angles(math.radians(370), math.radians(460), math.radians(5))
    np.testing.assert_allclose(plain, wound, atol=1e-12)


# --- physics that must hold ---------------------------------------------


def _simple_exposure(**overrides: object) -> object:
    mesh = cuboid(40.0, 30.0, 20.0)
    grid = voxel_grid(mesh, 0.5)
    occupied = occupancy(mesh, grid.coordinates())
    settings = {
        "beam": GaussianBeam(2.0e12, 12.1, 80.0, 60.0),
        "coefficients": MaterialCoefficients(2.3e-4, 2.67e-4, 1.9e-5, 1.77e-5),
        "density": 1.08,
        "angles": exposure_angles(0.0, math.radians(90), math.radians(10)),
        "exposure_seconds": 50.0,
    }
    settings.update(overrides)
    return expose(mesh, grid, occupied, **settings)


def test_dose_is_proportional_to_exposure_time() -> None:
    single = float(_simple_exposure(exposure_seconds=50.0).average_dose)
    double = float(_simple_exposure(exposure_seconds=100.0).average_dose)
    assert double == pytest.approx(2 * single, rel=1e-12)


def test_dose_is_proportional_to_flux() -> None:
    single = float(
        _simple_exposure(beam=GaussianBeam(1.0e12, 12.1, 80.0, 60.0)).average_dose
    )
    double = float(
        _simple_exposure(beam=GaussianBeam(2.0e12, 12.1, 80.0, 60.0)).average_dose
    )
    assert double == pytest.approx(2 * single, rel=1e-12)


def test_spreading_the_same_exposure_over_more_angles_gives_the_same_dose() -> None:
    """The exposure time is divided between the angles, so sampling a wedge
    more finely must not deposit more energy. Getting this wrong makes dose
    depend on the angular resolution, which is a setting, not a physical
    quantity.
    """
    coarse = float(
        _simple_exposure(
            angles=exposure_angles(0.0, math.radians(90), math.radians(10))
        ).average_dose
    )
    fine = float(
        _simple_exposure(
            angles=exposure_angles(0.0, math.radians(90), math.radians(2))
        ).average_dose
    )
    assert fine == pytest.approx(coarse, rel=0.02)


def test_attenuation_shields_the_far_side_of_the_crystal() -> None:
    """Without rotation the beam enters at low z, so voxels it reaches last
    get less. This is the whole point of tracking depth.
    """
    mesh = cuboid(40.0, 30.0, 400.0)
    grid = voxel_grid(mesh, 0.5)
    coordinates = np.asarray(grid.coordinates())
    occupied = occupancy(mesh, coordinates)
    result = expose(
        mesh,
        grid,
        occupied,
        TophatBeam(2.0e12, 12.1, 200.0, 200.0),
        MaterialCoefficients(2.3e-4, 2.67e-2, 1.9e-5, 1.77e-5),
        1.08,
        [0.0],
        50.0,
    )

    dose = np.asarray(result.dose)
    # The y axis runs -15, -13, ... 15, so there is no voxel at y = 0; the
    # window has to be wide enough to catch one.
    centre = (np.abs(coordinates[:, 0]) < 3) & (np.abs(coordinates[:, 1]) < 3)
    entry = centre & (coordinates[:, 2] < -190)
    far = centre & (coordinates[:, 2] > 190)
    assert entry.any() and far.any(), "the sampling window caught no voxels"

    assert dose[entry].mean() > dose[far].mean() * 5


def test_no_dose_outside_the_crystal() -> None:
    mesh = cuboid(40.0, 30.0, 20.0)
    grid = voxel_grid(mesh, 0.5)
    occupied = occupancy(mesh, grid.coordinates())
    result = _simple_exposure()

    assert np.all(np.asarray(result.dose)[~np.asarray(occupied)] == 0.0)


# --- differentiability --------------------------------------------------


def _torch_exposure(torch: object, **leaves: object) -> object:
    mesh = Mesh(
        torch.tensor(cuboid(40.0, 30.0, 20.0).vertices, dtype=torch.float64),
        CUBOID_TRIANGLES,
    )
    grid = voxel_grid(cuboid(40.0, 30.0, 20.0), 0.25)
    occupied = occupancy(cuboid(40.0, 30.0, 20.0), grid.coordinates())
    grid = type(grid)(
        origin=torch.tensor(np.asarray(grid.origin), dtype=torch.float64),
        shape=grid.shape,
        pixels_per_micron=grid.pixels_per_micron,
    )
    beam = GaussianBeam(
        leaves.get("flux", 2.0e12), leaves.get("energy", 12.1), 80.0, 60.0
    )
    return expose(
        mesh,
        grid,
        torch.tensor(np.asarray(occupied)),
        beam,
        MaterialCoefficients(2.3e-4, 2.67e-4, 1.9e-5, 1.77e-5),
        1.08,
        [0.0, 0.3],
        leaves.get("seconds", 50.0),
    )


def test_dose_is_differentiable_in_the_beam_settings() -> None:
    """The reason the port exists: a strategy optimiser needs these."""
    torch = pytest.importorskip("torch")
    flux = torch.tensor(2.0e12, dtype=torch.float64, requires_grad=True)
    energy = torch.tensor(12.1, dtype=torch.float64, requires_grad=True)
    seconds = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)

    _torch_exposure(
        torch, flux=flux, energy=energy, seconds=seconds
    ).average_dose.backward()

    for name, leaf in (("flux", flux), ("energy", energy), ("seconds", seconds)):
        assert leaf.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(leaf.grad), f"non-finite gradient at {name}"

    assert flux.grad.item() > 0, "more photons must mean more dose"
    assert seconds.grad.item() > 0, "a longer exposure must mean more dose"


def test_the_dose_gradient_is_numerically_correct() -> None:
    torch = pytest.importorskip("torch")

    def average(flux: object, seconds: object) -> object:
        return _torch_exposure(torch, flux=flux, seconds=seconds).average_dose

    args = (
        torch.tensor(2.0e12, dtype=torch.float64, requires_grad=True),
        torch.tensor(50.0, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(average, args, eps=1e-4, atol=1e-6, rtol=1e-4)


def test_both_backends_agree() -> None:
    """The same exposure, computed twice, must give the same dose."""
    torch = pytest.importorskip("torch")

    plain = cuboid(40.0, 30.0, 20.0)
    grid = voxel_grid(plain, 0.5)
    occupied = occupancy(plain, grid.coordinates())
    settings = {
        "coefficients": MaterialCoefficients(2.3e-4, 2.67e-4, 1.9e-5, 1.77e-5),
        "density": 1.08,
        "angles": exposure_angles(0.0, math.radians(90), math.radians(10)),
        "exposure_seconds": 50.0,
    }

    on_numpy = expose(
        plain, grid, occupied, GaussianBeam(2.0e12, 12.1, 80.0, 60.0), **settings
    )

    as_tensor = Mesh(
        torch.tensor(plain.vertices, dtype=torch.float64), CUBOID_TRIANGLES
    )
    torch_grid = type(grid)(
        origin=torch.tensor(np.asarray(grid.origin), dtype=torch.float64),
        shape=grid.shape,
        pixels_per_micron=grid.pixels_per_micron,
    )
    on_torch = expose(
        as_tensor,
        torch_grid,
        torch.tensor(np.asarray(occupied)),
        GaussianBeam(
            torch.tensor(2.0e12, dtype=torch.float64),
            torch.tensor(12.1, dtype=torch.float64),
            80.0,
            60.0,
        ),
        **settings,
    )

    interior = ~_on_a_face(np.asarray(grid.coordinates()), "plain")
    np.testing.assert_allclose(
        np.asarray(on_numpy.dose)[interior],
        on_torch.dose.numpy()[interior],
        rtol=1e-12,
        err_msg="the two backends must agree wherever the answer is stable",
    )

    # On-face voxels are no more stable between NumPy and torch than they
    # are between Python and Java: their trig differs in the last bit as
    # well. The aggregate stays close, which is what the summary asserts
    # against RADDOSE-3D.
    assert float(on_torch.average_dose) == pytest.approx(
        float(on_numpy.average_dose), rel=1e-4
    )
