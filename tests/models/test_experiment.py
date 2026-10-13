"""Tests for the configuration models and YAML round-tripping.

The models follow one rule: a field has a default only where RADDOSE-3D has
one, and it is RADDOSE-3D's. That is what makes leaving a setting out of a
YAML file mean the same as leaving it out of a ``.txt`` input. Both halves
are tested -- what the Java requires is required, and every default is the
Java's.
"""

import pytest
from _build import INSULIN
from _build import beam as _beam
from _build import crystal as _crystal
from _build import experiment as _experiment
from _build import wedge as _wedge
from pydantic import ValidationError

from pyraddose_3d.models.beam import Collimation, GaussianBeam, TophatBeam
from pyraddose_3d.models.composition import (
    AverageComposition,
    SaxsComposition,
    SequenceComposition,
    SmallMoleculeComposition,
    UnitCellComposition,
)
from pyraddose_3d.models.crystal import (
    CuboidCrystal,
    CylinderCrystal,
    PolyhedronCrystal,
    SphericalCrystal,
)
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.models.wedge import Wedge

# --- what RADDOSE-3D requires is required --------------------------------


@pytest.mark.parametrize(
    ("build", "missing"),
    [
        (lambda: Experiment(beam=_beam(), wedges=[_wedge()]), "crystal"),
        (lambda: Experiment(crystal=_crystal(), wedges=[_wedge()]), "beam"),
        (lambda: Experiment(crystal=_crystal(), beam=_beam()), "wedges"),
        (lambda: CuboidCrystal(composition=INSULIN), "dimensions"),
        (lambda: SphericalCrystal(composition=INSULIN), "diameter"),
        (lambda: CylinderCrystal(diameter=10.0, composition=INSULIN), "height"),
        (lambda: CuboidCrystal(dimensions=(1.0, 1.0, 1.0)), "composition"),
        (lambda: GaussianBeam(energy=12.1, fwhm=(20.0, 20.0)), "flux"),
        (lambda: GaussianBeam(flux=2e12, fwhm=(20.0, 20.0)), "energy"),
        (lambda: GaussianBeam(flux=2e12, energy=12.1), "fwhm"),
        (lambda: Wedge(end=90.0, exposure_time=10.0), "start"),
        (lambda: Wedge(start=0.0, exposure_time=10.0), "end"),
        (lambda: Wedge(start=0.0, end=90.0), "exposure_time"),
        (lambda: UnitCellComposition(num_monomers=24), "unit_cell"),
        (lambda: UnitCellComposition(unit_cell=(1.0, 1.0, 1.0)), "num_monomers"),
        (lambda: SaxsComposition(unit_cell=(1.0, 1.0, 1.0)), "protein_concentration"),
        (
            lambda: SequenceComposition(unit_cell=(1.0, 1.0, 1.0), sequence_file="s"),
            "num_monomers",
        ),
        (
            lambda: SmallMoleculeComposition(unit_cell=(1.0, 1.0, 1.0), atoms={"C": 1}),
            "num_monomers",
        ),
    ],
)
def test_what_raddose_3d_requires_is_required(build: object, missing: str) -> None:
    """These used to have defaults, several of them insulin's.

    A unit cell composition that left out its residue count was silently
    insulin, and a beam with no energy was 12.1 keV. RADDOSE-3D refuses or
    fails on each of these, and so do the models now.
    """
    with pytest.raises(ValidationError, match=missing):
        build()


def test_a_tophat_needs_its_aperture() -> None:
    """The aperture is the beam; BeamTophat will not construct without one."""
    with pytest.raises(ValidationError, match="aperture"):
        TophatBeam(flux=2e12, energy=12.1)


# --- every default is RADDOSE-3D's ----------------------------------------


@pytest.mark.parametrize(
    ("crystal", "expected"),
    [
        (_crystal(), 10 / 20.0),
        (SphericalCrystal(diameter=40.0, composition=INSULIN), 10 / 40.0),
        (CylinderCrystal(diameter=25.0, height=8.0, composition=INSULIN), 10 / 25.0),
        (PolyhedronCrystal(model_file="mesh.obj", composition=INSULIN), 0.5),
    ],
    ids=lambda value: getattr(value, "type", ""),
)
def test_default_resolution_is_ten_voxels_across(
    crystal: object, expected: float
) -> None:
    """``CRYSTAL_RESOLUTION_DEF = 10 / CRYSTAL_DIM_X`` in Crystal.java.

    Used to be a flat 0.5 here, which is the Java's static fallback for a
    shape with no dimensions and nothing else.
    """
    assert crystal.pixels_per_micron is None
    assert crystal.base_pixels_per_micron == pytest.approx(expected)


def test_a_given_resolution_is_kept() -> None:
    assert _crystal(pixels_per_micron=0.25).pixels_per_micron == 0.25


def test_no_collimation_means_uncollimated() -> None:
    """What RADDOSE-3D does when a beam block has no Collimation line.

    Used to be a 100 x 100 um aperture here, so the same omission meant
    different beams in the two formats.
    """
    assert _beam().collimation is None


def test_wedge_defaults_are_the_javas() -> None:
    wedge = _wedge()
    assert wedge.angular_resolution is None  # 2 degrees; see solve.wedge_resolution
    assert wedge.max_resolution == 2.0  # Wedge.java line 50
    assert wedge.start_offset is None
    assert wedge.translate_per_degree is None


def test_omitted_counts_are_zero() -> None:
    """RADDOSE-3D's counts are uninitialised ``int`` fields."""
    composition = UnitCellComposition(unit_cell=(78.0, 78.0, 78.0), num_monomers=4)
    assert composition.num_residues == 0
    assert composition.num_rna == 0
    assert composition.num_dna == 0
    assert composition.num_carb == 0


# --- validation -------------------------------------------------------------


def test_yaml_round_trip_preserves_the_experiment(tmp_path) -> None:
    exp = Experiment(
        crystal=CylinderCrystal(diameter=40.0, height=20.0, composition=INSULIN),
        beam=TophatBeam(flux=1e11, energy=9.0, collimation=Collimation(x=50.0, y=50.0)),
        wedges=[_wedge(start=0, end=180, exposure_time=30)],
        seed=1234,
    )
    path = tmp_path / "experiment.yaml"
    exp.to_yaml(path)

    assert Experiment.from_yaml(path) == exp


# A mistyped key must fail loudly rather than be silently ignored -- dose runs
# are often unattended, so a dropped parameter is worse than a hard error.
@pytest.mark.parametrize(
    "payload",
    [
        {"crystal": {"type": "cuboid", "dimenions": (1, 2, 3)}},  # typo
        {"beam": {"type": "gaussian", "fhwm": (20, 70)}},  # typo
        {"nonsense": 1},
    ],
)
def test_unknown_keys_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        _experiment(**payload)


def test_unknown_crystal_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="crystal"):
        _experiment(crystal={"type": "dodecahedron", "dimensions": (1, 2, 3)})


def test_unknown_composition_method_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _crystal(composition={"method": "telepathy"})


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_resolution_must_be_positive(value: float) -> None:
    with pytest.raises(ValidationError):
        _crystal(pixels_per_micron=value)


def test_a_wedge_is_required() -> None:
    with pytest.raises(ValidationError):
        _experiment(wedges=[])


def test_small_molecule_requires_its_atoms() -> None:
    with pytest.raises(ValidationError):
        SmallMoleculeComposition(unit_cell=(1.0, 1.0, 1.0), num_monomers=1)
    composition = SmallMoleculeComposition(
        unit_cell=(1.0, 1.0, 1.0), num_monomers=1, atoms={"Mg": 1, "O": 3}
    )
    assert composition.atoms["O"] == 3


def test_escape_reports_whether_anything_is_enabled() -> None:
    assert not _crystal().escape.any_enabled
    assert _crystal(escape={"photoelectron": True}).escape.any_enabled


@pytest.mark.parametrize(
    "crystal",
    [
        _crystal(),
        SphericalCrystal(diameter=50.0, composition=INSULIN),
        CylinderCrystal(diameter=40.0, height=20.0, composition=INSULIN),
        PolyhedronCrystal(model_file="mesh.obj", composition=INSULIN),
    ],
    ids=lambda c: c.type,
)
def test_every_crystal_type_round_trips(crystal: object, tmp_path) -> None:
    exp = _experiment(crystal=crystal)
    path = tmp_path / "c.yaml"
    exp.to_yaml(path)
    assert Experiment.from_yaml(path).crystal == crystal


@pytest.mark.parametrize(
    "composition",
    [
        AverageComposition(),
        UnitCellComposition(**INSULIN),
        SmallMoleculeComposition(
            unit_cell=(5.0, 5.0, 5.0), num_monomers=2, atoms={"C": 1}
        ),
    ],
    ids=lambda c: c.method,
)
def test_every_composition_round_trips(composition: object, tmp_path) -> None:
    exp = _experiment(crystal=_crystal(composition=composition))
    path = tmp_path / "c.yaml"
    exp.to_yaml(path)
    assert Experiment.from_yaml(path).crystal.composition == composition


def test_collimation_shape_is_constrained() -> None:
    assert Collimation(shape="circular", x=1.0, y=1.0).shape == "circular"
    with pytest.raises(ValidationError):
        Collimation(shape="triangular", x=1.0, y=1.0)


# --- the Monte Carlo and XFEL subprograms ------------------------------------


@pytest.mark.parametrize("subprogram", ["montecarlo", "gos"])
def test_the_monte_carlo_subprograms_take_the_usual_beams(subprogram: str) -> None:
    experiment = _experiment(crystal=_crystal(subprogram=subprogram))
    assert experiment.crystal.simulation.photons == 1_000_000
    tophat = TophatBeam(
        flux=2e12, energy=12.1, collimation=Collimation(shape="circular", x=5, y=5)
    )
    assert _experiment(crystal=_crystal(subprogram=subprogram), beam=tophat)


def test_simulation_settings_need_a_simulation() -> None:
    """Only the Monte Carlo and XFEL subprograms read them."""
    with pytest.raises(ValidationError, match="read only by the Monte Carlo"):
        _crystal(simulation={"runs": 2})
    assert _crystal(subprogram="gos", simulation={"runs": 2}).simulation.runs == 2


@pytest.mark.parametrize(
    ("simulation", "wrong"),
    [({"photons": 0}, "photons"), ({"runs": 0}, "runs")],
)
def test_simulation_counts_are_at_least_one(simulation: dict, wrong: str) -> None:
    with pytest.raises(ValidationError, match=wrong):
        _crystal(subprogram="montecarlo", simulation=simulation)


def test_a_surrounding_thickness_needs_a_surrounding() -> None:
    """Crystal.java reads it only when the composition has a surrounding."""
    thickness = {"surrounding_thickness": (1.0, 1.0, 1.0)}
    with pytest.raises(ValidationError, match=r"needs escape\.surrounding"):
        _crystal(subprogram="montecarlo", simulation=thickness)
    crystal = _crystal(
        subprogram="montecarlo", simulation=thickness, escape={"surrounding": True}
    )
    assert crystal.simulation.surrounding_thickness == (1.0, 1.0, 1.0)


def test_the_simulated_subprograms_refuse_the_average_composition() -> None:
    with pytest.raises(ValidationError, match="need the crystal's elements"):
        _crystal(subprogram="xfel", composition={"method": "average"})


def test_the_simulated_subprograms_refuse_an_experimental_beam(tmp_path) -> None:
    pgm = tmp_path / "beam.pgm"
    pgm.write_bytes(b"P2\n1 1\n255\n1\n")
    beam = {
        "type": "experimental",
        "flux": 2e12,
        "energy": 12.1,
        "file": str(pgm),
        "pixel_size": (1.0, 1.0),
    }
    with pytest.raises(ValidationError, match="Gaussian or TopHat"):
        _experiment(crystal=_crystal(subprogram="montecarlo"), beam=beam)


def test_xfel_reads_the_pulse_energy_and_only_xfel_does() -> None:
    xfel = _crystal(subprogram="xfel")
    pulse = _beam(flux=None, pulse_energy=1.4)
    assert _experiment(crystal=xfel, beam=pulse).beam.pulse_energy == 1.4
    with pytest.raises(ValidationError, match="needs the beam's pulse energy"):
        _experiment(crystal=xfel)
    with pytest.raises(ValidationError, match="read only by the XFEL"):
        _experiment(beam=_beam(pulse_energy=1.4))
    with pytest.raises(ValidationError, match="read only by the XFEL"):
        _experiment(crystal=_crystal(subprogram="gos"), beam=pulse)


def test_a_beam_needs_a_flux_or_a_pulse_energy() -> None:
    with pytest.raises(ValidationError, match="flux, or for the XFEL"):
        _beam(flux=None)
