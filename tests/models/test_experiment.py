"""Tests for the configuration models and YAML round-tripping."""

import pytest
from pydantic import ValidationError

from pyraddose_3d.models.beam import Collimation, GaussianBeam, TophatBeam
from pyraddose_3d.models.composition import (
    AverageComposition,
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


def test_defaults_build_a_usable_experiment():
    exp = Experiment()
    assert isinstance(exp.crystal, CuboidCrystal)
    assert isinstance(exp.beam, GaussianBeam)
    assert len(exp.wedges) == 1


def test_yaml_round_trip_preserves_the_experiment(tmp_path):
    exp = Experiment(
        crystal=CylinderCrystal(diameter=40.0, height=20.0),
        beam=TophatBeam(flux=1e11, energy=9.0),
        wedges=[Wedge(start=0, end=180, exposure_time=30)],
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
def test_unknown_keys_are_rejected(payload):
    with pytest.raises(ValidationError):
        Experiment(**payload)


def test_unknown_crystal_type_is_rejected():
    with pytest.raises(ValidationError, match="crystal"):
        Experiment(crystal={"type": "dodecahedron", "dimensions": (1, 2, 3)})


def test_unknown_composition_method_is_rejected():
    with pytest.raises(ValidationError):
        CuboidCrystal(composition={"method": "telepathy"})


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({"pixels_per_micron": 0.0}, "resolution must be positive"),
        ({"pixels_per_micron": -1.0}, "resolution must be positive"),
    ],
)
def test_invalid_crystal_values_are_rejected(payload, why):
    with pytest.raises(ValidationError):
        CuboidCrystal(**payload)


def test_a_wedge_is_required():
    with pytest.raises(ValidationError):
        Experiment(wedges=[])


def test_small_molecule_requires_its_atoms():
    with pytest.raises(ValidationError):
        SmallMoleculeComposition()
    assert SmallMoleculeComposition(atoms={"Mg": 1, "O": 3}).atoms["O"] == 3


def test_escape_reports_whether_anything_is_enabled():
    crystal = CuboidCrystal()
    assert not crystal.escape.any_enabled

    crystal = CuboidCrystal(escape={"photoelectron": True})
    assert crystal.escape.any_enabled


@pytest.mark.parametrize(
    "crystal",
    [
        CuboidCrystal(),
        SphericalCrystal(diameter=50.0),
        CylinderCrystal(diameter=40.0, height=20.0),
        PolyhedronCrystal(model_file="mesh.obj"),
    ],
    ids=lambda c: c.type,
)
def test_every_crystal_type_round_trips(crystal, tmp_path):
    exp = Experiment(crystal=crystal)
    path = tmp_path / "c.yaml"
    exp.to_yaml(path)
    assert Experiment.from_yaml(path).crystal == crystal


@pytest.mark.parametrize(
    "composition",
    [
        AverageComposition(),
        UnitCellComposition(),
        SmallMoleculeComposition(atoms={"C": 1}),
    ],
    ids=lambda c: c.method,
)
def test_every_composition_round_trips(composition, tmp_path):
    exp = Experiment(crystal=CuboidCrystal(composition=composition))
    path = tmp_path / "c.yaml"
    exp.to_yaml(path)
    assert Experiment.from_yaml(path).crystal.composition == composition


def test_collimation_shape_is_constrained():
    assert Collimation(shape="circular").shape == "circular"
    with pytest.raises(ValidationError):
        Collimation(shape="triangular")
