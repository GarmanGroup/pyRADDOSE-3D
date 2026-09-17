"""Tests for the legacy .txt emitter.

The emitter is what lets YAML be the single source of truth for conformance
cases: the .txt handed to the Java reference implementation is generated from
the same model the Python code will consume, so the two inputs cannot drift.
If these tests are wrong, every conformance result is meaningless.
"""

import pytest

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

BLOCK_STARTS = ("Crystal", "Beam", "Wedge")


def blocks(text: str) -> dict[str, dict[str, str]]:
    """Parse emitted text into {block: {keyword: rest-of-line}}.

    Block-aware because ``Type`` appears in both the Crystal and Beam blocks;
    a flat dict would let one silently overwrite the other.
    """
    out: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key, _, rest = line.partition(" ")
        if key in BLOCK_STARTS and (key != "Wedge" or "Wedge" not in out):
            current = out.setdefault(key, {})
            if key == "Wedge":
                current["Wedge"] = rest.strip()
            continue
        assert current is not None, f"directive outside a block: {line}"
        current[key] = rest.strip()
    return out


def directives(text: str) -> dict[str, str]:
    """All directives, flattened. Only safe for keywords unique across blocks."""
    flat: dict[str, str] = {}
    for block in blocks(text).values():
        flat.update(block)
    return flat


def test_blocks_appear_in_the_order_raddose_expects():
    text = Experiment().to_raddose_txt()
    assert text.index("Crystal") < text.index("Beam") < text.index("Wedge")


def test_baseline_emits_the_expected_directives():
    exp = Experiment(
        crystal=CuboidCrystal(
            dimensions=(100, 100, 100),
            pixels_per_micron=0.1,
            composition=UnitCellComposition(
                unit_cell=(78.02, 78.02, 78.02),
                num_monomers=24,
                num_residues=51,
                protein_heavy_atoms={"Zn": 0.333, "S": 6},
                solvent_heavy_conc={"P": 425},
                solvent_fraction=0.64,
            ),
        ),
        beam=GaussianBeam(
            flux=2e12,
            fwhm=(20, 70),
            energy=12.1,
            collimation=Collimation(shape="rectangular", x=100, y=100),
        ),
        wedges=[Wedge(start=0, end=90, exposure_time=50)],
    )
    parsed = blocks(exp.to_raddose_txt())
    d = directives(exp.to_raddose_txt())

    assert parsed["Crystal"]["Type"] == "Cuboid"
    assert parsed["Beam"]["Type"] == "Gaussian"
    assert d["Dimensions"] == "100 100 100"
    assert d["PixelsPerMicron"] == "0.1"
    assert d["AbsCoefCalc"] == "RD3D"
    assert d["UnitCell"] == "78.02 78.02 78.02"
    assert d["ProteinHeavyAtoms"] == "Zn 0.333 S 6"
    assert d["SolventHeavyConc"] == "P 425"
    assert d["SolventFraction"] == "0.64"
    assert d["FWHM"] == "20 70"
    assert d["Energy"] == "12.1"
    assert d["Collimation"] == "Rectangular 100 100"
    assert d["Wedge"] == "0 90"
    assert d["ExposureTime"] == "50"


@pytest.mark.parametrize(
    ("crystal", "expected_type"),
    [
        (CuboidCrystal(), "Cuboid"),
        (SphericalCrystal(diameter=50.0), "Spherical"),
        (CylinderCrystal(diameter=40.0, height=20.0), "Cylinder"),
        (PolyhedronCrystal(model_file="mesh.obj"), "Polyhedron"),
    ],
    ids=lambda x: getattr(x, "type", x),
)
def test_crystal_types_emit_their_keyword(crystal, expected_type):
    parsed = blocks(Experiment(crystal=crystal).to_raddose_txt())
    assert parsed["Crystal"]["Type"] == expected_type


def test_cylinder_dimensions_follow_raddose_axis_convention():
    """Diameter, then height, then an ignored third value.

    RADDOSE-3D reads radius from Dimensions[0] and height from Dimensions[1],
    and ignores Dimensions[2] entirely.
    """
    text = Experiment(
        crystal=CylinderCrystal(diameter=40.0, height=20.0)
    ).to_raddose_txt()
    assert directives(text)["Dimensions"].split()[:2] == ["40", "20"]


def test_polyhedron_emits_its_mesh_file():
    text = Experiment(
        crystal=PolyhedronCrystal(model_file="tests/data/cuboid.obj")
    ).to_raddose_txt()
    d = directives(text)
    assert d["ModelFile"] == "tests/data/cuboid.obj"
    assert d["WireframeType"] == "obj"


@pytest.mark.parametrize(
    ("composition", "keyword"),
    [
        (AverageComposition(), "Average"),
        (UnitCellComposition(), "RD3D"),
        (SmallMoleculeComposition(atoms={"Mg": 1, "O": 3}), "SmallMole"),
        (SaxsComposition(), "SAXS"),
        (SequenceComposition(sequence_file="seq.fasta"), "Sequence"),
    ],
    ids=lambda x: getattr(x, "method", x),
)
def test_composition_emits_the_right_abscoefcalc(composition, keyword):
    text = Experiment(crystal=CuboidCrystal(composition=composition)).to_raddose_txt()
    assert directives(text)["AbsCoefCalc"] == keyword


def test_average_composition_emits_no_cell_parameters():
    """Average takes no composition parameters; emitting them would error."""
    text = Experiment(
        crystal=CuboidCrystal(composition=AverageComposition())
    ).to_raddose_txt()
    assert "UnitCell" not in text
    assert "NumMonomers" not in text


def test_tophat_beam_has_no_fwhm():
    text = Experiment(beam=TophatBeam()).to_raddose_txt()
    assert blocks(text)["Beam"]["Type"] == "TopHat"
    assert "FWHM" not in text


def test_circular_collimation_is_emitted():
    text = Experiment(
        beam=GaussianBeam(collimation=Collimation(shape="circular", x=60, y=60))
    ).to_raddose_txt()
    assert directives(text)["Collimation"] == "Circular 60 60"


def test_escape_flags_are_only_emitted_when_enabled():
    off = Experiment().to_raddose_txt()
    assert "CalculatePEEscape" not in off
    assert "CalcSurrounding" not in off

    on = Experiment(
        crystal=CuboidCrystal(
            escape={"photoelectron": True, "fluorescence": True, "surrounding": True}
        )
    ).to_raddose_txt()
    assert "CalculatePEEscape TRUE" in on
    assert "CalculateFLEscape TRUE" in on
    assert "CalcSurrounding TRUE" in on


def test_multiple_wedges_each_emit_a_block():
    exp = Experiment(
        wedges=[
            Wedge(start=0, end=90, exposure_time=10),
            Wedge(start=90, end=180, exposure_time=20),
        ]
    )
    text = exp.to_raddose_txt()
    assert text.count("Wedge ") == 2
    assert "Wedge 0 90" in text
    assert "Wedge 90 180" in text


def test_optional_wedge_fields_are_omitted_when_unset():
    text = Experiment(wedges=[Wedge()]).to_raddose_txt()
    assert "StartOffset" not in text
    assert "TranslatePerDegree" not in text
    assert "RotAxBeamOffset" not in text


def test_optional_wedge_fields_are_emitted_when_set():
    text = Experiment(
        wedges=[
            Wedge(
                start_offset=(1.0, 2.0, 3.0),
                translate_per_degree=(0.1, 0.0, 0.3),
                rot_axis_beam_offset=5.0,
                angular_resolution=2.0,
            )
        ]
    ).to_raddose_txt()
    d = directives(text)
    assert d["StartOffset"] == "1 2 3"
    assert d["TranslatePerDegree"] == "0.1 0 0.3"
    assert d["RotAxBeamOffset"] == "5"
    assert d["AngularResolution"] == "2"


def test_rotation_angles_are_only_emitted_when_non_zero():
    assert "AngleP" not in Experiment().to_raddose_txt()
    text = Experiment(crystal=CuboidCrystal(angle_p=45.0)).to_raddose_txt()
    assert directives(text)["AngleP"] == "45"


@pytest.mark.parametrize("model", ["simple", "linear", "leal", "bfactor"])
def test_every_decay_model_is_emitted(model):
    text = Experiment(crystal=CuboidCrystal(decay_model=model)).to_raddose_txt()
    assert directives(text)["DiffractionDecayModel"].lower() == model
