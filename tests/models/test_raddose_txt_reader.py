"""Tests for reading RADDOSE-3D's ``.txt`` input format.

The format is specified by an ANTLR grammar in the Java repository
(``lib/antlrworks-parsergenerator/Inputfile.g``), and these tests are organised
by what that grammar says:

- **Round trip.** Every conformance case is written as ``.txt`` and read back
  to an identical experiment, so the reader is at least the exact inverse of
  the writer the Java reference runs are generated with.
- **Real files.** Every example input that ships with RADDOSE-3D is either read
  or refused, and each outcome is pinned with its reason.
- **Grammar quirks**, **defaults**, **requirements** and **refusals**, each
  taken from the grammar or the Java and named for the line that says so.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import yaml

from pyraddose_3d._raddose_txt import (
    RaddoseTxtError,
    RaddoseTxtWarning,
    parse_raddose_txt,
)
from pyraddose_3d.models.beam import Collimation, ElectronBeam
from pyraddose_3d.models.experiment import Experiment

DATA = Path(__file__).parent.parent / "data" / "raddose_txt"
CASES = Path(__file__).parent.parent / "conformance" / "cases"

CRYSTAL = """Crystal
Type Cuboid
Dimensions 20 16 12
UnitCell 78.02 78.02 78.02
NumMonomers 24
NumResidues 51
"""
BEAM = """Beam
Type Gaussian
Flux 2e12
FWHM 20 20
Energy 12.1
"""
WEDGE = """Wedge 0 90
ExposureTime 10
"""


def read(text: str) -> Experiment:
    return Experiment(**parse_raddose_txt(text, source="test.txt"))


def with_lines(block: str, *extra: str) -> str:
    return block + "".join(f"{line}\n" for line in extra)


# --- round trip -------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(p.name for p in CASES.iterdir() if p.is_dir()))
def test_every_conformance_case_reads_back_identically(case: str) -> None:
    """Writer then reader is the identity on the whole corpus.

    The corpus is what the Java reference runs are generated from, so this
    also says the reader understands exactly what the Java was given.

    Except the seed, which RADDOSE-3D takes from its command line
    (``--seed``), so a ``.txt`` file has nowhere to put it.
    """
    experiment = Experiment.from_yaml(CASES / case / "config.yaml")
    assert read(experiment.to_raddose_txt()) == experiment.model_copy(
        update={"seed": None}
    )


# --- the files RADDOSE-3D ships ----------------------------------------------

#: Example inputs from a RADDOSE-3D directory, and what happens to each. Its
#: repository tracks only the two SMXray inputs and the ``input.txt`` template.
SHIPPED = {
    "MyInput.txt": None,
    "MyInput_circular.txt": None,
    "MyInput_circular_tight.txt": None,
    "SMXray_example_input.txt": None,
    "SMXray2_example_input.txt": None,
    "input-b3x11.txt": None,
    # A stray "ß" after a number, which RADDOSE-3D refuses too.
    "input-b3x14.txt": "line 24: 'ß'",
    "input_MC.txt": None,
    # MicroED: a small molecule, and a protein from its unit cell.
    "ED_input.txt": None,
    "ED_input_cell.txt": None,
    # "SURROUNDINGELEMENTS C" has no amount; RADDOSE-3D refuses it too.
    "MyInput_hole.txt": "SURROUNDINGELEMENTS needs element and amount pairs",
    # The same, in a Monte Carlo run.
    "MyInput_hole_MC.txt": "SURROUNDINGELEMENTS needs element and amount pairs",
    # A template for RADDOSE-3D's GUI, with placeholders where values go: "?"
    # and "@" are characters no keyword or value can contain.
    "input.txt": r"line 9: '\?', line 47: '@'",
}


def test_every_shipped_example_is_accounted_for() -> None:
    assert set(SHIPPED) == {p.name for p in DATA.glob("*.txt")}


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_examples(name: str) -> None:
    """Read where the port can run them; refused, with the reason, where not."""
    refusal = SHIPPED[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RaddoseTxtWarning)
        if refusal is None:
            assert isinstance(Experiment.from_raddose_txt(DATA / name), Experiment)
        else:
            with pytest.raises(RaddoseTxtError, match=refusal):
                Experiment.from_raddose_txt(DATA / name)


def test_raddose_3ds_own_example_reads_as_written() -> None:
    """MyInput.txt, value by value, comments and all."""
    experiment = Experiment.from_raddose_txt(DATA / "MyInput.txt")
    crystal = experiment.crystal

    assert crystal.type == "cuboid"
    assert crystal.dimensions == (100.0, 100.0, 100.0)
    assert crystal.pixels_per_micron == 0.1
    assert crystal.composition.unit_cell == (78.02, 78.02, 78.02)
    assert crystal.composition.num_monomers == 24
    assert crystal.composition.num_residues == 51
    assert crystal.composition.protein_heavy_atoms == {"Zn": 0.333, "S": 6.0}
    assert crystal.composition.solvent_heavy_conc == {"P": 425.0}
    assert crystal.composition.solvent_fraction == 0.64

    assert experiment.beam.flux == 2e12
    assert experiment.beam.fwhm == (20.0, 70.0)
    assert experiment.beam.energy == 12.1
    assert experiment.beam.collimation.x == experiment.beam.collimation.y == 100.0

    assert [(w.start, w.end, w.exposure_time) for w in experiment.wedges] == [
        (0.0, 90.0, 50.0),
        (90.0, 180.0, 10.0),
    ]


def test_a_stray_character_is_refused() -> None:
    """RADDOSE-3D's lexer skipped it, so ``AngularResolution 0.5ß`` ran at 0.5;
    both refuse it now, naming every one."""
    with pytest.raises(RaddoseTxtError, match="line 24: 'ß'"):
        Experiment.from_raddose_txt(DATA / "input-b3x14.txt")
    with pytest.raises(RaddoseTxtError, match="line 1: '@', line 2: ','"):
        parse_raddose_txt("Crystal @\nType Cuboid,\n")


def test_a_byte_order_mark_and_a_final_comment_are_read() -> None:
    """Neither is an error, in RADDOSE-3D or here."""
    text = (DATA / "MyInput.txt").read_text(encoding="utf-8", errors="replace")
    plain = parse_raddose_txt(text)
    assert parse_raddose_txt("\ufeff" + text + "\n# no newline after this") == plain


# --- grammar quirks -----------------------------------------------------------


def test_values_may_continue_on_the_next_line() -> None:
    """Newlines are whitespace in the grammar, not separators."""
    text = CRYSTAL.replace("Dimensions 20 16 12", "Dimensions 20\n  16\n  12")
    assert read(text + BEAM + WEDGE).crystal.dimensions == (20.0, 16.0, 12.0)


@pytest.mark.parametrize("marker", ["#", "!", "//"])
def test_comment_markers(marker: str) -> None:
    text = with_lines(CRYSTAL, f"PixelsPerMicron 0.25 {marker} not 0.5")
    assert read(text + BEAM + WEDGE).crystal.pixels_per_micron == 0.25


def test_a_double_slash_inside_a_path_is_not_a_comment() -> None:
    """``/`` is part of a word, so ``//`` only starts a comment at a word's start."""
    text = CRYSTAL.replace(
        "UnitCell 78.02 78.02 78.02\nNumMonomers 24\nNumResidues 51\n",
        "AbsCoefCalc CIF\nCIF data//Fe3O4.cif\n",
    )
    assert read(text + BEAM + WEDGE).crystal.composition.cif == "data//Fe3O4.cif"


def test_a_comment_can_touch_a_value() -> None:
    text = with_lines(CRYSTAL, "PixelsPerMicron 0.25#no space")
    assert read(text + BEAM + WEDGE).crystal.pixels_per_micron == 0.25


def test_keywords_and_values_are_case_insensitive() -> None:
    text = (
        "crystal\n"
        + CRYSTAL.split("\n", 1)[1].upper()
        + "beam\n"
        + BEAM.split("\n", 1)[1].lower()
        + WEDGE.upper()
    )
    assert read(text) == read(CRYSTAL + BEAM + WEDGE)


@pytest.mark.parametrize(
    ("written", "meaning"),
    [
        ("AbsCoefCalc RD3D", "unit_cell"),
        ("CoefCalc RD3D", "unit_cell"),  # ABSCOEFCALC makes "Abs" optional
        ("AbsCoefCalc RD3", "unit_cell"),  # RDJAVA makes the final D optional
        ("AbsCoefCalc Default", "unit_cell"),
        ("AbsCoefCalc Average", "average"),
        ("AbsCoefCalc Dummy", "average"),
    ],
)
def test_abscoefcalc_spellings(written: str, meaning: str) -> None:
    text = with_lines(CRYSTAL, written)
    assert read(text + BEAM + WEDGE).crystal.composition.method == meaning


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("EXP", "pdb"), ("PDB", "pdb"), ("EXPSM", "cif"), ("CIF", "cif")],
)
def test_structure_file_keywords(keyword: str, value: str) -> None:
    """The tokens *named* PDB and CIF match "EXP" and "EXPSM"; both work."""
    text = CRYSTAL.replace(
        "UnitCell 78.02 78.02 78.02\nNumMonomers 24\nNumResidues 51\n",
        f"AbsCoefCalc {keyword}\n{value.upper()} model.{value}\n",
    )
    assert read(text + BEAM + WEDGE).crystal.composition.method == value


def test_other_keyword_spellings() -> None:
    text = CRYSTAL.replace("Dimensions", "Dimension") + "DDM Linear\n" + BEAM + WEDGE
    assert read(text).crystal.decay_model == "linear"


def test_energy_may_carry_its_unit() -> None:
    text = CRYSTAL + BEAM.replace("Energy 12.1", "Energy 12.1 keV") + WEDGE
    assert read(text).beam.energy == 12.1


@pytest.mark.parametrize(
    ("lines", "shape", "x", "y"),
    [
        (["Collimation Rectangular 40 30"], "rectangular", 40, 30),
        (["Collimation", "Rectangular 40 30"], "rectangular", 40, 30),
        (["Rectangular 40 30"], "rectangular", 40, 30),
        (["Collimation Circular 40 30"], "circular", 40, 30),
        (
            ["Collimation Horizontal 40", "Collimation Vertical 30"],
            "rectangular",
            40,
            30,
        ),
    ],
)
def test_collimation_forms(lines: list[str], shape: str, x: float, y: float) -> None:
    """``Collimation`` is a rule of its own that matches with no values."""
    collimation = read(CRYSTAL + with_lines(BEAM, *lines) + WEDGE).beam.collimation
    assert (collimation.shape, collimation.x, collimation.y) == (shape, x, y)


def test_one_sided_collimation_is_refused() -> None:
    with pytest.raises(RaddoseTxtError, match="only one side"):
        read(CRYSTAL + with_lines(BEAM, "Collimation Horizontal 40") + WEDGE)


def test_a_two_component_offset_has_zero_z() -> None:
    """A missing z is left unset in the Java, which means zero."""
    wedge = read(CRYSTAL + BEAM + with_lines(WEDGE, "StartOffset 1 2")).wedges[0]
    assert wedge.start_offset == (1.0, 2.0, 0.0)


def test_a_unit_cell_may_give_its_angles() -> None:
    text = CRYSTAL.replace("UnitCell 78.02 78.02 78.02", "UnitCell 50 60 70 90 100 120")
    composition = read(text + BEAM + WEDGE).crystal.composition
    assert composition.unit_cell == (50.0, 60.0, 70.0)
    assert composition.unit_cell_angles == (90.0, 100.0, 120.0)


@pytest.mark.parametrize("count", [4, 5])
def test_a_unit_cell_with_some_angles_is_refused(count: int) -> None:
    values = " ".join(["50"] * count)
    text = CRYSTAL.replace("UnitCell 78.02 78.02 78.02", f"UnitCell {values}")
    with pytest.raises(RaddoseTxtError, match=f"found {count} numbers"):
        read(text + BEAM + WEDGE)


def test_counts_must_be_whole_numbers() -> None:
    """``Integer.parseInt`` refuses "24.0", so the Java does too."""
    text = CRYSTAL.replace("NumMonomers 24", "NumMonomers 24.0")
    with pytest.raises(RaddoseTxtError, match=r"Integer\.parseInt"):
        read(text + BEAM + WEDGE)


def test_an_element_listed_twice_is_refused() -> None:
    text = with_lines(CRYSTAL, "ProteinHeavyAtoms S 6 Zn 1 S 2")
    with pytest.raises(RaddoseTxtError, match="S appears twice"):
        read(text + BEAM + WEDGE)


def test_a_repeated_keyword_replaces_the_earlier_one() -> None:
    """As ``properties.put`` does."""
    text = CRYSTAL + with_lines(BEAM, "Flux 5e11") + WEDGE
    assert read(text).beam.flux == 5e11


# --- RADDOSE-3D's defaults ------------------------------------------------------


def test_defaults_for_omitted_settings() -> None:
    experiment = read(CRYSTAL + BEAM + WEDGE)

    assert experiment.crystal.pixels_per_micron is None
    assert experiment.crystal.base_pixels_per_micron == pytest.approx(10 / 20)
    assert experiment.crystal.composition.method == "unit_cell"
    assert experiment.crystal.decay_model == "simple"
    assert not experiment.crystal.escape.any_enabled
    assert experiment.beam.collimation is None
    assert experiment.wedges[0].angular_resolution is None
    assert experiment.wedges[0].max_resolution == 2.0


@pytest.mark.parametrize(
    ("shape", "dimensions", "first"),
    [
        ("Spherical", "40", 40.0),
        ("Cylinder", "25 8", 25.0),
    ],
)
def test_default_resolution_follows_the_first_dimension(
    shape: str, dimensions: str, first: float
) -> None:
    text = CRYSTAL.replace("Type Cuboid", f"Type {shape}").replace(
        "Dimensions 20 16 12", f"Dimensions {dimensions}"
    )
    crystal = read(text + BEAM + WEDGE).crystal
    assert crystal.base_pixels_per_micron == pytest.approx(10 / first)


@pytest.mark.parametrize(
    "line",
    [
        "PolarisationDirection 90",
        "SurroundingThickness 1 1 1",
        "SimPhotons 1000",
        "Runs 3",
    ],
)
def test_simulation_settings_are_ignored_by_the_mx_calculation(line: str) -> None:
    """Read only by the Monte Carlo and XFEL subprograms, as Crystal.java shows."""
    text = with_lines(CRYSTAL, line, "Subprogram RD3D")
    with pytest.warns(RaddoseTxtWarning, match=f"{line.split()[0]} is ignored"):
        assert read(text + BEAM + WEDGE) == read(CRYSTAL + BEAM + WEDGE)


def test_escape_and_the_surrounding_are_read() -> None:
    text = with_lines(
        CRYSTAL,
        "CalculatePEEscape TRUE",
        "CalcSurrounding TRUE",
        "GoniometerAxis 90",
        "SurroundingHeavyConc Na 150 Cl 150",
        "DensityBased TRUE",
        "SurroundingElements C 1 H 2",
        "SurroundingDensity 0.85",
    )
    escape = read(text + BEAM + WEDGE).crystal.escape
    assert escape.photoelectron and escape.surrounding
    assert escape.goniometer_axis == 90.0
    assert escape.surrounding_heavy_conc == {"Na": 150.0, "Cl": 150.0}
    assert escape.density_based
    assert escape.surrounding_elements == {"C": 1.0, "H": 2.0}
    assert escape.surrounding_density == 0.85


SAXS_CRYSTAL = """Crystal
Type Cuboid
Dimensions 20 16 12
AbsCoefCalc SAXSSEQ
SeqFile insulin.fasta
ProteinConc 10
"""


def test_saxs_from_a_sequence_is_read() -> None:
    composition = read(SAXS_CRYSTAL + BEAM + WEDGE).crystal.composition
    assert composition.method == "saxs_sequence"
    assert composition.sequence_file == "insulin.fasta"
    assert composition.protein_concentration == 10.0


@pytest.mark.parametrize("coefcalc", ["SAXS", "SAXSSEQ"])
def test_a_saxs_box_defaults_to_1000_angstroms(coefcalc: str) -> None:
    """As CoefCalcSAXS, which takes 1000 for any edge not given."""
    text = SAXS_CRYSTAL.replace("SAXSSEQ", coefcalc) + "NumResidues 51\n"
    composition = read(text + BEAM + WEDGE).crystal.composition
    assert composition.unit_cell == (1000.0, 1000.0, 1000.0)
    emitted = read(text + BEAM + WEDGE).to_raddose_txt()
    assert "UnitCell" not in emitted
    assert read(emitted) == read(text + BEAM + WEDGE)


def test_saxs_from_a_sequence_reads_numcarb() -> None:
    """RADDOSE-3D used to read it here and never add the carbohydrate."""
    text = with_lines(SAXS_CRYSTAL, "NumCarb 4") + BEAM + WEDGE
    composition = read(text).crystal.composition
    assert composition.num_carb == 4
    assert read(read(text).to_raddose_txt()) == read(text)


def test_a_pink_beam_is_read() -> None:
    assert (
        read(CRYSTAL + with_lines(BEAM, "EnergyFWHM 0.1") + WEDGE).beam.energy_fwhm
        == 0.1
    )


# --- what RADDOSE-3D requires -------------------------------------------------


@pytest.mark.parametrize(
    ("remove", "named"),
    [
        ("Type Cuboid\n", "Type is required"),
        ("Dimensions 20 16 12\n", "Dimensions is required"),
        ("UnitCell 78.02 78.02 78.02\n", "UnitCell is required"),
        ("NumMonomers 24\n", "NumMonomers is required"),
        ("Flux 2e12\n", "Flux is required"),
        ("Energy 12.1\n", "Energy is required"),
        ("FWHM 20 20\n", "FWHM is required"),
        ("ExposureTime 10\n", "ExposureTime is required"),
    ],
)
def test_what_raddose_3d_requires_is_required(remove: str, named: str) -> None:
    text = (CRYSTAL + BEAM + WEDGE).replace(remove, "")
    with pytest.raises(RaddoseTxtError, match=named):
        read(text)


def test_errors_name_the_file_and_line() -> None:
    text = CRYSTAL + BEAM.replace("Flux 2e12", "Flux lots") + WEDGE
    with pytest.raises(
        RaddoseTxtError, match=r"^test\.txt, line 9: Flux needs a number"
    ):
        read(text)


# --- what this port refuses -----------------------------------------------------


@pytest.mark.parametrize(
    ("block", "line", "reason"),
    [
        ("crystal", "Subprogram EMSP", "MicroED needs a TopHat beam"),
        ("crystal", "AbsCoefCalc RDV2", "Fortran"),
    ],
)
def test_unported_calculations_are_refused_with_the_reason(
    block: str, line: str, reason: str
) -> None:
    """Never dropped: dropping one would compute something else without a word."""
    crystal = with_lines(CRYSTAL, line) if block == "crystal" else CRYSTAL
    beam = with_lines(BEAM, line) if block == "beam" else BEAM
    with pytest.raises(RaddoseTxtError, match=reason):
        read(crystal + beam + WEDGE)


def test_type_experimental_is_refused() -> None:
    """RADDOSE-3D refuses it too, pointing to ExperimentalPGM."""
    with pytest.raises(RaddoseTxtError, match="use Type ExperimentalPGM"):
        read(CRYSTAL + BEAM.replace("Type Gaussian", "Type Experimental") + WEDGE)


PGM_BEAM = """Beam
Type ExperimentalPGM
File beam.pgm
PixelSize 1.5 1.4
Flux 2e12
Energy 12.1
"""


def test_an_experimental_beam_is_read() -> None:
    beam = read(CRYSTAL + PGM_BEAM + WEDGE).beam
    assert beam.type == "experimental"
    assert (beam.file, beam.pixel_size, beam.collimation) == (
        "beam.pgm",
        (1.5, 1.4),
        None,
    )


@pytest.mark.parametrize("shape", ["Rectangular", "Circular"])
def test_an_experimental_beam_takes_a_collimation(shape: str) -> None:
    """With its sizes, which RADDOSE-3D used to ignore."""
    text = with_lines(PGM_BEAM, f"Collimation {shape} 12 10")
    beam = read(CRYSTAL + text + WEDGE).beam
    assert beam.collimation is not None
    assert (beam.collimation.shape, beam.collimation.x, beam.collimation.y) == (
        shape.lower(),
        12.0,
        10.0,
    )
    assert read(read(CRYSTAL + text + WEDGE).to_raddose_txt()).beam == beam


@pytest.mark.parametrize(
    ("remove", "named"), [("File", "File"), ("PixelSize", "PixelSize")]
)
def test_an_experimental_beam_needs_its_image(remove: str, named: str) -> None:
    text = "".join(
        line + "\n" for line in PGM_BEAM.splitlines() if not line.startswith(remove)
    )
    with pytest.raises(RaddoseTxtError, match=f"{named} is required"):
        read(CRYSTAL + text + WEDGE)


def test_small_molecule_refuses_solvent() -> None:
    """RADDOSE-3D itself refuses this; SmallMole has no solvent model."""
    text = CRYSTAL.replace(
        "NumResidues 51\n",
        "AbsCoefCalc SmallMole\nSmallMoleAtoms C 6 H 6\nSolventHeavyConc Na 500\n",
    )
    with pytest.raises(RaddoseTxtError, match="no solvent model"):
        read(text + BEAM + WEDGE)


def test_an_unknown_keyword_is_refused() -> None:
    with pytest.raises(
        RaddoseTxtError, match="'Fluxx' is not a keyword of a Beam block"
    ):
        read(CRYSTAL + with_lines(BEAM, "Fluxx 2") + WEDGE)


def test_a_keyword_in_the_wrong_block_is_refused() -> None:
    with pytest.raises(
        RaddoseTxtError, match="'Flux' is not a keyword of a Crystal block"
    ):
        read(with_lines(CRYSTAL, "Flux 2e12") + BEAM + WEDGE)


# --- block order ------------------------------------------------------------------


def test_beam_may_come_before_crystal() -> None:
    assert read(BEAM + CRYSTAL + WEDGE) == read(CRYSTAL + BEAM + WEDGE)


def test_several_wedges() -> None:
    text = CRYSTAL + BEAM + WEDGE + "Wedge 90 180\nExposureTime 20\n"
    assert [w.end for w in read(text).wedges] == [90.0, 180.0]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (WEDGE + CRYSTAL + BEAM, "needs a Crystal and a Beam before it"),
        (CRYSTAL + BEAM + WEDGE + BEAM, "change the beam part way through"),
        (CRYSTAL + BEAM + WEDGE + CRYSTAL, "change the crystal part way through"),
        (CRYSTAL + CRYSTAL + BEAM + WEDGE, "a second Crystal block"),
        (CRYSTAL + BEAM, "no Wedge block"),
        ("Flux 2e12\n" + CRYSTAL + BEAM + WEDGE, "expected Crystal, Beam or Wedge"),
    ],
)
def test_block_order(text: str, reason: str) -> None:
    """RADDOSE-3D exposes each wedge with whatever crystal and beam are current.

    So a file can change the beam between wedges. An experiment here has one
    of each, and that is refused rather than silently flattened.
    """
    with pytest.raises(RaddoseTxtError, match=reason):
        read(text)


# --- choosing the format ------------------------------------------------------------


def test_from_file_chooses_by_extension(tmp_path: Path) -> None:
    experiment = read(CRYSTAL + BEAM + WEDGE)
    (tmp_path / "in.txt").write_text(CRYSTAL + BEAM + WEDGE)
    experiment.to_yaml(tmp_path / "in.yaml")
    (tmp_path / "in.yml").write_text((tmp_path / "in.yaml").read_text())
    (tmp_path / "in").write_text(CRYSTAL + BEAM + WEDGE)

    for name in ("in.txt", "in.yaml", "in.yml", "in"):
        assert Experiment.from_file(tmp_path / name) == experiment, name


def test_a_txt_file_converts_to_yaml(tmp_path: Path) -> None:
    """The migration path: read the old file, write YAML, read it back."""
    experiment = Experiment.from_raddose_txt(DATA / "MyInput.txt")
    experiment.to_yaml(tmp_path / "MyInput.yaml")

    assert Experiment.from_yaml(tmp_path / "MyInput.yaml") == experiment
    assert (
        yaml.safe_load((tmp_path / "MyInput.yaml").read_text())["beam"]["flux"] == 2e12
    )
    assert math.isclose(experiment.crystal.pixels_per_micron, 0.1)


# --- containers -------------------------------------------------------------------


def test_containers_read_both_ways() -> None:
    mixture = read(
        with_lines(
            CRYSTAL,
            "ContainerMaterialType Mixture",
            "ContainerMaterialMixture mylar",
            "ContainerThickness 100",
            "ContainerDensity 1.4",
        )
        + BEAM
        + WEDGE
    ).crystal.container
    assert (mixture.type, mixture.material, mixture.thickness) == (
        "mixture",
        "mylar",
        100,
    )

    elemental = read(
        with_lines(
            CRYSTAL,
            "MaterialType Elemental",
            "MaterialElements C 22 H 10 N 2 O 5",
            "ContainerThickness 50",
            "ContainerDensity 1.42",
        )
        + BEAM
        + WEDGE
    ).crystal.container
    assert elemental.elements == {"C": 22, "H": 10, "N": 2, "O": 5}


def test_container_type_none_means_no_container() -> None:
    text = with_lines(CRYSTAL, "ContainerMaterialType None")
    assert read(text + BEAM + WEDGE).crystal.container is None


def test_a_partly_specified_container_is_no_container() -> None:
    """``ContainerSemiTransparent``: "Assuming no container around sample."

    RADDOSE-3D carries on without the container. So does this, but with a
    warning rather than a line on the console that is easy to miss.
    """
    text = with_lines(
        CRYSTAL,
        "ContainerMaterialType Mixture",
        "ContainerMaterialMixture mylar",
        "ContainerThickness 100",
    )
    with pytest.warns(RaddoseTxtWarning, match="no ContainerDensity"):
        assert read(text + BEAM + WEDGE).crystal.container is None


def test_container_settings_without_a_type_are_ignored() -> None:
    text = with_lines(CRYSTAL, "ContainerThickness 100", "ContainerDensity 1.4")
    with pytest.warns(RaddoseTxtWarning, match="without ContainerMaterialType"):
        assert read(text + BEAM + WEDGE).crystal.container is None


def test_an_elemental_container_needs_its_elements() -> None:
    """The Java builds the name from the element list first, and fails there."""
    text = with_lines(
        CRYSTAL,
        "ContainerMaterialType Elemental",
        "ContainerThickness 50",
        "ContainerDensity 1.42",
    )
    with pytest.raises(RaddoseTxtError, match="needs ContainerMaterialElements"):
        read(text + BEAM + WEDGE)


def test_a_mixture_nist_does_not_tabulate_is_refused() -> None:
    """Kapton is the obvious one to try, and NIST does not have it."""
    from pydantic import ValidationError

    text = with_lines(
        CRYSTAL,
        "ContainerMaterialType Mixture",
        "ContainerMaterialMixture kapton",
        "ContainerThickness 50",
        "ContainerDensity 1.42",
    )
    with pytest.raises(ValidationError, match="NIST does not tabulate 'kapton'"):
        read(text + BEAM + WEDGE)


# --- MicroED: Subprogram EMED -----------------------------------------------

MICROED_CRYSTAL = with_lines(CRYSTAL, "Subprogram EMED")
ELECTRON_BEAM = """Beam
Type TopHat
Energy 300
Exposure 1
Collimation Rectangular 4 4
"""


def test_subprogram_emed_reads_an_electron_beam() -> None:
    """A TopHat beam with an Exposure, in electrons per square Angstrom."""
    experiment = read(MICROED_CRYSTAL + ELECTRON_BEAM + WEDGE)
    assert experiment.crystal.subprogram == "microed"
    assert experiment.beam == ElectronBeam(
        energy=300.0,
        exposure=1.0,
        collimation=Collimation(shape="rectangular", x=4.0, y=4.0),
    )


def test_emsp_runs_the_same_engine() -> None:
    """``MicroED.CalculateEM`` serves both keywords."""
    crystal = with_lines(CRYSTAL, "Subprogram EMSP")
    assert read(crystal + ELECTRON_BEAM + WEDGE).crystal.subprogram == "microed"


def test_a_microed_beam_may_come_before_its_crystal() -> None:
    """The beam is built once the crystal says which engine runs."""
    experiment = read(ELECTRON_BEAM + MICROED_CRYSTAL + WEDGE)
    assert experiment.beam.type == "electron"


@pytest.mark.parametrize(
    ("profile", "why"),
    [
        ("Type Gaussian\nFWHM 2 2", "this beam is Gaussian"),
        ("Type ExperimentalPGM\nFile beam.pgm\nPixelSize 1 1", "needs a TopHat beam"),
    ],
)
def test_microed_takes_its_exposure_from_a_tophat_only(profile: str, why: str) -> None:
    beam = f"Beam\n{profile}\nEnergy 300\nExposure 1\n"
    with pytest.raises(RaddoseTxtError, match=why):
        read(MICROED_CRYSTAL + beam + WEDGE)


def test_microed_needs_an_exposure() -> None:
    """RADDOSE-3D refuses a TopHat beam without one too."""
    beam = "Beam\nType TopHat\nEnergy 300\nCollimation Rectangular 4 4\n"
    with pytest.raises(RaddoseTxtError, match="Exposure is required"):
        read(MICROED_CRYSTAL + beam + WEDGE)


@pytest.mark.parametrize(
    "extra",
    [
        "Flux 1e12",
        "EnergyFWHM 0.1",
        "SemiAngle 10",
        "ApertureRadius 5",
        "PulseEnergy 1",
    ],
)
def test_what_microed_ignores_is_warned_about(extra: str) -> None:
    """RADDOSE-3D accepts these silently and never reads them."""
    keyword = extra.split()[0]
    with pytest.warns(RaddoseTxtWarning, match=f"{keyword} is ignored"):
        experiment = read(MICROED_CRYSTAL + with_lines(ELECTRON_BEAM, extra) + WEDGE)
    assert experiment.beam.type == "electron"


@pytest.mark.parametrize("line", ["SimElectrons 1000", "Runs 2"])
def test_monte_carlo_settings_are_ignored_by_microed(line: str) -> None:
    """MicroED's Monte Carlo is commented out in RADDOSE-3D.

    The grammar's ``wholeNumber`` calls both spellings ``SimPhotons``.
    """
    label = {"SimElectrons": "SimPhotons"}.get(line.split()[0], line.split()[0])
    with pytest.warns(RaddoseTxtWarning, match=f"{label} is ignored.*disabled"):
        read(with_lines(MICROED_CRYSTAL, line) + ELECTRON_BEAM + WEDGE)


def test_an_exposure_belongs_to_microed() -> None:
    with pytest.raises(RaddoseTxtError, match="belongs to a MicroED beam"):
        read(CRYSTAL + with_lines(BEAM, "Exposure 1") + WEDGE)


# --- Monte Carlo and XFEL: Subprogram MONTECARLO, GOS, XFEL ----------------------

MC_CRYSTAL = with_lines(CRYSTAL, "Subprogram MonteCarlo")
XFEL_CRYSTAL = with_lines(CRYSTAL, "Subprogram XFEL")
XFEL_BEAM = BEAM.replace("Flux 2e12", "PulseEnergy 1.4")


@pytest.mark.parametrize(
    ("written", "subprogram"),
    [
        ("MonteCarlo", "montecarlo"),
        ("MONTECARLO", "montecarlo"),
        ("gos", "gos"),
        ("XFEL", "xfel"),
    ],
)
def test_the_simulated_subprograms_are_read(written: str, subprogram: str) -> None:
    """Crystal.java upper-cases the value before comparing it."""
    crystal = with_lines(CRYSTAL, f"Subprogram {written}")
    beam = XFEL_BEAM if subprogram == "xfel" else BEAM
    experiment = read(crystal + beam + WEDGE)
    assert experiment.crystal.subprogram == subprogram
    assert experiment.crystal.simulation.photons == 1_000_000
    assert experiment.crystal.simulation.runs == 1


def test_the_simulation_settings_are_read() -> None:
    text = with_lines(
        MC_CRYSTAL,
        "CalcSurrounding TRUE",
        "SimPhotons 20000",
        "Runs 3",
        "PolarisationDirection 90",
        "SurroundingThickness 1 2 0.5",
    )
    simulation = read(text + BEAM + WEDGE).crystal.simulation
    assert simulation.photons == 20_000
    assert simulation.runs == 3
    assert simulation.polarisation_direction == 90.0
    assert simulation.surrounding_thickness == (1.0, 2.0, 0.5)


@pytest.mark.parametrize(
    ("written", "photons"),
    [("1000000", 1_000_000), ("1e6", 1_000_000), ("1000000.0", 1_000_000)],
)
def test_photons_are_whole_numbers_in_any_notation(written: str, photons: int) -> None:
    """The grammar's ``wholeNumber``: ``BigDecimal.longValueExact``."""
    for keyword in ("SimPhotons", "SimElectrons"):
        text = with_lines(MC_CRYSTAL, f"{keyword} {written}")
        assert read(text + BEAM + WEDGE).crystal.simulation.photons == photons


@pytest.mark.parametrize(
    "line", ["SimPhotons 1.5", "SimPhotons 0", "Runs 2.5", "Runs -1", "Runs 3e9"]
)
def test_fractions_and_counts_out_of_range_are_refused(line: str) -> None:
    """``Runs`` must fit an ``int``, and both must be at least 1."""
    with pytest.raises(RaddoseTxtError, match="needs a whole number from 1 to"):
        read(with_lines(MC_CRYSTAL, line) + BEAM + WEDGE)


def test_a_surrounding_thickness_without_a_surrounding_is_ignored() -> None:
    """Crystal.java reads it only when the composition has a surrounding."""
    text = with_lines(MC_CRYSTAL, "SurroundingThickness 1 1 1")
    with pytest.warns(RaddoseTxtWarning, match="SurroundingThickness is ignored"):
        experiment = read(text + BEAM + WEDGE)
    assert experiment.crystal.simulation.surrounding_thickness is None


def test_xfel_takes_a_pulse_energy_rather_than_a_flux() -> None:
    """XFEL.java takes its photons from ``getPulseEnergy``, never the flux."""
    beam = read(XFEL_CRYSTAL + XFEL_BEAM + WEDGE).beam
    assert beam.pulse_energy == 1.4
    assert beam.flux is None
    with pytest.warns(RaddoseTxtWarning, match="Flux is ignored"):
        beam = read(XFEL_CRYSTAL + with_lines(BEAM, "PulseEnergy 1.4") + WEDGE).beam
    assert (beam.flux, beam.pulse_energy) == (None, 1.4)


def test_xfel_needs_a_pulse_energy() -> None:
    with pytest.raises(RaddoseTxtError, match="PulseEnergy is required"):
        read(XFEL_CRYSTAL + BEAM + WEDGE)


@pytest.mark.parametrize("crystal", [CRYSTAL, MC_CRYSTAL])
def test_a_pulse_energy_is_ignored_by_every_other_engine(crystal: str) -> None:
    with pytest.warns(RaddoseTxtWarning, match="PulseEnergy is ignored"):
        beam = read(crystal + with_lines(BEAM, "PulseEnergy 1.4") + WEDGE).beam
    assert (beam.flux, beam.pulse_energy) == (2e12, None)
    with pytest.raises(RaddoseTxtError, match="Flux is required"):
        read(crystal + XFEL_BEAM + WEDGE)


def test_the_simulated_subprograms_refuse_an_experimental_beam() -> None:
    """SubprogramInputs.check: every photon started at (0, 0)."""
    beam = """Beam
Type ExperimentalPGM
File beam.pgm
PixelSize 1 1
Flux 2e12
Energy 12.1
"""
    with pytest.raises(RaddoseTxtError, match="needs a TopHat or Gaussian beam"):
        read(MC_CRYSTAL + beam + WEDGE)


def test_the_simulated_subprograms_refuse_the_average_composition() -> None:
    """SubprogramInputs.check: they choose elements, and Average has none."""
    crystal = """Crystal
Type Cuboid
Dimensions 20 16 12
AbsCoefCalc Average
Subprogram GOS
"""
    with pytest.raises(RaddoseTxtError, match=r"line 4: .*AbsCoefCalc Average"):
        read(crystal + BEAM + WEDGE)


@pytest.mark.parametrize("subprogram", ["montecarlo", "gos", "xfel"])
def test_simulated_experiments_read_back_identically(subprogram: str) -> None:
    """The emitter writes whole numbers, so old RADDOSE-3D reads its files too."""
    experiment = read(
        with_lines(
            CRYSTAL,
            f"Subprogram {subprogram}",
            "CalcSurrounding TRUE",
            "SimPhotons 2e4",
            "Runs 2",
            "PolarisationDirection 90",
            "SurroundingThickness 1 2 0.5",
        )
        + (XFEL_BEAM if subprogram == "xfel" else BEAM)
        + WEDGE
    )
    text = experiment.to_raddose_txt()
    assert "SimPhotons 20000\n" in text
    assert "Runs 2\n" in text
    assert read(text) == experiment


def test_xfel_reads_a_pulses_shape_and_window() -> None:
    beam = read(
        XFEL_CRYSTAL
        + with_lines(XFEL_BEAM, "PulseShape Gaussian", "PulseWindow 3")
        + WEDGE
    ).beam
    assert (beam.pulse_shape, beam.pulse_window) == ("gaussian", 3.0)
    beam = read(XFEL_CRYSTAL + with_lines(XFEL_BEAM, "PulseShape TOPHAT") + WEDGE).beam
    assert (beam.pulse_shape, beam.pulse_window) == ("tophat", None)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (("PulseShape Lorentzian",), "not a pulse shape"),
        (("PulseWindow 2",), "a top-hat pulse lasts the exposure time"),
        (("PulseShape TopHat", "PulseWindow 2"), "a top-hat pulse lasts"),
        (("PulseShape Gaussian", "PulseWindow 0"), "more than 0 FWHMs"),
    ],
)
def test_a_pulses_shape_and_window_are_checked(
    lines: tuple[str, ...], message: str
) -> None:
    with pytest.raises(RaddoseTxtError, match=message):
        read(XFEL_CRYSTAL + with_lines(XFEL_BEAM, *lines) + WEDGE)


def test_a_pulse_shape_is_refused_by_the_monte_carlo_and_ignored_by_mx() -> None:
    """SubprogramInputs.check refuses it; MX never reads it."""
    with pytest.raises(RaddoseTxtError, match="describes an XFEL pulse"):
        read(MC_CRYSTAL + with_lines(BEAM, "PulseShape Gaussian") + WEDGE)
    with pytest.warns(RaddoseTxtWarning, match="PulseShape is ignored"):
        beam = read(CRYSTAL + with_lines(BEAM, "PulseShape Gaussian") + WEDGE).beam
    assert beam.pulse_shape is None


def test_a_gaussian_pulse_is_written_back() -> None:
    experiment = read(
        XFEL_CRYSTAL
        + with_lines(XFEL_BEAM, "PulseShape Gaussian", "PulseWindow 3")
        + WEDGE
    )
    text = experiment.to_raddose_txt()
    assert "PulseShape Gaussian" in text and "PulseWindow 3" in text
    assert read(text) == experiment


def test_a_cylinders_third_dimension_is_ignored_with_a_warning() -> None:
    """As CrystalCylinder reads it: the first two values, the diameter and the
    length. The crystal is a cylinder all the same."""
    text = CRYSTAL.replace("Type Cuboid", "Type Cylinder").replace(
        "Dimensions 20 16 12", "Dimensions 40 20 7"
    )
    with pytest.warns(RaddoseTxtWarning, match=r"third value of Dimensions \(7 um\)"):
        crystal = read(text + BEAM + WEDGE).crystal
    assert (crystal.type, crystal.diameter, crystal.height) == ("cylinder", 40.0, 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RaddoseTxtWarning)
        read(text.replace("Dimensions 40 20 7", "Dimensions 40 20 20") + BEAM + WEDGE)
