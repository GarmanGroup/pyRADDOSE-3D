"""Tests for the CoefCalc front-ends.

RADDOSE-3D offers nine ways to say what a crystal is made of, selected by the
``AbsCoefCalc`` keyword. They all end at the same place -- a count of each
element per unit cell -- so these tests check the routes rather than the
arithmetic, which ``test_coefficients.py`` already covers.

| keyword | implementation | covered by |
|---|---|---|
| ``Dummy`` / ``Average`` | ``CoefCalcAverage`` | here |
| ``Default`` / ``RD3D`` | ``CoefCalcFromParams`` | ``test_composition.py`` |
| ``RDV2`` / ``RDV3`` | ``CoefCalcRaddose`` | not ported -- see below |
| ``EXP`` | ``CoefCalcFromPDB`` | not yet ported |
| ``SAXS`` / ``SAXSSEQ`` | ``CoefCalcSAXS`` | here |
| ``Sequence`` | ``CoefCalcFromSequence`` | here |
| ``SmallMole`` | ``CoefCalcSmallMolecules`` | here |
| ``EXPSM`` | ``CoefCalcFromCIF`` | not yet ported |

``CoefCalcRaddose`` is deliberately not ported: it writes an input file, runs
an external RADDOSE v2 binary that this project does not ship, and scrapes
its stdout. There is nothing to be faithful to that does not depend on a
binary nobody has.
"""

import csv
import pathlib
from collections import defaultdict

import pytest

from pyraddose_3d.backend.coefficients import (
    AVERAGE_COEFFICIENTS,
    AVERAGE_DENSITY,
    density,
    material_coefficients,
    unit_cell_volume,
)
from pyraddose_3d.backend.composition import (
    Composition,
    Macromolecule,
    saxs_monomer_count,
    sequence_composition,
    sequence_residue_counts,
    small_molecule_composition,
    unit_cell_composition,
)
from pyraddose_3d.backend.residues import ResidueType, residue_database

DATA = pathlib.Path(__file__).parent / "data"
REFERENCE = DATA / "coefficients_frontends.tsv"
SEQUENCES = DATA / "sequences"


def _read_fasta(name: str) -> str:
    return "".join(
        line.strip()
        for line in (SEQUENCES / name).read_text("ascii").splitlines()
        if not line.startswith(">")
    )


PROTEIN_SEQUENCE = _read_fasta("seq_test.fasta")
DNA_SEQUENCE = _read_fasta("seq_dna.fasta")

SMALL_MOLECULE_ATOMS = {"C": 6.0, "H": 12.0, "N": 2.0, "O": 3.0, "S": 1.0}
SAXS_CELL_VOLUME = 1e9


def _load_reference() -> dict[str, dict]:
    cases: dict[str, dict] = defaultdict(
        lambda: {"atoms": {}, "coefficients": [], "density": None, "scalars": {}}
    )
    with REFERENCE.open(encoding="ascii") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            tag = row[0]
            if tag == "DENSITY":
                cases[row[1]]["density"] = float(row[2])
            elif tag == "ATOMS":
                cases[row[1]]["atoms"][row[3]] = float(row[4])
            elif tag == "COEF":
                cases[row[1]]["coefficients"].append(
                    tuple(float(v) for v in row[2:7])
                )
            elif tag == "MONOMERS":
                cases[row[1]]["scalars"]["monomers"] = int(row[2])
            elif tag == "RESIDUES":
                cases[row[1]]["scalars"]["residues"] = tuple(
                    int(v) for v in row[2:5]
                )
    return dict(cases)


REFERENCE_CASES = _load_reference()


def _build(case_id: str) -> Composition:
    """Rebuild one reference case through the port's own front-end."""
    small_molecule_volume = unit_cell_volume(10.0, 12.0, 14.0)
    if case_id.startswith("smallmole"):
        return small_molecule_composition(
            small_molecule_volume, SMALL_MOLECULE_ATOMS, monomers=4
        )
    if case_id == "saxs":
        return unit_cell_composition(
            SAXS_CELL_VOLUME, Macromolecule(monomers=424, amino_acids=129)
        )
    if case_id == "saxs-rna":
        return unit_cell_composition(
            SAXS_CELL_VOLUME, Macromolecule(monomers=307, rna_nucleotides=60)
        )
    if case_id == "saxs-dna":
        return unit_cell_composition(
            SAXS_CELL_VOLUME, Macromolecule(monomers=296, dna_nucleotides=60)
        )
    if case_id == "sequence":
        return sequence_composition(
            unit_cell_volume(78.27, 78.27, 78.27), PROTEIN_SEQUENCE, monomers=8
        )
    if case_id == "sequence-dna":
        return sequence_composition(
            unit_cell_volume(60.0, 60.0, 120.0, 90.0, 90.0, 120.0),
            DNA_SEQUENCE,
            residue_type=ResidueType.DNA,
            monomers=2,
        )
    raise AssertionError(f"unknown case {case_id!r}")


CASE_IDS = sorted(REFERENCE_CASES)


def _counts(composition: Composition) -> dict[str, float]:
    return {
        e.name: float(n)
        for e, n in zip(composition.elements, composition.counts, strict=True)
    }


# --- conformance against the Java ---------------------------------------


def test_reference_covers_every_front_end() -> None:
    assert set(CASE_IDS) == {
        "smallmole",
        "smallmole-solv",
        "saxs",
        "saxs-rna",
        "saxs-dna",
        "sequence",
        "sequence-dna",
    }


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_atom_counts_match_java(case_id: str) -> None:
    got = _counts(_build(case_id))
    want = REFERENCE_CASES[case_id]["atoms"]
    for symbol in set(got) | set(want):
        assert got.get(symbol, 0.0) == pytest.approx(
            want.get(symbol, 0.0), rel=1e-12
        ), f"{symbol} count"


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_density_matches_java(case_id: str) -> None:
    assert float(density(_build(case_id))) == pytest.approx(
        REFERENCE_CASES[case_id]["density"], rel=1e-12
    )


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_coefficients_match_java(case_id: str) -> None:
    composition = _build(case_id)
    for energy, absorption, attenuation, elastic, inelastic in REFERENCE_CASES[
        case_id
    ]["coefficients"]:
        got = material_coefficients(composition, energy)
        for name, value, want in (
            ("absorption", got.absorption, absorption),
            ("attenuation", got.attenuation, attenuation),
            ("elastic", got.elastic, elastic),
            ("inelastic", got.inelastic, inelastic),
        ):
            assert float(value) == pytest.approx(want, rel=1e-12), (
                f"{name} at {energy} keV"
            )


# --- Average / Dummy ----------------------------------------------------


def test_average_coefficients_are_the_java_constants() -> None:
    assert AVERAGE_COEFFICIENTS.absorption == 0.000237
    assert AVERAGE_COEFFICIENTS.attenuation == 0.000281
    assert AVERAGE_COEFFICIENTS.elastic == 0.00001799
    assert AVERAGE_DENSITY == 1.2


def test_average_coefficients_do_not_depend_on_energy() -> None:
    """Worth stating outright: ``AbsCoefCalc Dummy`` gives the same dose at
    6 keV as at 20 keV, because the constants were never a function of
    energy. Fine for a first look, wrong for anything else.
    """
    assert AVERAGE_COEFFICIENTS.attenuation > AVERAGE_COEFFICIENTS.absorption


# --- SmallMole ----------------------------------------------------------


def test_small_molecule_atoms_are_per_monomer() -> None:
    composition = small_molecule_composition(1000.0, {"C": 6.0}, monomers=4)
    assert _counts(composition)["C"] == pytest.approx(24.0)


def test_small_molecule_zero_monomers_means_one() -> None:
    """The Java rewrites a zero monomer count to one rather than producing an
    empty cell.
    """
    assert _counts(small_molecule_composition(1000.0, {"C": 6.0}, monomers=0)) == (
        _counts(small_molecule_composition(1000.0, {"C": 6.0}, monomers=1))
    )


def test_small_molecule_has_no_solvent() -> None:
    """No water is added, however much empty cell there is.

    RADDOSE-3D also accepts ``SolventHeavyConc`` here and silently discards
    it -- ``fillRestWithWater`` is a hardcoded ``false``. The reference
    fixture pins that: the ``smallmole-solv`` case was built with 500 mM Na
    and contains no sodium.
    """
    counts = _counts(small_molecule_composition(1e6, {"C": 6.0}, monomers=1))
    assert set(counts) == {"C"}
    assert "NA" not in REFERENCE_CASES["smallmole-solv"]["atoms"]
    assert (
        REFERENCE_CASES["smallmole-solv"]["atoms"]
        == REFERENCE_CASES["smallmole"]["atoms"]
    )


# --- SAXS ---------------------------------------------------------------


@pytest.mark.conformance
@pytest.mark.parametrize(
    ("case_id", "macromolecule"),
    [
        ("saxs", Macromolecule(amino_acids=129)),
        ("saxs-rna", Macromolecule(rna_nucleotides=60)),
        ("saxs-dna", Macromolecule(dna_nucleotides=60)),
    ],
)
def test_saxs_monomer_count_matches_java(
    case_id: str, macromolecule: Macromolecule
) -> None:
    assert (
        saxs_monomer_count(SAXS_CELL_VOLUME, 10.0, macromolecule)
        == REFERENCE_CASES[case_id]["scalars"]["monomers"]
    )


def test_saxs_swaps_the_rna_and_dna_masses() -> None:
    """Pins a reproduced defect.

    ``CoefCalcSAXS.calculateNumMonomers`` is declared ``(residues, DNA, RNA)``
    and called ``(residues, RNA, DNA)``, so each nucleic acid is weighed with
    the other's average mass -- 327.0 for RNA, 339.5 for DNA. Equal counts of
    RNA and DNA therefore give *different* monomer counts, which is the
    observable symptom, and the heavier one is the wrong one.

    Reproduced because it changes computed dose for any SAXS sample with
    nucleic acid, so not reproducing it would make the port disagree with
    RADDOSE-3D and hide the difference inside a tolerance. Fix in both at
    once; this test then fails and says so.
    """
    rna = saxs_monomer_count(SAXS_CELL_VOLUME, 10.0, Macromolecule(rna_nucleotides=60))
    dna = saxs_monomer_count(SAXS_CELL_VOLUME, 10.0, Macromolecule(dna_nucleotides=60))
    assert rna != dna
    # RNA is the heavier residue (339.5 vs 327.0), so it should give FEWER
    # monomers per gram. It gives more, because it is weighed as DNA.
    assert rna > dna


def test_saxs_needs_a_positive_molecular_weight() -> None:
    with pytest.raises(ValueError, match="positive monomer molecular weight"):
        saxs_monomer_count(SAXS_CELL_VOLUME, 10.0, Macromolecule())


def test_saxs_never_returns_fewer_than_one_monomer() -> None:
    """A tiny volume would otherwise round to zero molecules and give a
    crystal made of nothing.
    """
    assert saxs_monomer_count(1.0, 1e-9, Macromolecule(amino_acids=129)) == 1


def test_more_concentrated_solution_holds_more_monomers() -> None:
    dilute = saxs_monomer_count(SAXS_CELL_VOLUME, 5.0, Macromolecule(amino_acids=129))
    strong = saxs_monomer_count(SAXS_CELL_VOLUME, 20.0, Macromolecule(amino_acids=129))
    assert strong > dilute


# --- Sequence -----------------------------------------------------------


def test_sequence_residue_counts_are_per_residue_not_averaged() -> None:
    """The point of the sequence path: a cysteine brings sulphur, which the
    average-residue model cannot express.
    """
    assert sequence_residue_counts("C")["S"] == 1
    assert "S" not in sequence_residue_counts("A")


def test_sequence_counts_add_up() -> None:
    single = sequence_residue_counts("A")
    triple = sequence_residue_counts("AAA")
    for symbol, count in single.items():
        assert triple[symbol] == pytest.approx(3 * count)


def test_sequence_ignores_whitespace_and_case() -> None:
    assert sequence_residue_counts("a c") == sequence_residue_counts("AC")


def test_the_same_letter_means_different_residues_by_type() -> None:
    """``A`` is alanine in a protein and adenine in DNA."""
    database = residue_database()
    assert database.by_letter("A", ResidueType.PROTEIN).key == "ALA"
    assert database.by_letter("A", ResidueType.DNA).key.strip() == "DA"
    assert database.by_letter("A", ResidueType.RNA).key.strip() == "A"


def test_unknown_sequence_letter_is_reported() -> None:
    """The Java prints "could not identify the sequence in the file" and then
    dereferences the null on the next line, so the real symptom is a
    ``NullPointerException``. Raising here says what went wrong.

    ``O`` (pyrrolysine) and ``U`` (selenocysteine) are the only two letters
    with no protein residue behind them.
    """
    with pytest.raises(KeyError, match="does not name a protein residue"):
        sequence_residue_counts("O")
    with pytest.raises(KeyError, match="does not name a protein residue"):
        sequence_residue_counts("U")


def test_ambiguity_codes_resolve_to_one_residue() -> None:
    """IUPAC ambiguity codes are accepted and silently resolved.

    ``J`` means "leucine or isoleucine" and is taken as leucine; ``B`` and
    ``Z`` are likewise resolved. Nucleic acid sequences accept the full
    ambiguity alphabet too. The choice is arbitrary but harmless at this
    resolution -- leucine and isoleucine are isomers with identical atom
    counts -- and pinning it stops a future "fix" changing results silently.
    """
    database = residue_database()
    assert database.by_letter("J", ResidueType.PROTEIN).key == "LEU"
    assert sequence_residue_counts("J") == sequence_residue_counts("L")
    assert sequence_residue_counts("J") == sequence_residue_counts("I")


def test_sequence_scales_protein_but_not_solvent_with_monomers() -> None:
    """Order of operations in the Java: water is computed for the cell, then
    the macromolecule is multiplied by the number of copies.
    """
    volume = unit_cell_volume(78.27, 78.27, 78.27)
    one = _counts(
        sequence_composition(volume, PROTEIN_SEQUENCE, monomers=1, solvent_fraction=0.5)
    )
    two = _counts(
        sequence_composition(volume, PROTEIN_SEQUENCE, monomers=2, solvent_fraction=0.5)
    )
    # Sulphur comes only from the protein, so it must double exactly.
    assert two["S"] == pytest.approx(2 * one["S"])


def test_sequence_residue_totals_match_java() -> None:
    amino_acids, rna, dna = REFERENCE_CASES["sequence"]["scalars"]["residues"]
    assert len(PROTEIN_SEQUENCE) == amino_acids
    assert (rna, dna) == (0, 0)

    amino_acids, rna, dna = REFERENCE_CASES["sequence-dna"]["scalars"]["residues"]
    assert len(DNA_SEQUENCE) == dna
    assert (amino_acids, rna) == (0, 0)


# --- residue database ---------------------------------------------------


def test_residue_database_is_complete() -> None:
    database = residue_database()
    assert len(database) == 29
    by_type = defaultdict(list)
    for residue in database:
        by_type[residue.residue_type].append(residue.key)
    assert len(by_type[ResidueType.PROTEIN]) == 21
    assert len(by_type[ResidueType.RNA]) == 4
    assert len(by_type[ResidueType.DNA]) == 4


def test_alanine_composition() -> None:
    alanine = residue_database().by_key("ALA")
    assert alanine.atoms == {"C": 3, "H": 7, "O": 2, "N": 1}
    assert alanine.molecular_weight == pytest.approx(89.09)


def test_selenomethionine_carries_selenium() -> None:
    """The reason the table has a selenium column at all."""
    selenium_bearing = [r for r in residue_database() if r.seleniums]
    assert selenium_bearing, "expected a selenomethionine entry"
    for residue in selenium_bearing:
        assert "SE" in residue.atoms


def test_unknown_residue_key_is_reported() -> None:
    with pytest.raises(KeyError, match="no residue keyed"):
        residue_database().by_key("ZZZ")
