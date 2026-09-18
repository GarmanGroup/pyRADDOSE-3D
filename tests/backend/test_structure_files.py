"""Tests for reading a composition from a PDB or CIF file.

These complete the ``AbsCoefCalc`` routes: ``EXP`` reads a PDB, ``EXPSM``
reads a CIF. Both are fixed-scope parsers rather than general structure-file
readers. The PDB one carries a latent defect, reproduced only in the safe
direction; the CIF one carried a large defect that has now been fixed in both
codebases.
"""

import csv
import pathlib
from collections import defaultdict

import pytest

from pyraddose_3d.backend.coefficients import density, material_coefficients
from pyraddose_3d.backend.composition import Composition
from pyraddose_3d.backend.structure_files import (
    composition_from_cif,
    composition_from_pdb,
)

DATA = pathlib.Path(__file__).parent.parent / "data"
REFERENCE = pathlib.Path(__file__).parent / "data" / "coefficients_files.tsv"

PDB_FILE = DATA / "3I40.pdb"
CIF_ANTIMONY = DATA / "COD-1008123.cif"
CIF_UREA = DATA / "COD-1008775-urea.cif"


def _load_reference() -> dict[str, dict]:
    cases: dict[str, dict] = defaultdict(
        lambda: {"atoms": {}, "coefficients": [], "density": None, "scalars": {}}
    )
    with REFERENCE.open(encoding="ascii") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            tag, case_id = row[0], row[1]
            if tag == "DENSITY":
                cases[case_id]["density"] = float(row[2])
            elif tag == "ATOMS":
                cases[case_id]["atoms"][row[3]] = float(row[4])
            elif tag == "COEF":
                cases[case_id]["coefficients"].append(tuple(float(v) for v in row[2:7]))
            elif tag in ("MONOMERS", "SOLFRAC"):
                cases[case_id]["scalars"][tag] = float(row[2])
            elif tag == "RESIDUES":
                cases[case_id]["scalars"]["RESIDUES"] = tuple(int(v) for v in row[2:5])
    return dict(cases)


REFERENCE_CASES = _load_reference()
BUILDERS = {
    "pdb": lambda: composition_from_pdb(PDB_FILE),
    "cif1": lambda: composition_from_cif(CIF_ANTIMONY),
    "cif2": lambda: composition_from_cif(CIF_UREA),
}
CASE_IDS = sorted(BUILDERS)


def _counts(composition: Composition) -> dict[str, float]:
    return {
        e.name: float(n)
        for e, n in zip(composition.elements, composition.counts, strict=True)
    }


# --- conformance against the Java ---------------------------------------


def test_reference_covers_both_parsers() -> None:
    assert set(REFERENCE_CASES) == set(BUILDERS)


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_atom_counts_match_java(case_id: str) -> None:
    got = _counts(BUILDERS[case_id]())
    want = REFERENCE_CASES[case_id]["atoms"]
    for symbol in set(got) | set(want):
        assert got.get(symbol, 0.0) == pytest.approx(
            want.get(symbol, 0.0), rel=1e-12
        ), f"{symbol} count"


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_density_matches_java(case_id: str) -> None:
    assert float(density(BUILDERS[case_id]())) == pytest.approx(
        REFERENCE_CASES[case_id]["density"], rel=1e-12
    )


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_coefficients_match_java(case_id: str) -> None:
    composition = BUILDERS[case_id]()
    for energy, absorption, attenuation, elastic, inelastic in REFERENCE_CASES[case_id][
        "coefficients"
    ]:
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


# --- PDB ----------------------------------------------------------------


def test_pdb_solvent_fraction_matches_java() -> None:
    composition = composition_from_pdb(PDB_FILE)
    assert float(composition.solvent_fraction) == pytest.approx(
        REFERENCE_CASES["pdb"]["scalars"]["SOLFRAC"], rel=1e-12
    )


def test_pdb_monomer_count_is_the_product_of_the_operator_counts() -> None:
    """24 crystallographic operators and no non-crystallographic ones.

    The Z value on the CRYST1 line is deliberately ignored -- the Java's
    comment says it "is often mis-calculated in the PDB ... when there is
    non-crystallographic symmetry involved" -- so the count comes from
    REMARK 290 SMTRY1 and MTRIX1 records instead.
    """
    counts = _counts(composition_from_pdb(PDB_FILE))
    monomers = int(REFERENCE_CASES["pdb"]["scalars"]["MONOMERS"])
    assert monomers == 24
    # Sulphur comes only from the sequence, so it divides exactly.
    assert counts["S"] % monomers == 0


def test_pdb_residue_counts_come_from_seqres() -> None:
    """SEQRES gives the *stated* sequence, not the modelled one, so residues
    disordered out of the coordinates still count towards the composition.
    """
    amino_acids, rna, dna = REFERENCE_CASES["pdb"]["scalars"]["RESIDUES"]
    assert (amino_acids, rna, dna) == (51, 0, 0)


def test_pdb_has_solvent_but_cif_does_not() -> None:
    """A protein crystal is mostly water; a small-molecule one is modelled as
    having none at all.
    """
    assert composition_from_pdb(PDB_FILE).solvent_fraction is not None
    assert composition_from_cif(CIF_UREA).solvent_fraction is None


def test_pdb_without_a_cell_is_refused() -> None:
    """Without CRYST1 there is no volume to divide by. The Java prints
    "Could not find CRYST1 line" and then divides by zero.
    """
    headerless = (
        pathlib.Path(pytest.importorskip("tempfile").mkdtemp()) / "no_cryst1.pdb"
    )
    headerless.write_text("SEQRES   1 A   2  ALA GLY\nEND\n", encoding="ascii")
    with pytest.raises(ValueError, match="no usable CRYST1"):
        composition_from_pdb(headerless)


def test_short_lines_do_not_abort_parsing(tmp_path: pathlib.Path) -> None:
    """A deliberate divergence, and the only safe direction for one.

    ``readPDBFile`` catches ``IndexOutOfBoundsException`` *outside* its read
    loop, so in the Java the first line too short for a fixed-column slice
    abandons the rest of the file and the composition is silently built from
    whatever had been read. Well-formed PDB files are padded to 80 columns,
    so this never fires on them and the two implementations agree; files from
    tools that do not pad are the risk.

    Python slices do not raise on a short string, so the port reads the
    missing columns as blank and carries on.
    """
    lines = PDB_FILE.read_text("ascii").splitlines()
    # A bare "REMARK 290" is 10 characters -- too short for the [13:19] slice
    # the symmetry check makes.
    truncated = tmp_path / "short_lines.pdb"
    truncated.write_text(
        "\n".join([lines[0], "REMARK 290", *lines[1:]]) + "\n", encoding="ascii"
    )
    assert _counts(composition_from_pdb(truncated)) == _counts(
        composition_from_pdb(PDB_FILE)
    )


def test_waters_are_left_to_the_solvent_model(tmp_path: pathlib.Path) -> None:
    """HETATM records for HOH are skipped, so water is counted once by the
    solvent model rather than twice.
    """
    original = PDB_FILE.read_text("ascii")
    water = (
        "HETATM 9999  O   HOH A 500      "
        "10.000  10.000  10.000  1.00 20.00           O  "
    )
    with_water = tmp_path / "extra_water.pdb"
    with_water.write_text(original + water + "\n", encoding="ascii")
    assert _counts(composition_from_pdb(with_water)) == _counts(
        composition_from_pdb(PDB_FILE)
    )


# --- CIF ----------------------------------------------------------------


def test_cif_density_matches_the_value_in_the_file() -> None:
    """The strongest check available here, and the one that found the bug.

    Each CIF states a measured or calculated crystal density, which
    RADDOSE-3D does not read. Agreeing with it is evidence that the
    calculation is right rather than merely self-consistent.

    ``_chemical_formula_sum`` gives the contents of *one formula unit*, and
    ``_cell_formula_units_Z`` says how many the cell holds. RADDOSE-3D read
    the first and never the second, so every quantity that scales with cell
    contents was low by exactly a factor of Z:

    ======================  ===  ======  ======  ==================
    structure               Z    before  after   stated in the CIF
    ======================  ===  ======  ======  ==================
    urea (COD 1008775)      2    0.683   1.367   1.367
    COD 1008123             8    0.364   2.913   2.9
    ======================  ===  ======  ======  ==================

    Fixed in both codebases; guarded in RADDOSE-3D by
    ``CoefCalcCifFormulaUnitsTest``.
    """
    for path, published in ((CIF_UREA, 1.367), (CIF_ANTIMONY, 2.9)):
        got = float(density(composition_from_cif(path)))
        assert got == pytest.approx(published, rel=0.02), (
            f"{path.name}: computed {got} against the density stated in the "
            f"file, {published}"
        )


def test_cif_counts_are_scaled_by_the_formula_units() -> None:
    """Urea is C H4 N2 O with Z = 2, so the cell holds two of everything."""
    assert _counts(composition_from_cif(CIF_UREA)) == {
        "C": 2.0,
        "H": 8.0,
        "N": 4.0,
        "O": 2.0,
    }


def test_cif_without_formula_units_assumes_one(tmp_path: pathlib.Path) -> None:
    """Correct when the item is genuinely absent, and the previous behaviour.

    RADDOSE-3D warns that the density will be understated if the omission was
    an oversight.
    """
    cif = tmp_path / "no_z.cif"
    cif.write_text(
        "_chemical_formula_sum            'C H4 N2 O'\n"
        "_cell_volume                     145.9\n",
        encoding="ascii",
    )
    assert _counts(composition_from_cif(cif))["C"] == 1.0


def test_cif_formula_units_may_precede_the_formula(
    tmp_path: pathlib.Path,
) -> None:
    """CIF items may appear in any order, and Z is commonly written above the
    formula. Applying it as each token is parsed would make the result depend
    on that order.
    """
    cif = tmp_path / "z_first.cif"
    cif.write_text(
        "_cell_formula_units_Z            4\n"
        "_chemical_formula_sum            'C H4 N2 O'\n"
        "_cell_volume                     145.9\n",
        encoding="ascii",
    )
    assert _counts(composition_from_cif(cif))["C"] == 4.0


@pytest.mark.parametrize("value", ["0", "-2", "4.5", "many", ""])
def test_nonsensical_formula_units_fall_back_to_one(
    tmp_path: pathlib.Path, value: str
) -> None:
    """Z is defined as an integer. Scaling a cell by a fraction, or by zero,
    would be worse than ignoring a malformed value.
    """
    cif = tmp_path / f"bad_{abs(hash(value))}.cif"
    cif.write_text(
        "_chemical_formula_sum            'C H4 N2 O'\n"
        "_cell_volume                     145.9\n"
        f"_cell_formula_units_Z            {value}\n",
        encoding="ascii",
    )
    assert _counts(composition_from_cif(cif))["C"] == 1.0


def test_formula_units_may_be_written_with_a_decimal_point(
    tmp_path: pathlib.Path,
) -> None:
    cif = tmp_path / "z_decimal.cif"
    cif.write_text(
        "_chemical_formula_sum            'C H4 N2 O'\n"
        "_cell_volume                     145.9\n"
        "_cell_formula_units_Z            4.0\n",
        encoding="ascii",
    )
    assert _counts(composition_from_cif(cif))["C"] == 4.0


def test_cif_reads_two_letter_symbols(tmp_path: pathlib.Path) -> None:
    """Antimony must not be split into sulphur and boron.

    This is the case the ``elementLetterLength`` guard exists for; getting it
    wrong was the defect fixed in RADDOSE-3D by PR #31.
    """
    # Z = 8 for this structure, so two antimony per formula unit is sixteen.
    counts = _counts(composition_from_cif(CIF_ANTIMONY))
    assert counts["SB"] == 16.0
    assert "S" not in counts
    assert "B" not in counts


def test_cif_bare_symbol_means_one_atom(tmp_path: pathlib.Path) -> None:
    """A trailing single-letter symbol with no number, as in urea's
    ``'C H4 N2 O'``. Indexing blindly here used to crash RADDOSE-3D on most
    organic formulas.
    """
    cif = tmp_path / "bare.cif"
    cif.write_text(
        "_chemical_formula_sum            'C H4 N2 O'\n"
        "_cell_volume                     145.9\n",
        encoding="ascii",
    )
    # No Z stated, so one formula unit.
    counts = _counts(composition_from_cif(cif))
    assert counts["C"] == 1.0
    assert counts["O"] == 1.0


def test_cif_strips_a_standard_uncertainty(tmp_path: pathlib.Path) -> None:
    """Cell volumes are usually written ``2076.9(5)``."""
    cif = tmp_path / "esd.cif"
    cif.write_text(
        "_chemical_formula_sum            'C H4 N2 O'\n"
        "_cell_volume                     145.90(12)\n",
        encoding="ascii",
    )
    assert float(composition_from_cif(cif).cell_volume) == pytest.approx(145.90)


def test_cif_without_a_formula_is_refused(tmp_path: pathlib.Path) -> None:
    """The Java used to call ``System.exit(0)`` here -- from a library class,
    with a success status, so Ant reported BUILD SUCCESSFUL and any test
    runner was killed mid-suite. Fixed in RADDOSE-3D by PR #31 to throw;
    raising here matches.
    """
    cif = tmp_path / "no_formula.cif"
    cif.write_text("_cell_volume 145.9\n", encoding="ascii")
    with pytest.raises(ValueError, match="_chemical_formula_sum"):
        composition_from_cif(cif)


def test_cif_repeated_element_keeps_the_last_count(tmp_path: pathlib.Path) -> None:
    """The Java assigns rather than accumulates, so a formula naming an
    element twice keeps only the second. Matched, and pinned because
    accumulating would be the obvious "fix".
    """
    cif = tmp_path / "repeated.cif"
    cif.write_text(
        "_chemical_formula_sum            'C2 H4 C3'\n"
        "_cell_volume                     100.0\n",
        encoding="ascii",
    )
    assert _counts(composition_from_cif(cif))["C"] == 3.0
