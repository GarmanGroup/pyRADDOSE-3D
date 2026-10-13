"""Tests for the element cross-section database.

Three independent kinds of check, deliberately:

1. **External reference values** (:func:`test_sulphur_matches_published`,
   :func:`test_magnesium_matches_published`) come from csrri.iit.edu, not
   from RADDOSE-3D. They are the only evidence here that the *Java* is right,
   rather than that the port agrees with it.
2. **A sweep against the Java** over every element and 20 energies. This is
   the conformance check, and it is tight: 1e-9, against a measured worst
   case of 3.7e-15.
3. **Properties the Java never had** -- vectorisation, gradients, and the
   piecewise structure at absorption edges.
4. **Integrity of the vendored tables**, which replaced RADDOSE-3D's single
   87-by-80 unlabelled file. The sweep in (2) is what proves the split did
   not corrupt a value; these check the things the old format could not
   express, such as a row being the wrong length.
"""

import csv
import gzip
import math
import pathlib

import numpy as np
import pytest

from pyraddose_3d.backend.elements import (
    ATOMIC_MASS_UNIT,
    Element,
    ElementDatabase,
    element_database,
    read_table,
)

REFERENCE = pathlib.Path(__file__).parent / "data" / "element_cross_sections.tsv.gz"

#: Barns/Atom = CONVERSION * cm^2/g, where CONVERSION is the atomic weight
#: times the atomic mass unit scaled into barns.
_SULPHUR_C = 53.2400017
_MAGNESIUM_C = 40.3800011


@pytest.fixture(scope="module")
def db() -> ElementDatabase:
    return element_database()


# --- structure ----------------------------------------------------------


def test_database_covers_the_expected_elements(db: ElementDatabase) -> None:
    assert len(db) == 87
    assert db.by_number(1).name == "H"
    assert db.by_symbol("s").atomic_number == 16


def test_lookup_is_case_insensitive(db: ElementDatabase) -> None:
    assert db.by_symbol("FE") is db.by_symbol("fe") is db.by_symbol("Fe")


def test_database_is_shared(db: ElementDatabase) -> None:
    """The Java is a singleton; parsing per call would be waste, not a bug,
    but callers rely on elements being identical objects.
    """
    assert element_database() is db


@pytest.mark.parametrize("missing", [0, 93, -1])
def test_unknown_atomic_number_names_the_range(
    db: ElementDatabase, missing: int
) -> None:
    with pytest.raises(KeyError, match="Z = 1 to"):
        db.by_number(missing)


def test_unknown_symbol_is_reported(db: ElementDatabase) -> None:
    with pytest.raises(KeyError, match="Unobtainium"):
        db.by_symbol("Unobtainium")


def test_sulphur_fields(db: ElementDatabase) -> None:
    s = db.by_symbol("S")
    assert s.atomic_number == 16
    assert s.atomic_weight == pytest.approx(3.20660e01)
    assert s.atomic_weight_in_grams == pytest.approx(5.3246841e-23, rel=1e-7)
    assert s.edge_k == pytest.approx(2.472)
    assert s.coefficients["K"] == pytest.approx(
        (1.37394e01, -2.04786e00, -2.73259e-01, 2.29976e-02)
    )
    # M and N have no parameterisation for sulphur and become zero.
    assert s.coefficients["M"] == (0.0, 0.0, 0.0, 0.0)


def test_atomic_mass_unit_is_the_java_value() -> None:
    """Pinned because it is spelled as a literal in Element.java and is a
    plausible thing to silently 'correct' to CODATA 1.66053907e-24.
    """
    assert ATOMIC_MASS_UNIT == 1.66e-24


def test_light_elements_have_no_m_edge(db: ElementDatabase) -> None:
    """15 rows leave the M edge blank; the M and N branches are then
    unreachable and everything below K uses the L polynomial.
    """
    without_m = [e.name for e in db if e.edge_m is None]
    assert len(without_m) == 15
    assert without_m[0] == "H"


# --- external reference values ------------------------------------------


def _compare_published(
    element: Element,
    conversion: float,
    energy: float,
    photoelectric: float,
    coherent: float,
    total: float,
) -> None:
    """Compare against a published cm^2/g value, as ElementTest.java does."""
    xs = element.absorption_coefficients(energy)
    for name, got, want in (
        ("photoelectric", xs.photoelectric, photoelectric),
        ("coherent", xs.coherent, coherent),
        ("total", xs.total, total),
    ):
        assert float(got) / conversion == pytest.approx(want, abs=0.001), (
            f"{name} at {energy} keV"
        )


def test_sulphur_matches_published(db: ElementDatabase) -> None:
    """Values from http://csrri.iit.edu/cgi-bin/period-form?ener=1.1&name=S.

    The two points either side of the 2.472 keV K edge are the interesting
    ones: the cross-section jumps tenfold across 0.0002 keV.
    """
    s = db.by_symbol("S")
    c = _SULPHUR_C
    _compare_published(s, c, 1.1, 1978.71497, 2.62428641, 1981.35059)
    _compare_published(s, c, 1.2, 1568.09387, 2.63965917, 1570.74646)
    _compare_published(s, c, 1.7, 609.046936, 2.56492448, 611.633484)
    _compare_published(s, c, 2.1, 339.237946, 2.42386913, 341.690186)
    _compare_published(s, c, 2.3, 262.987366, 2.34494638, 265.363983)
    _compare_published(s, c, 2.4, 233.32811, 2.30482531, 235.666229)
    _compare_published(s, c, 2.4719, 214.701355, 2.27591038, 217.011719)
    _compare_published(s, c, 2.4721, 2217.97024, 2.27582908, 2220.2803)
    _compare_published(s, c, 2.5, 2156.88818, 2.26461291, 2159.18774)
    _compare_published(s, c, 5, 349.58316, 1.44702506, 351.098572)
    _compare_published(s, c, 10, 48.4815712, 0.716733396, 49.3056908)


def test_magnesium_matches_published(db: ElementDatabase) -> None:
    """Values from http://csrri.iit.edu/cgi-bin/period-form?ener=1.1&name=Mg.

    ElementTest.java carries these too, under the name ``manganeseTests``
    (they are magnesium), but nothing ever calls that method -- it is dead
    test code in the Java. Revived here, and they pass.
    """
    mg = db.by_symbol("Mg")
    c = _MAGNESIUM_C
    _compare_published(mg, c, 1.1, 753.605774, 2.02505422, 755.646667)
    _compare_published(mg, c, 1.2, 588.960938, 2.03198099, 591.010681)
    _compare_published(mg, c, 1.30, 470.938721, 2.02889442, 472.987274)
    _compare_published(mg, c, 1.31, 5354.29818, 2.0281291, 5356.34603)
    _compare_published(mg, c, 1.7, 2900.64331, 1.95550072, 2902.62573)
    _compare_published(mg, c, 2.1, 1719.7324, 1.83737242, 1721.60361)
    _compare_published(mg, c, 2.3, 1364.00793, 1.77321875, 1365.81836)
    _compare_published(mg, c, 2.5, 1099.26038, 1.70869017, 1101.0094)
    _compare_published(mg, c, 5, 163.331894, 1.07319105, 164.477081)
    _compare_published(mg, c, 10, 20.6076527, 0.52411586, 21.2394829)
    _compare_published(mg, c, 0.05, 41337066.185, 0.0305607822, 41337066.215)
    _compare_published(mg, c, 0.1, 2718520.463, 0.169477433, 2718520.633)


# --- conformance against the Java ---------------------------------------


def _reference_rows() -> list[dict[str, str]]:
    with gzip.open(REFERENCE, "rt", encoding="ascii") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@pytest.mark.conformance
def test_matches_java_across_every_element_and_energy(db: ElementDatabase) -> None:
    """The port specification: 1720 element/energy points, four quantities.

    Asserted at 1e-9 rather than the 1e-5 used for whole-experiment
    conformance. This module is a closed-form polynomial with no summation
    order to differ, so it should agree to machine precision; a tolerance
    that merely met the project-wide target would hide a real divergence.
    """
    rows = _reference_rows()
    assert len(rows) == 1720, "reference fixture is truncated"

    worst = 0.0
    worst_at = ""
    for row in rows:
        xs = db.by_number(int(row["z"])).absorption_coefficients(float(row["energy"]))
        for name, got in (
            ("photoelectric", xs.photoelectric),
            ("coherent", xs.coherent),
            ("compton", xs.compton),
            ("total", xs.total),
        ):
            want = float(row[name])
            value = float(got)
            if want == value:
                continue
            rel = abs(value - want) / abs(want) if want else abs(value)
            if rel > worst:
                worst, worst_at = (
                    rel,
                    f"{name} for {row['name']} at {row['energy']} keV",
                )

    assert worst < 1e-9, f"worst relative deviation {worst:.3e} at {worst_at}"


@pytest.mark.conformance
def test_vectorised_evaluation_matches_scalar(db: ElementDatabase) -> None:
    """Whatever the Java did one energy at a time, the port must do in a
    batch and get the same answer. This is what makes the exposure loop
    affordable, so it is worth an explicit check rather than an assumption.
    """
    rows = _reference_rows()
    by_element: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        by_element.setdefault(int(row["z"]), []).append(
            (float(row["energy"]), float(row["total"]))
        )

    for z, points in by_element.items():
        energies = np.array([e for e, _ in points])
        expected = np.array([t for _, t in points])
        got = db.by_number(z).absorption_coefficients(energies).total
        assert got.shape == energies.shape
        np.testing.assert_allclose(got, expected, rtol=1e-9)


# --- properties the Java did not have -----------------------------------


def test_energy_equal_to_one_is_discontinuous(db: ElementDatabase) -> None:
    """A deliberate wart, carried over: at exactly 1 keV the Java sums the
    coefficients instead of evaluating the polynomial. Pinned so that
    removing it is a visible decision rather than an accident, and so the
    port is not "corrected" into disagreeing with the Java.
    """
    s = db.by_symbol("S")
    at_one = float(s.absorption_coefficients(1.0).photoelectric)
    just_below = float(s.absorption_coefficients(1.0 - 1e-12).photoelectric)
    just_above = float(s.absorption_coefficients(1.0 + 1e-12).photoelectric)

    assert just_below == pytest.approx(just_above, rel=1e-9)
    assert at_one != pytest.approx(just_below, rel=1e-6)
    # 1 keV is below sulphur's 2.472 keV K edge, so the L polynomial is the
    # one selected, and the value is the sum of its coefficients exponentiated.
    assert at_one == pytest.approx(math.exp(sum(s.coefficients["L"])))


def test_cross_section_jumps_at_the_k_edge(db: ElementDatabase) -> None:
    """The physics the branch selection exists to reproduce."""
    s = db.by_symbol("S")
    below = float(s.absorption_coefficients(s.edge_k - 1e-4).photoelectric)
    above = float(s.absorption_coefficients(s.edge_k + 1e-4).photoelectric)
    assert above / below > 9


def test_photoelectric_falls_with_energy_within_a_shell(db: ElementDatabase) -> None:
    """Monotone between edges -- a cheap check that no branch is mis-selected
    in the middle of a shell, which a tolerance-based comparison could miss
    if the reference were ever regenerated from a broken build.
    """
    s = db.by_symbol("S")
    energies = np.linspace(s.edge_k + 0.01, 30.0, 200)
    photoelectric = s.absorption_coefficients(energies).photoelectric
    assert np.all(np.diff(photoelectric) < 0)


def test_no_shipped_element_lacks_scattering_coefficients(
    db: ElementDatabase,
) -> None:
    """Records that the Java's two ``COEFF_0 != 0`` guards are dead code.

    Every element has non-zero leading coherent and
    incoherent coefficients, so neither guard can fire. Kept in the port
    anyway -- see the next test for why it must stay -- but pinned here so
    that if a future database revision does introduce a blank column, this
    fails and says so rather than the guard quietly coming alive.
    """
    for key in ("coherent", "incoherent"):
        assert not [e.name for e in db if e.coefficients[key][0] == 0]


def test_absent_scattering_polynomial_contributes_zero() -> None:
    """An absent polynomial must mean no scattering, not exp(0) = 1 barn.

    Uses a synthetic element because no shipped one exercises this, and the
    consequence of dropping the guard would be a silent 1 barn per atom
    rather than a crash.
    """
    element = Element(
        name="Xx",
        atomic_number=999,
        atomic_weight=1.0,
        edge_k=2.0,
        edge_l=0.1,
        edge_m=None,
        l2=0.09,
        l3=0.08,
        coefficients={
            "K": (1.0, 0.0, 0.0, 0.0),
            "L": (1.0, 0.0, 0.0, 0.0),
            "M": (0.0, 0.0, 0.0, 0.0),
            "N": (0.0, 0.0, 0.0, 0.0),
            "coherent": (0.0, 0.0, 0.0, 0.0),
            "incoherent": (0.0, 0.0, 0.0, 0.0),
        },
    )
    xs = element.absorption_coefficients(12.1)
    assert float(xs.coherent) == 0.0
    assert float(xs.compton) == 0.0
    assert float(xs.total) == float(xs.photoelectric)


# --- differentiability --------------------------------------------------


def test_gradient_flows_through_to_energy(db: ElementDatabase) -> None:
    """Dose must be differentiable in beam energy; this is where that starts."""
    torch = pytest.importorskip("torch")
    s = db.by_symbol("S")
    energy = torch.tensor(12.1, dtype=torch.float64, requires_grad=True)
    s.absorption_coefficients(energy).total.backward()
    assert energy.grad is not None
    assert torch.isfinite(energy.grad)
    # Above the K edge, absorption falls as energy rises.
    assert energy.grad.item() < 0


def test_gradient_is_numerically_correct(db: ElementDatabase) -> None:
    torch = pytest.importorskip("torch")
    s = db.by_symbol("S")
    # Well inside a shell: the derivative at an edge is genuinely undefined.
    energies = torch.tensor(
        [3.0, 6.0, 12.1, 20.0], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda e: s.absorption_coefficients(e).total, (energies,), eps=1e-7, atol=1e-6
    )


def test_both_backends_agree(db: ElementDatabase) -> None:
    torch = pytest.importorskip("torch")
    s = db.by_symbol("Se")
    energies = [0.5, 1.0, 3.0, 12.1, 25.0]
    got_np = s.absorption_coefficients(np.array(energies)).total
    got_torch = s.absorption_coefficients(
        torch.tensor(energies, dtype=torch.float64)
    ).total
    np.testing.assert_allclose(got_np, got_torch.numpy(), rtol=1e-14)


# --- integrity of the vendored tables -----------------------------------
#
# The seven files replaced RADDOSE-3D's constants/MuCalcConstants.txt, which
# was 87 rows of 80 tab-separated columns with no header. Its column names
# lived in a Java enum in a different class, six columns were named nowhere
# at all, and nothing checked that a row was the right length -- gadolinium's
# had 82 fields.

#: Every table, and whether the port reads it. The two it does not are the
#: rest of the original 80 columns, waiting for the Monte Carlo subprogram.
ALL_TABLES = {
    "elements.tsv": True,
    "absorption_edges.tsv": True,
    "cross_section_coefficients.tsv": True,
    "fluorescence.tsv": True,
    "edge_ratios.tsv": True,
    "electron_ionisation.tsv": False,
    "elsepa_elastic.tsv": False,
}


@pytest.mark.parametrize("table", sorted(ALL_TABLES))
def test_every_table_is_well_formed(table: str) -> None:
    """Reading it at all is the check: ``read_table`` raises on a bad row.

    Including the two nothing reads yet. Data that no test touches is data
    that can rot silently, and these carry the electron-impact ionisation and
    elastic cross-sections the Monte Carlo subprogram would need.
    """
    rows = read_table(table)
    assert rows, f"{table} has no data rows"
    assert all(row["z"] and row["symbol"] for row in rows)


@pytest.mark.parametrize("table", sorted(ALL_TABLES))
def test_every_table_covers_the_same_elements(table: str, db: ElementDatabase) -> None:
    """The split must not have lost or duplicated an element anywhere.

    Each table repeats the atomic number and symbol so it can be read on its
    own; that redundancy is only worth having if it is checked.
    """
    expected = {(e.atomic_number, e.name) for e in db}
    found = [(int(row["z"]), row["symbol"]) for row in read_table(table)]

    assert set(found) == expected
    if table != "cross_section_coefficients.tsv":
        assert len(found) == len(expected), f"{table} repeats an element"


def test_a_ragged_row_is_rejected(tmp_path: pathlib.Path, monkeypatch) -> None:
    """The guarantee the old format could not give.

    A row with the wrong number of fields used to be undetectable: with no
    header there was nothing to be wrong against, and a stray trailing tab
    shifted nothing because the columns it added were past the last one
    anyone read.
    """
    from importlib import resources

    bad = tmp_path / "ragged.tsv"
    bad.write_text("z\tsymbol\tvalue\n1\tH\t1.0\n2\tHE\t2.0\t\n", "ascii")
    monkeypatch.setattr(resources, "files", lambda _package: tmp_path, raising=True)

    with pytest.raises(ValueError, match="4 fields, but the header declares 3"):
        read_table("ragged.tsv")


def test_absent_shells_are_the_only_ones_missing_from_the_table() -> None:
    """A shell is either fully parameterised or absent; never half of one.

    The coefficients table is written long, one row per element per shell,
    and omits a shell the element has no parameterisation for rather than
    writing four blanks. That is only safe because no element has a
    partially blank polynomial -- 502 shells are complete and 20 are empty,
    with nothing in between. Pinned so a future revision that introduces a
    half-filled shell fails here rather than silently reading as zero.
    """
    rows = read_table("cross_section_coefficients.tsv")

    assert len(rows) == 502
    for row in rows:
        assert all(row[f"c{i}"].strip() for i in range(4)), row

    shells = {row["shell"] for row in rows}
    assert shells == {"K", "L", "M", "N", "coherent", "incoherent"}


def test_an_absent_shell_reads_as_four_zeros(db: ElementDatabase) -> None:
    """Which is what the Java's blank columns meant -- see quirk (3).

    Every element must offer all six polynomials whether or not the table
    lists them, because the cross-section code indexes them unconditionally.
    """
    listed = {
        (int(row["z"]), row["shell"])
        for row in read_table("cross_section_coefficients.tsv")
    }
    missing = [
        (e.name, shell)
        for e in db
        for shell in ("K", "L", "M", "N", "coherent", "incoherent")
        if (e.atomic_number, shell) not in listed
    ]

    assert missing, "expected some elements to lack a parameterised shell"
    for name, shell in missing:
        assert db.by_symbol(name).coefficients[shell] == (0.0, 0.0, 0.0, 0.0)


def test_a_blank_edge_is_absent_and_a_zero_edge_is_zero() -> None:
    """The distinction the old file drew with an empty column, preserved.

    A blank M edge means the element has none, and the photoelectric code
    branches on that. A zero L edge means the edge is tabulated at zero,
    which is a different thing, and there are elements with each.
    """
    rows = {row["symbol"]: row for row in read_table("absorption_edges.tsv")}

    assert rows["H"]["m1_kev"] == ""
    assert rows["H"]["l1_kev"] == "0.0"
    assert rows["FE"]["m1_kev"] == "0.094"

    db = element_database()
    assert db.by_symbol("H").edge_m is None
    assert db.by_symbol("H").edge_l == 0.0
    assert db.by_symbol("Fe").edge_m == 0.094


def test_published_edge_energies_are_right(db: ElementDatabase) -> None:
    """Spot-checks against the X-ray Data Booklet, not against RADDOSE-3D.

    Independent evidence that the tabulated K edges are the real ones,
    rather than merely that the port agrees with whatever it read last time.
    Loose, because the published values are themselves rounded; the exact
    ones are pinned by the test below.
    """
    published = {"C": 0.284, "S": 2.472, "FE": 7.112, "SE": 12.658, "AU": 80.725}
    for symbol, energy in published.items():
        assert db.by_symbol(symbol).edge_k == pytest.approx(energy, abs=5e-3), symbol


#: Exact edge energies in keV: K, L1, M1, L2, L3. Transcribed from the
#: database itself, so this pins the values rather than validating them --
#: :func:`test_published_edge_energies_are_right` is the external check.
EXACT_EDGES = {
    "H": (0.014, 0.0, None, 0.0, 0.0),
    "C": (0.284, 0.013925, None, 0.00899, 0.00898),
    "S": (2.472, 0.193, 0.017, 0.1636, 0.1625),
    "FE": (7.112, 0.842, 0.094, 0.7199, 0.7068),
    "SE": (12.658, 1.653, 0.232, 1.477, 1.436),
    "AU": (80.723, 14.353, 3.425, 13.733, 11.918),
    "U": (115.603, 21.756, 5.549, 20.947, 17.167),
}


@pytest.mark.parametrize("symbol", sorted(EXACT_EDGES))
def test_edge_energies_are_exactly_as_tabulated(
    symbol: str, db: ElementDatabase
) -> None:
    """Exact equality, because nothing else notices a small edge change.

    The sweep against the Java samples twenty energies across the MX range,
    so a K edge moved by a few eV changes no sampled cross-section at all --
    moving iron's from 7.112 to 7.113 leaves the whole suite green. The edge
    is not a smooth parameter, though: it is the boundary that decides which
    polynomial a photon gets, and a crystal probed at exactly that energy
    would take the wrong branch. So it is pinned to the bit.
    """
    element = db.by_symbol(symbol)
    edge_k, edge_l, edge_m, l2, l3 = EXACT_EDGES[symbol]

    assert element.edge_k == edge_k
    assert element.edge_l == edge_l
    assert element.edge_m == edge_m
    assert element.l2 == l2
    assert element.l3 == l3
