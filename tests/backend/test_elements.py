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
                worst, worst_at = rel, f"{name} for {row['name']} at {row['energy']} keV"

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

    Every row of MuCalcConstants.txt has non-zero leading coherent and
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
