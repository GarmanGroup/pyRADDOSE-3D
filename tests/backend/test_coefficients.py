"""Tests for absorption, attenuation and scattering coefficients.

The conformance tests take the atom counts from the Java reference as *input*
and check that the same coefficients come out. Building those counts from
residue counts and a solvent fraction is a later module; keeping the seam here
means a failure points at the arithmetic rather than at the composition model.
"""

import numpy as np
import pytest
from _cases import CASE_IDS, CASES, Case

from pyraddose_3d.backend.coefficients import (
    density,
    material_coefficients,
    unit_cell_volume,
)
from pyraddose_3d.backend.composition import Composition
from pyraddose_3d.backend.elements import element_database


def _composition(case: Case) -> Composition:
    """The reference's own atom counts, taken as given.

    Deliberately not built with `unit_cell_composition`: that is tested in
    test_composition.py, and keeping it out here means a failure below is
    about the coefficient arithmetic and nothing else.
    """
    db = element_database()
    return Composition(
        elements=tuple(db.by_symbol(s) for s in case.symbols),
        counts=np.array(case.counts),
        cell_volume=case.cell_volume,
    )


def test_reference_is_complete() -> None:
    """Guards against a truncated fixture quietly making the suite trivial."""
    assert len(CASES) == 6
    for case in CASES.values():
        assert case.symbols, f"{case.id} has no atoms"
        assert len(case.coefficients) == 6, f"{case.id} has too few energies"


# --- conformance against the Java ---------------------------------------


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_cell_volume_matches_java(case_id: str) -> None:
    case = CASES[case_id]
    got = unit_cell_volume(*case.cell)
    assert float(got) == pytest.approx(case.cell_volume, rel=1e-12)


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_density_matches_java(case_id: str) -> None:
    case = CASES[case_id]
    got = density(_composition(case))
    assert float(got) == pytest.approx(case.density, rel=1e-12)


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_coefficients_match_java(case_id: str) -> None:
    """Asserted at 1e-12: a weighted sum over at most nine elements has no
    real scope to diverge, so anything looser would be measuring nothing.
    """
    case = CASES[case_id]
    composition = _composition(case)
    for energy, absorption, attenuation, elastic, inelastic in case.coefficients:
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


# --- unit cell volume ---------------------------------------------------


def test_cubic_cell_is_the_cube_of_its_edge() -> None:
    assert float(unit_cell_volume(10.0, 10.0, 10.0)) == pytest.approx(1000.0)


def test_angles_default_to_ninety() -> None:
    assert float(unit_cell_volume(3.0, 4.0, 5.0)) == pytest.approx(
        float(unit_cell_volume(3.0, 4.0, 5.0, 90.0, 90.0, 90.0))
    )


def test_oblique_cell_is_smaller_than_orthorhombic() -> None:
    """Shearing a cell can only lose volume."""
    orthorhombic = float(unit_cell_volume(30.0, 40.0, 50.0))
    oblique = float(unit_cell_volume(30.0, 40.0, 50.0, 80.0, 95.0, 110.0))
    assert oblique < orthorhombic


def test_impossible_cell_angles_raise() -> None:
    """The deliberate divergence from the Java, which warns and returns NaN.

    Note that three *equal* angles can never fail: the expression under the
    root factors as ``(1 - c)^2 (1 + 2c)``, which is non-negative for any
    real angle. It takes a genuinely inconsistent set, such as two narrow
    angles and one obtuse one.
    """
    with pytest.raises(ValueError, match="realisable unit cell"):
        unit_cell_volume(10.0, 10.0, 10.0, 20.0, 20.0, 150.0)


def test_impossible_cell_does_not_silently_produce_nan() -> None:
    """The point of the divergence: the Java propagates NaN into every
    downstream quantity, so the first sign of a mistyped angle is a NaN dose.
    """
    try:
        volume = float(unit_cell_volume(10.0, 10.0, 10.0, 20.0, 20.0, 150.0))
    except ValueError:
        return
    pytest.fail(f"expected ValueError, got volume {volume}")


# --- coefficients -------------------------------------------------------


def test_empty_composition_has_no_coefficients() -> None:
    empty = Composition(elements=(), counts=np.array([]), cell_volume=1000.0)
    got = material_coefficients(empty, 12.1)
    assert float(got.attenuation) == 0.0


def test_attenuation_is_the_sum_of_the_channels() -> None:
    """Attenuation comes from the elements' own TOTAL, summed independently
    of the three channels, so agreement is a real cross-check rather than a
    restatement.
    """
    case = CASES["baseline"]
    got = material_coefficients(_composition(case), 12.1)
    assert float(got.attenuation) == pytest.approx(
        float(got.absorption) + float(got.elastic) + float(got.inelastic), rel=1e-12
    )


def test_doubling_the_cell_contents_doubles_the_coefficients() -> None:
    """Linear in atom count at fixed volume."""
    case = CASES["baseline"]
    single = _composition(case)
    double = Composition(single.elements, single.counts * 2, single.cell_volume)
    assert float(material_coefficients(double, 12.1).absorption) == pytest.approx(
        2 * float(material_coefficients(single, 12.1).absorption), rel=1e-12
    )


def test_coefficients_are_vectorised_over_energy() -> None:
    case = CASES["baseline"]
    composition = _composition(case)
    energies = np.array([e for e, *_ in case.coefficients])
    expected = np.array([a for _, a, *_ in case.coefficients])
    got = material_coefficients(composition, energies).absorption
    assert got.shape == energies.shape
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_heavy_atoms_increase_absorption() -> None:
    """Adding iron to a protein must absorb more, not less -- the reason
    anyone runs this calculation.
    """
    plain = float(
        material_coefficients(_composition(CASES["baseline"]), 12.1).absorption
    )
    with_iron = float(
        material_coefficients(_composition(CASES["heavy-protein"]), 12.1).absorption
    )
    assert with_iron > plain


# --- differentiability --------------------------------------------------


def test_gradient_flows_to_energy_and_cell() -> None:
    torch = pytest.importorskip("torch")
    case = CASES["baseline"]
    db = element_database()
    elements = tuple(db.by_symbol(s) for s in case.symbols)

    edge = torch.tensor(78.27, dtype=torch.float64, requires_grad=True)
    energy = torch.tensor(12.1, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor(case.counts, dtype=torch.float64, requires_grad=True)

    volume = unit_cell_volume(edge, edge, edge)
    coefficients = material_coefficients(Composition(elements, counts, volume), energy)
    coefficients.absorption.backward()

    for name, tensor in (("edge", edge), ("energy", energy), ("counts", counts)):
        assert tensor.grad is not None, f"no gradient reached {name}"
        assert torch.all(torch.isfinite(tensor.grad)), f"non-finite gradient at {name}"

    # A bigger cell dilutes the same atoms, and higher energy penetrates more.
    assert edge.grad.item() < 0
    assert energy.grad.item() < 0


def test_gradient_is_numerically_correct() -> None:
    torch = pytest.importorskip("torch")
    case = CASES["baseline"]
    db = element_database()
    elements = tuple(db.by_symbol(s) for s in case.symbols)
    counts = torch.tensor(case.counts, dtype=torch.float64)

    def absorption(energy: object, volume: object) -> object:
        return material_coefficients(
            Composition(elements, counts, volume), energy
        ).absorption

    args = (
        torch.tensor(12.1, dtype=torch.float64, requires_grad=True),
        torch.tensor(479497.1, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(absorption, args, eps=1e-6, atol=1e-9)


def test_unit_cell_volume_gradient_is_correct() -> None:
    torch = pytest.importorskip("torch")
    args = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for v in (30.0, 40.0, 50.0, 80.0, 95.0, 110.0)
    )
    assert torch.autograd.gradcheck(unit_cell_volume, args, eps=1e-6, atol=1e-8)


def test_both_backends_agree() -> None:
    torch = pytest.importorskip("torch")
    case = CASES["triclinic"]
    db = element_database()
    elements = tuple(db.by_symbol(s) for s in case.symbols)

    numpy_result = material_coefficients(
        Composition(elements, np.array(case.counts), case.cell_volume), 12.1
    ).attenuation
    torch_result = material_coefficients(
        Composition(
            elements,
            torch.tensor(case.counts, dtype=torch.float64),
            torch.tensor(case.cell_volume, dtype=torch.float64),
        ),
        torch.tensor(12.1, dtype=torch.float64),
    ).attenuation
    assert float(numpy_result) == pytest.approx(float(torch_result), rel=1e-14)
