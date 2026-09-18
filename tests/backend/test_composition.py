"""Tests for building unit cell contents from a description of the crystal.

This is the module where RADDOSE-3D's physical assumptions live -- average
residue compositions, assumed densities, the water concentration -- so most
of these tests pin a number that someone might one day want to change, and
say what changing it would mean.
"""

import numpy as np
import pytest
from _cases import CASE_IDS, CASES, Case

from pyraddose_3d.backend.coefficients import density, material_coefficients
from pyraddose_3d.backend.composition import (
    WATER_CONCENTRATION,
    Macromolecule,
    estimate_solvent_fraction,
    unit_cell_composition,
)


def _counts(composition: object) -> dict[str, float]:
    """Atom counts keyed by the database's symbol spelling."""
    return {
        e.name: float(n)
        for e, n in zip(composition.elements, composition.counts, strict=False)
    }


def _build(case: Case) -> object:
    return unit_cell_composition(
        case.cell_volume,
        Macromolecule(
            monomers=case.monomers,
            amino_acids=case.amino_acids,
            rna_nucleotides=case.rna_nucleotides,
            dna_nucleotides=case.dna_nucleotides,
        ),
        heavy_protein_atoms=case.heavy_protein_atoms or None,
        solvent_concentrations=case.solvent_concentrations or None,
        solvent_fraction=case.given_solvent_fraction,
    )


# --- conformance against the Java ---------------------------------------


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_atom_counts_match_java(case_id: str) -> None:
    """The whole builder, end to end: residue counts in, atoms per cell out."""
    case = CASES[case_id]
    composition = _build(case)

    got = {
        e.name: float(n)
        for e, n in zip(composition.elements, composition.counts, strict=False)
    }
    want = dict(zip(case.symbols, case.counts, strict=False))

    # The Java records an element once it appears in any occurrence map, even
    # at zero (phosphorus in a protein-only cell). Compare on the union.
    for symbol in set(got) | set(want):
        assert got.get(symbol, 0.0) == pytest.approx(
            want.get(symbol, 0.0), rel=1e-12
        ), f"{symbol} count"


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_density_matches_java_from_a_built_composition(case_id: str) -> None:
    """Chains the builder into the density calculation, so a mistake in
    either shows up even if the two happen to cancel elsewhere.
    """
    case = CASES[case_id]
    assert float(density(_build(case))) == pytest.approx(case.density, rel=1e-12)


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_coefficients_match_java_from_a_built_composition(case_id: str) -> None:
    case = CASES[case_id]
    composition = _build(case)
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


@pytest.mark.conformance
@pytest.mark.parametrize(
    "case_id", [i for i in CASE_IDS if CASES[i].estimates_solvent_fraction]
)
def test_estimated_solvent_fraction_matches_java(case_id: str) -> None:
    case = CASES[case_id]
    got = estimate_solvent_fraction(
        case.cell_volume,
        Macromolecule(
            monomers=case.monomers,
            amino_acids=case.amino_acids,
            rna_nucleotides=case.rna_nucleotides,
            dna_nucleotides=case.dna_nucleotides,
        ),
    )
    assert float(got) == pytest.approx(case.solvent_fraction, rel=1e-12)


def test_an_explicit_solvent_fraction_is_reported_back() -> None:
    """A deliberate divergence, because the Java has a defect here.

    ``CoefCalcCompute.solFraction`` is assigned only inside
    ``calculateSolventFractionFromNums``, which is skipped when the input
    file supplies a ``SolventFraction``. ``getSolventFraction()`` then
    returns the uninitialised 0.0 rather than the value given -- visible in
    the reference fixture, where the ``solvent-explicit`` case records
    ``SOLFRAC 0.0`` after being handed 0.5.

    The consequence is contained: the only caller is ``MicroED.java:4533``,
    so the X-ray path is unaffected and the atom counts are right either way,
    because ``calculateSolventWater`` is passed the correct value directly.
    Not reproduced -- returning zero for a number the user supplied would be
    indefensible in new code.
    """
    case = CASES["solvent-explicit"]
    assert case.given_solvent_fraction == 0.5
    assert case.solvent_fraction == 0.0, "the Java defect this documents"

    composition = _build(case)
    assert float(composition.solvent_fraction) == 0.5


# --- the physical assumptions -------------------------------------------


def test_water_concentration_is_amorphous_ice() -> None:
    """51666 M is 0.93 g/cm^3, not liquid water's 55555 M at 1.0 g/cm^3.

    Changed in RADDOSE-3D at b0a79ff (October 2023) and confirmed as
    intended. It moves absorption coefficients by ~2.6%, so it is worth a
    test that fails loudly rather than a constant someone tidies.
    """
    assert WATER_CONCENTRATION == 51666.0


def test_empty_cell_is_all_solvent() -> None:
    assert float(estimate_solvent_fraction(100000.0, Macromolecule())) == 1.0


def test_more_protein_leaves_less_solvent() -> None:
    small = float(estimate_solvent_fraction(479497.0, Macromolecule(8, 129)))
    large = float(estimate_solvent_fraction(479497.0, Macromolecule(8, 400)))
    assert large < small


def test_overfull_cell_gives_a_negative_solvent_fraction() -> None:
    """Not clamped, deliberately: a negative value means the residue counts
    do not fit the cell, which the user needs to see. The Java prints a
    warning and carries on with the negative number too.
    """
    assert float(estimate_solvent_fraction(10000.0, Macromolecule(8, 500))) < 0


def test_solvent_is_water_in_the_right_proportion() -> None:
    """Two hydrogens per oxygen, for a cell with no other source of either."""
    composition = unit_cell_composition(100000.0, Macromolecule(), solvent_fraction=1.0)
    counts = {
        e.name: float(n)
        for e, n in zip(composition.elements, composition.counts, strict=False)
    }
    assert counts["H"] == pytest.approx(2 * counts["O"])


def test_solutes_displace_water_rather_than_adding_to_it() -> None:
    plain = unit_cell_composition(100000.0, Macromolecule(), solvent_fraction=1.0)
    salty = unit_cell_composition(
        100000.0,
        Macromolecule(),
        solvent_concentrations={"Na": 500.0},
        solvent_fraction=1.0,
    )
    water_plain = dict(
        zip([e.name for e in plain.elements], plain.counts, strict=False)
    )["O"]
    water_salty = dict(
        zip([e.name for e in salty.elements], salty.counts, strict=False)
    )["O"]
    assert float(water_salty) < float(water_plain)


def test_water_count_is_clamped_at_zero() -> None:
    """An absurd solute concentration must not make water negative, which
    would subtract atoms from the cell.
    """
    composition = unit_cell_composition(
        100000.0,
        Macromolecule(),
        solvent_concentrations={"Na": 1e9},
        solvent_fraction=1.0,
    )
    counts = {
        e.name: float(n)
        for e, n in zip(composition.elements, composition.counts, strict=False)
    }
    assert counts["O"] == 0.0


def test_heavy_atoms_are_given_per_monomer() -> None:
    """A plausible unit error: the input file means "per monomer", so eight
    monomers with one iron each give eight irons in the cell.
    """
    composition = unit_cell_composition(
        479497.0,
        Macromolecule(monomers=8, amino_acids=129),
        heavy_protein_atoms={"Fe": 1.0},
    )
    # The database spells two-letter symbols in upper case, so "Fe" in goes
    # in as written and comes back out as "FE".
    counts = {
        e.name: float(n)
        for e, n in zip(composition.elements, composition.counts, strict=False)
    }
    assert counts["FE"] == pytest.approx(8.0)


def test_residue_atoms_scale_with_monomers() -> None:
    one = unit_cell_composition(479497.0, Macromolecule(1, 129), solvent_fraction=0.0)
    two = unit_cell_composition(479497.0, Macromolecule(2, 129), solvent_fraction=0.0)
    carbon_one = dict(zip([e.name for e in one.elements], one.counts, strict=False))[
        "C"
    ]
    carbon_two = dict(zip([e.name for e in two.elements], two.counts, strict=False))[
        "C"
    ]
    assert float(carbon_two) == pytest.approx(2 * float(carbon_one))


def test_nucleic_acids_bring_phosphorus() -> None:
    """One phosphate per nucleotide, and none from protein."""
    protein = unit_cell_composition(479497.0, Macromolecule(1, 129))
    rna = unit_cell_composition(479497.0, Macromolecule(1, 0, rna_nucleotides=40))
    protein_p = dict(
        zip([e.name for e in protein.elements], protein.counts, strict=False)
    ).get("P")
    rna_p = dict(zip([e.name for e in rna.elements], rna.counts, strict=False))["P"]
    assert protein_p is None or float(protein_p) == 0.0
    assert float(rna_p) == pytest.approx(40.0)


def test_elements_are_ordered_by_atomic_number() -> None:
    """The Java iterates a HashSet, so its order is arbitrary. Fixing an
    order here keeps outputs and summation reproducible.
    """
    composition = unit_cell_composition(
        479497.0, Macromolecule(8, 129), heavy_protein_atoms={"Fe": 1.0, "S": 2.0}
    )
    numbers = [e.atomic_number for e in composition.elements]
    assert numbers == sorted(numbers)


# --- differentiability --------------------------------------------------


def test_gradient_flows_through_the_builder() -> None:
    torch = pytest.importorskip("torch")
    volume = torch.tensor(479497.0, dtype=torch.float64, requires_grad=True)
    composition = unit_cell_composition(volume, Macromolecule(8, 129))
    material_coefficients(composition, 12.1).absorption.backward()
    assert volume.grad is not None
    assert torch.isfinite(volume.grad)


def test_solvent_fraction_gradient_is_correct() -> None:
    torch = pytest.importorskip("torch")
    volume = torch.tensor(479497.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v: estimate_solvent_fraction(v, Macromolecule(8, 129)),
        (volume,),
        eps=1e-2,
        atol=1e-9,
    )


def test_both_backends_build_the_same_cell() -> None:
    torch = pytest.importorskip("torch")
    case = CASES["heavy-solvent"]
    numpy_counts = _build(case).counts
    torch_composition = unit_cell_composition(
        torch.tensor(case.cell_volume, dtype=torch.float64),
        Macromolecule(monomers=case.monomers, amino_acids=case.amino_acids),
        solvent_concentrations={
            k: torch.tensor(v, dtype=torch.float64)
            for k, v in case.solvent_concentrations.items()
        },
    )
    np.testing.assert_allclose(
        np.asarray(numpy_counts, dtype=float),
        torch_composition.counts.numpy(),
        rtol=1e-12,
    )
