"""Building the contents of a unit cell from a description of the crystal.

Ports ``CoefCalcFromParams.calculateAtomOccurrences`` and the solvent
machinery it leans on in ``CoefCalcCompute``: turn "eight copies of a
129-residue protein in a 78.27 A cube" into a count of each element per unit
cell, which :mod:`pyraddose_3d.backend.coefficients` can then turn into
absorption coefficients.

The model is coarse by design. A residue is not looked up by identity; every
amino acid contributes the same average 5 C, 1.35 N, 1.5 O and 8 H. Solvent
is water at a fixed concentration, less whatever the specified solute
concentrations displace.

Units
-----
Cell volume in cubic Angstroms, solute concentrations in millimoles per litre,
heavy atoms per monomer, counts in atoms per unit cell.

Differentiability
-----------------
Everything here is differentiable in cell volume, solvent fraction and the
heavy-atom and solute amounts. Residue counts are integers and so are not
differentiable, which is no loss: they are not something a strategy
optimiser can vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pyraddose_3d.backend._array import array_namespace, promote
from pyraddose_3d.backend.elements import ATOMIC_MASS_UNIT, element_database

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyraddose_3d.backend.elements import Element

__all__ = [
    "Composition",
    "Macromolecule",
    "estimate_solvent_fraction",
    "unit_cell_composition",
]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""

# --- constants, transcribed from CoefCalcCompute.java -------------------
AVOGADRO = 6.022e23
"""Avogadro's number, as spelled in the Java (not the CODATA value)."""

#: Cubic Angstroms to millilitres.
_ANGSTROMS_TO_ML = 1e-24
_MASS_TO_CELL_VOLUME = 1e27
_UNITS_PER_MILLI = 1000.0

#: Molar concentration of water. The Java's comment says "55.555M instead of
#: the 55M in Fortran", but the active value is 51666 -- amorphous ice at
#: 0.93 g/cm^3 rather than liquid water at 1.0. Changed in RADDOSE-3D at
#: b0a79ff (October 2023), which moved absorption coefficients ~2.6%.
WATER_CONCENTRATION = 51666.0

#: Average residue masses in unified atomic mass units, and the densities
#: used to convert those masses into a fraction of the cell.
_AMINO_ACID_AVERAGE_MASS = 110.0
_PROTEIN_DENSITY = 1.35
_RNA_NUCLEOTIDE_MASS = 321.0
_RNA_DENSITY = 1.3
_DNA_NUCLEOTIDE_MASS = 312.0
_DNA_DENSITY = 1.35
_CARBOHYDRATE_AVERAGE_MASS = 162.0
_CARBOHYDRATE_DENSITY = 1.54
_HETATM_DENSITY = 1.35

#: Heteroatoms at or above this atomic number are left out of the solvent
#: estimate, so that a few heavy atoms cannot dominate it. The Java carries a
#: "TODO: Is this < or <= ?!" against the comparison; it is ``<``.
LIGHT_ATOM_MAX_NUMBER = 29

#: Atoms contributed by one residue of each kind. Averages over composition,
#: not the formula of any particular residue.
_PER_AMINO_ACID = {"C": 5.0, "N": 1.35, "O": 1.5, "H": 8.0}
_PER_RNA_NUCLEOTIDE = {"C": 9.5, "N": 3.75, "O": 7.0, "H": 11.25, "P": 1.0}
_PER_DNA_NUCLEOTIDE = {"C": 9.75, "N": 4.0, "O": 6.0, "H": 11.75, "P": 1.0}
_PER_CARBOHYDRATE = {"C": 6.0, "O": 5.0, "H": 11.0}


@dataclass(frozen=True)
class Macromolecule:
    """Residue counts for what the unit cell contains.

    Attributes
    ----------
    monomers : int
        Copies of the macromolecule per unit cell. Every other count here is
        per monomer and is multiplied by this.
    amino_acids : int
        Protein residues per monomer.
    rna_nucleotides, dna_nucleotides : int
        Nucleic acid residues per monomer.
    carbohydrate_residues : int
        Sugar residues per monomer.
    """

    monomers: int = 1
    amino_acids: int = 0
    rna_nucleotides: int = 0
    dna_nucleotides: int = 0
    carbohydrate_residues: int = 0


@dataclass(frozen=True)
class Composition:
    """What is in one unit cell.

    Attributes
    ----------
    elements : tuple[Element, ...]
        The elements present, ordered by atomic number.
    counts : Array
        Atoms of each element per unit cell, aligned with ``elements``.
        Fractional counts are normal: they are averages over the cell.
    cell_volume : Array
        Unit cell volume in cubic Angstroms.
    solvent_fraction : Array | None
        The fraction of the cell taken to be solvent, when known. ``None``
        for a composition assembled without one.
    """

    elements: tuple[Element, ...]
    counts: Array
    cell_volume: Array
    solvent_fraction: Array | None = field(default=None)


def estimate_solvent_fraction(
    cell_volume: Array,
    macromolecule: Macromolecule,
    hetero_atoms: Mapping[str, Array] | None = None,
) -> Array:
    """Estimate what fraction of the unit cell is solvent.

    Works by subtraction: convert each kind of residue to a mass, divide by
    an assumed density to get the volume it occupies, and call whatever is
    left over solvent.

    Parameters
    ----------
    cell_volume : Array
        Unit cell volume in cubic Angstroms.
    macromolecule : Macromolecule
        Residue counts.
    hetero_atoms : Mapping[str, Array], optional
        Non-solvent, non-polymer atoms per unit cell, keyed by chemical
        symbol. Only ever non-empty for compositions read from a PDB file.
        Atoms with atomic number at or above ``LIGHT_ATOM_MAX_NUMBER`` are
        ignored, so that a handful of heavy atoms cannot dominate.

    Returns
    -------
    Array
        Solvent fraction. Not clamped: a value below zero means the residue
        counts do not fit in the cell, and is worth seeing rather than
        hiding.

    Notes
    -----
    ``proteinDensity`` is a constant 1.35 here, matching the Java. Directly
    beneath it in ``CoefCalcCompute.java`` sits a commented-out
    implementation of the Fischer et al. (2004) molecular-weight-dependent
    formula, under a comment saying to use Fischer. The two differ by around
    11%. Reproduced as-is; changing it is a decision for both codebases at
    once.
    """
    database = element_database()
    monomers = macromolecule.monomers

    def occupied(mass: float, count: int, density: float) -> Array:
        total = ATOMIC_MASS_UNIT * mass * count * monomers
        return total / (cell_volume * density * _ANGSTROMS_TO_ML)

    protein = occupied(
        _AMINO_ACID_AVERAGE_MASS, macromolecule.amino_acids, _PROTEIN_DENSITY
    )
    rna = occupied(
        _RNA_NUCLEOTIDE_MASS, macromolecule.rna_nucleotides, _RNA_DENSITY
    )
    dna = occupied(
        _DNA_NUCLEOTIDE_MASS, macromolecule.dna_nucleotides, _DNA_DENSITY
    )
    carbohydrate = occupied(
        _CARBOHYDRATE_AVERAGE_MASS,
        macromolecule.carbohydrate_residues,
        _CARBOHYDRATE_DENSITY,
    )

    hetatm_mass: Array = 0.0
    for symbol, count in (hetero_atoms or {}).items():
        element = database.by_symbol(symbol)
        if element.atomic_number < LIGHT_ATOM_MAX_NUMBER:
            hetatm_mass = hetatm_mass + count * element.atomic_weight_in_grams
    hetatm = hetatm_mass / (cell_volume * _HETATM_DENSITY * _ANGSTROMS_TO_ML)

    return 1 - protein - rna - dna - hetatm - carbohydrate


def unit_cell_composition(
    cell_volume: Array,
    macromolecule: Macromolecule,
    *,
    heavy_protein_atoms: Mapping[str, Array] | None = None,
    solvent_concentrations: Mapping[str, Array] | None = None,
    solvent_fraction: Array | None = None,
    hetero_atoms: Mapping[str, Array] | None = None,
) -> Composition:
    """Assemble the contents of a unit cell.

    Parameters
    ----------
    cell_volume : Array
        Unit cell volume in cubic Angstroms.
    macromolecule : Macromolecule
        Residue counts.
    heavy_protein_atoms : Mapping[str, Array], optional
        Extra atoms bound to the macromolecule, keyed by chemical symbol,
        given **per monomer** and multiplied up accordingly.
    solvent_concentrations : Mapping[str, Array], optional
        Solute concentrations in millimoles per litre. These displace water
        rather than adding to it.
    solvent_fraction : Array, optional
        Fraction of the cell that is solvent. Estimated with
        :func:`estimate_solvent_fraction` when not given.
    hetero_atoms : Mapping[str, Array], optional
        Atoms per unit cell that are neither solvent nor polymer. Passed on
        to the solvent estimate and added to the cell contents.

    Returns
    -------
    Composition
        Elements and their counts per unit cell, ordered by atomic number.
    """
    database = element_database()
    monomers = macromolecule.monomers

    if solvent_fraction is None:
        solvent_fraction = estimate_solvent_fraction(
            cell_volume, macromolecule, hetero_atoms
        )

    # Counts are accumulated against chemical symbols rather than Element
    # objects: Element carries a dict of coefficients and so is not hashable.
    # Keys are the database's own spelling, which is upper case for
    # two-letter symbols ("FE", not "Fe"), so callers may write either.
    counts: dict[str, Array] = {}

    def add(symbol: str, amount: Array) -> None:
        name = database.by_symbol(symbol).name
        counts[name] = counts.get(name, 0.0) + amount

    for symbol, per_monomer in (heavy_protein_atoms or {}).items():
        add(symbol, per_monomer * monomers)

    for symbol, amount in (hetero_atoms or {}).items():
        add(symbol, amount)

    # Solutes first: they displace water, so the water count depends on them.
    non_water_atoms: Array = 0.0
    for symbol, concentration in (solvent_concentrations or {}).items():
        atom_count = (
            concentration * AVOGADRO * cell_volume * solvent_fraction * 1e-3 * 1e-27
        )
        add(symbol, atom_count)
        non_water_atoms = non_water_atoms + atom_count

    xp = array_namespace(cell_volume, solvent_fraction, non_water_atoms)
    water_molecules = (
        WATER_CONCENTRATION
        * AVOGADRO
        / _UNITS_PER_MILLI
        * cell_volume
        * (1 / _MASS_TO_CELL_VOLUME)
        * solvent_fraction
        - non_water_atoms
    )
    # More solute than the solvent can hold. The Java clamps rather than
    # letting a negative water count subtract atoms from the cell.
    water_molecules = xp.where(water_molecules < 0, 0.0, water_molecules)
    add("H", water_molecules * 2)
    add("O", water_molecules)

    residues = (
        (_PER_AMINO_ACID, macromolecule.amino_acids),
        (_PER_RNA_NUCLEOTIDE, macromolecule.rna_nucleotides),
        (_PER_DNA_NUCLEOTIDE, macromolecule.dna_nucleotides),
        (_PER_CARBOHYDRATE, macromolecule.carbohydrate_residues),
    )
    for per_residue, residue_count in residues:
        for symbol, per_one in per_residue.items():
            add(symbol, per_one * residue_count * monomers)

    elements = sorted(
        (database.by_symbol(s) for s in counts), key=lambda e: e.atomic_number
    )
    # Counts are a mix: residue contributions are plain floats, while
    # anything derived from the solvent carries whatever type cell_volume
    # had. promote() brings them together before stacking.
    xp, values = promote(*(counts[e.name] for e in elements))
    return Composition(
        elements=tuple(elements),
        counts=xp.stack(values),
        cell_volume=cell_volume,
        solvent_fraction=solvent_fraction,
    )
