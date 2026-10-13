"""Reading a unit cell composition from a structure file or a CIF file.

Ports ``CoefCalcFromPDB`` and ``CoefCalcFromCIF``, the two ``AbsCoefCalc``
routes that read a file rather than taking parameters. RADDOSE-3D spells them
``EXP`` and ``EXPSM`` respectively -- not ``PDB`` and ``CIF``, which is a
naming accident in the grammar rather than a choice.

The files are read with Gemmi, as CCP4 asks of code it distributes, rather
than by hand as RADDOSE-3D reads them. What is taken from a file is what
RADDOSE-3D takes; how it is found differs, and on some files so does the
result.

Where results differ from RADDOSE-3D
------------------------------------
- **Symmetry operators are counted from the space group**, not from the
  ``SMTRY1`` records in ``REMARK 290``. The two agree on any file that has
  those records. A file without them gave RADDOSE-3D no operators, and so no
  macromolecule in the cell.
- **A file without a crystal cell is refused.** RADDOSE-3D refused a file
  with no ``CRYST1`` record, but accepted the placeholder 1 A cell of an NMR
  or EM model and, finding no symmetry operators, filled it with solvent.
- **A blank occupancy counts as unoccupied**, because Gemmi reads a blank as
  zero. RADDOSE-3D counted it as full. Well-formed files have no blanks.
- **An atom with blank element columns has its element inferred** from its
  name by Gemmi. RADDOSE-3D skipped it.
- **The whole file is read.** RADDOSE-3D stopped at the first line too short
  for its fixed columns. Well-formed files are padded to 80 columns.
- **mmCIF is read as well as PDB format.** RADDOSE-3D reads PDB format only.
- **CIF items are found wherever CIF allows them**: a formula in double
  quotes, on the line after its tag, or in a text field, and a tag in any
  case. RADDOSE-3D found ``_chemical_formula_sum`` only in single quotes on
  its tag's line, and otherwise refused the file. A file that is not valid
  CIF is refused, where RADDOSE-3D read whatever lines it recognised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gemmi
import numpy as np

from pyraddose_3d.backend._array import promote
from pyraddose_3d.backend.coefficients import unit_cell_volume
from pyraddose_3d.backend.composition import (
    Composition,
    Macromolecule,
    _add_solvent,
    estimate_solvent_fraction,
)
from pyraddose_3d.backend.elements import element_database
from pyraddose_3d.backend.residues import ResidueType, residue_database

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = ["composition_from_cif", "composition_from_pdb"]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""

#: The element Gemmi gives an atom whose element it cannot tell.
_UNKNOWN_ELEMENT = gemmi.Element("X")


@dataclass
class _StructureRecords:
    """What a structure file contributes to a composition, before scaling."""

    macromolecular: dict[str, float]
    hetero_atoms: dict[str, float]
    residues: Macromolecule
    crystallographic_operators: int
    non_crystallographic_operators: int
    cell_volume: Any

    @property
    def monomers(self) -> int:
        """Copies of the asymmetric unit in the cell."""
        return self.crystallographic_operators * self.non_crystallographic_operators


def _structure_records(
    structure: gemmi.Structure, path: str | Path
) -> _StructureRecords:
    """Collect what a composition needs from a structure Gemmi has read."""
    cell = structure.cell
    if not cell.is_crystal():
        raise ValueError(
            f"{path} has no crystal unit cell -- no CRYST1 record, or the "
            f"placeholder cell of an NMR or EM model -- so there is no volume "
            f"to fill and no composition can be built from it"
        )
    # Computed as RADDOSE-3D computes it, rather than taken from Gemmi, so the
    # two agree to the last bit.
    cell_volume = unit_cell_volume(
        cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma
    )

    space_group = structure.find_spacegroup()
    if space_group is None:
        raise ValueError(
            f"{path} gives the space group {structure.spacegroup_hm!r}, which "
            f"Gemmi does not recognise, so the number of molecules in the "
            f"cell is unknown"
        )
    # One copy per operator, centring translations included: the count of
    # SMTRY1 records RADDOSE-3D reads from REMARK 290, wherever a file has
    # them.
    crystallographic_operators = len(space_group.operations())
    # Gemmi leaves the identity out. An operator already applied to the
    # coordinates must not be counted again.
    non_crystallographic_operators = 1 + sum(
        1 for operator in structure.ncs if not operator.given
    )

    # The sequence is the *stated* one -- SEQRES, or an mmCIF file's entity
    # sequences -- not the modelled one, so disordered residues absent from
    # the coordinates still count.
    database = residue_database()
    residue_atoms: dict[str, float] = {}
    amino_acids = rna = dna = 0
    for entity in structure.entities:
        # A PDB file's SEQRES gives one entity per chain, and an mmCIF file
        # one per molecule, listing each copy as a subchain. A SEQRES chain
        # with no atoms has no subchain and still counts once.
        copies = len(entity.subchains) or 1
        for item in entity.full_sequence:
            # Residue keys are right-justified, as SEQRES writes them: "  A"
            # for adenosine, " DA" for deoxyadenosine.
            name = gemmi.Entity.first_mon(item).rjust(3)
            try:
                tabulated = database.by_key(name)
            except KeyError:
                # "could not decipher PDB three letter code".
                continue
            if tabulated.residue_type is ResidueType.PROTEIN:
                amino_acids += copies
            elif tabulated.residue_type is ResidueType.RNA:
                rna += copies
            else:
                dna += copies
            for symbol, number in tabulated.atoms.items():
                residue_atoms[symbol] = residue_atoms.get(symbol, 0.0) + number * copies

    hetero_atoms: dict[str, float] = {}
    # Every model, as RADDOSE-3D reads every HETATM record in the file.
    for model in structure:
        for chain in model:
            for residue in chain:
                # Water is solvent, and is counted by the solvent model rather
                # than atom by atom.
                if residue.het_flag != "H" or residue.name.upper() == "HOH":
                    continue
                for atom in residue:
                    if atom.element == _UNKNOWN_ELEMENT:
                        continue
                    symbol = atom.element.name.upper()
                    amount = _occupancy(atom)
                    hetero_atoms[symbol] = hetero_atoms.get(symbol, 0.0) + amount

    # Residue atoms and heteroatoms both belong to the macromolecule as far
    # as the occurrence maps are concerned; heteroatoms are additionally
    # tracked on their own, because the solvent estimate treats them
    # differently.
    macromolecular = dict(residue_atoms)
    for symbol, amount in hetero_atoms.items():
        macromolecular[symbol] = macromolecular.get(symbol, 0.0) + amount

    return _StructureRecords(
        macromolecular=macromolecular,
        hetero_atoms=hetero_atoms,
        residues=Macromolecule(
            amino_acids=amino_acids, rna_nucleotides=rna, dna_nucleotides=dna
        ),
        crystallographic_operators=crystallographic_operators,
        non_crystallographic_operators=non_crystallographic_operators,
        cell_volume=cell_volume,
    )


def _occupancy(atom: gemmi.Atom) -> float:
    """An atom's occupancy as written in its file.

    Gemmi holds occupancies in single precision, so 0.2 comes back as
    0.20000000298. The shortest decimal that rounds to the same
    single-precision number is what the file wrote, and reading that gives
    the double RADDOSE-3D reads.
    """
    return float(str(np.float32(atom.occ)))


def composition_from_pdb(
    path: str | Path,
    solvent_concentrations: Mapping[str, Array] | None = None,
) -> Composition:
    """Read a unit cell composition from a PDB or mmCIF file.

    Ports ``CoefCalcFromPDB``, minus its ability to download an entry from
    RCSB by four-letter code. Five things are read:

    the unit cell
        Its dimensions give the volume.
    the space group
        One copy of the asymmetric unit per operator.
    non-crystallographic operators
        One more copy per operator, unless the file marks it as already
        applied to the coordinates.
    the sequence
        From ``SEQRES``, or an mmCIF file's entity sequences. Residue
        compositions are looked up from it. This is the *stated* sequence,
        not the modelled one, so disordered residues still count.
    heteroatoms
        Everything in a ``HETATM`` record but water, weighted by occupancy.
        Water is left to the solvent model.

    The number of monomers in the cell is the product of the two operator
    counts. The Z value on the ``CRYST1`` line is deliberately ignored: the
    Java's comment says it "is often mis-calculated in the PDB ... when there
    is non-crystallographic symmetry involved".

    Parameters
    ----------
    path : str | Path
        Path to a PDB or mmCIF file.
    solvent_concentrations : Mapping[str, Array], optional
        Heavy-atom concentrations in the solvent, in millimolar, keyed by
        element symbol. Displaces water, as in
        :func:`pyraddose_3d.backend.composition.unit_cell_composition`.

    Returns
    -------
    Composition
        Elements and counts per unit cell, ordered by atomic number.

    Raises
    ------
    ValueError
        If the file has no crystal unit cell, since without a cell volume
        there is nothing to divide by, or a space group Gemmi does not
        recognise.

    Notes
    -----
    The module docstring lists where this differs from RADDOSE-3D.
    """
    structure = gemmi.read_structure(str(path), format=gemmi.CoorFormat.Detect)
    records = _structure_records(structure, path)

    monomers = records.monomers
    scaled = {s: n * monomers for s, n in records.macromolecular.items()}
    scaled_hetero = {s: n * monomers for s, n in records.hetero_atoms.items()}

    solvent_fraction = estimate_solvent_fraction(
        records.cell_volume,
        Macromolecule(
            monomers=monomers,
            amino_acids=records.residues.amino_acids,
            rna_nucleotides=records.residues.rna_nucleotides,
            dna_nucleotides=records.residues.dna_nucleotides,
        ),
        scaled_hetero,
    )
    return _assemble(
        records.cell_volume, scaled, solvent_fraction, solvent_concentrations
    )


_CIF_FORMULA_KEY = "_chemical_formula_sum"
_CIF_VOLUME_KEY = "_cell_volume"
_CIF_FORMULA_UNITS_KEY = "_cell_formula_units_Z"


def composition_from_cif(path: str | Path) -> Composition:
    """Read a unit cell composition from a small-molecule CIF file.

    Ports ``CoefCalcFromCIF``, which RADDOSE-3D selects for ``AbsCoefCalc
    EXPSM``. Three data items are read: ``_chemical_formula_sum`` for the
    contents of one formula unit, ``_cell_formula_units_Z`` for the number of
    formula units, and ``_cell_volume`` for the cell. There is no solvent
    model -- a small-molecule crystal is its formula and nothing else.

    Parameters
    ----------
    path : str | Path
        Path to a CIF file.

    Returns
    -------
    Composition
        Elements and counts per unit cell, ordered by atomic number.

    Raises
    ------
    ValueError
        If the file has no ``_chemical_formula_sum``.

    Notes
    -----
    ``_chemical_formula_sum`` gives the contents of one formula unit, so every
    count from it is multiplied by ``_cell_formula_units_Z``. RADDOSE-3D read
    the formula and never Z, filling the whole cell with a single formula unit
    and understating the density -- and every absorption coefficient and dose
    that follows from it -- by exactly that factor:

    ======================  ===  ======  ======  ====================
    structure               Z    before  after   stated in the CIF
    ======================  ===  ======  ======  ====================
    urea (COD 1008775)      2    0.683   1.367   1.367
    COD 1008123             8    0.364   2.913   2.9
    ======================  ===  ======  ======  ====================

    Found during this port and fixed in both, so the two still agree. A file
    that states no Z keeps the reading of one formula unit, which is correct
    when the item is genuinely absent. See RADDOSE-3D ``TEST-TRIAGE.md``
    item 10 and ``CoefCalcCifFormulaUnitsTest``.

    The module docstring lists where this differs from RADDOSE-3D.
    """
    document = gemmi.cif.read_file(str(path))
    formula: str | None = None
    cell_volume: Any = None
    formula_units = 1
    # RADDOSE-3D reads the whole file, and a later value replaces an earlier
    # one. Block by block, so does this.
    for block in document:
        value = block.find_value(_CIF_FORMULA_KEY)
        if value is not None and gemmi.cif.as_string(value):
            formula = gemmi.cif.as_string(value)
        value = block.find_value(_CIF_FORMULA_UNITS_KEY)
        if value is not None:
            formula_units = _parse_formula_units(gemmi.cif.as_string(value))
        value = block.find_value(_CIF_VOLUME_KEY)
        if value is not None:
            # A standard uncertainty in brackets is dropped; "?" and "." are
            # NaN, and leave the volume as it was.
            volume = gemmi.cif.as_number(value)
            if not math.isnan(volume):
                cell_volume = volume

    if formula is None:
        raise ValueError(
            f"{path} has no {_CIF_FORMULA_KEY}, which is the only thing "
            f"RADDOSE-3D reads a CIF for"
        )

    # Applied here rather than while parsing the formula: CIF items may
    # appear in any order, and Z is commonly written above the formula.
    counts = {
        symbol: amount * formula_units
        for symbol, amount in _parse_chemical_formula(formula).items()
    }
    return _assemble(cell_volume, counts, None, None)


def _parse_formula_units(value: str) -> int:
    """Parse ``_cell_formula_units_Z``, falling back to one formula unit.

    Parameters
    ----------
    value : str
        The item's value as written in the file.

    Returns
    -------
    int
        Formula units per cell. One if the value is not a positive whole
        number, which is safer than silently scaling a cell by a fraction.
    """
    try:
        parsed = float(value)
    except ValueError:
        return 1
    # Defined as an integer in the CIF dictionary, but parsed leniently so
    # that "4.0" is accepted and "4.5" is not.
    if parsed >= 1 and parsed == int(parsed):
        return int(parsed)
    return 1


def _parse_chemical_formula(formula: str) -> dict[str, float]:
    """Split a ``_chemical_formula_sum`` value into element counts.

    Parameters
    ----------
    formula : str
        Space-separated tokens such as ``"C2 H8 F4 N4 O3 Sb2"``. A token with
        no number means one atom.

    Returns
    -------
    dict[str, float]
        Atom counts keyed by the element database's symbol spelling.

    Notes
    -----
    The Java assigns rather than accumulates, so a formula naming the same
    element twice keeps only the last count. Matched here.
    """
    database = element_database()
    counts: dict[str, float] = {}
    for token in formula.split():
        # A two-letter symbol is one whose second character is a letter.
        length = 2 if len(token) > 1 and token[1].isalpha() else 1
        symbol = token[:length]
        remainder = token[length:]
        amount = float(remainder) if remainder else 1.0
        counts[database.by_symbol(symbol).name] = amount
    return counts


def _assemble(
    cell_volume: Array,
    counts: Mapping[str, Array],
    solvent_fraction: Array | None,
    solvent_concentrations: Mapping[str, Array] | None,
) -> Composition:
    """Turn accumulated atom counts into a :class:`Composition`."""
    database = element_database()
    totals: dict[str, Array] = dict(counts)

    def add(symbol: str, amount: Array) -> None:
        name = database.by_symbol(symbol).name
        totals[name] = totals.get(name, 0.0) + amount

    if solvent_fraction is not None:
        _add_solvent(add, cell_volume, solvent_fraction, solvent_concentrations)

    elements = sorted(
        (database.by_symbol(s) for s in totals), key=lambda e: e.atomic_number
    )
    xp, values = promote(*(totals[e.name] for e in elements))
    return Composition(
        elements=tuple(elements),
        counts=xp.stack(values),
        cell_volume=cell_volume,
        solvent_fraction=solvent_fraction,
    )
