"""Reading a unit cell composition from a PDB or CIF file.

Ports ``CoefCalcFromPDB`` and ``CoefCalcFromCIF``, the two ``AbsCoefCalc``
routes that read a structure file rather than taking parameters. RADDOSE-3D
spells them ``EXP`` and ``EXPSM`` respectively -- not ``PDB`` and ``CIF``,
which is a naming accident in the grammar rather than a choice.

Neither parser is a general structure-file reader. The PDB one wants six
record types and ignores the rest; the CIF one wants two data items. They are
ported at exactly that scope, because widening them would change results.

Known defect
------------
**The PDB path aborts on a short line.** ``readPDBFile`` catches
``IndexOutOfBoundsException`` *outside* its read loop, so the first line too
short for a fixed-column slice stops parsing of the whole remaining file, and
the composition is built from however much was read. Well-formed PDB files
are padded to 80 columns so this does not fire; files from tools that do not
pad are at risk. Python's slicing does not raise on a short string, so the
port simply reads such a line as blank -- see :func:`composition_from_pdb`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

__all__ = ["composition_from_cif", "composition_from_pdb"]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""

# --- PDB fixed-column positions, transcribed from CoefCalcFromPDB.java ---
_DIRECTIVE = slice(0, 6)
_CRYST1_FIELDS = (
    slice(6, 15),  # a
    slice(15, 24),  # b
    slice(24, 33),  # c
    slice(33, 40),  # alpha
    slice(40, 47),  # beta
    slice(47, 54),  # gamma
)
_ATOM_RESIDUE_NAME = slice(17, 20)
_OCCUPANCY = slice(54, 60)
_ELEMENT_SYMBOL = slice(76, 78)
_SMTRY_FLAG = slice(13, 19)
_MTRIX_FLAG = slice(59, 60)
_SEQRES_START = 19
_SEQRES_RESIDUE_LENGTH = 3
#: SEQRES residue names are three characters with one space between them.
_SEQRES_STRIDE = _SEQRES_RESIDUE_LENGTH + 1


@dataclass
class _PdbRecords:
    """What a PDB file contributes to a composition, before scaling."""

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


def _pdb_records(text: str) -> _PdbRecords:
    """Read the records a composition needs out of a PDB file."""
    database = residue_database()

    hetero_atoms: dict[str, float] = {}
    residue_atoms: dict[str, float] = {}
    amino_acids = rna = dna = 0
    crystallographic_operators = 0
    # One is the identity, which the file does not list.
    non_crystallographic_operators = 1
    cell_volume: Any = None

    for line in text.splitlines():
        directive = line[_DIRECTIVE]

        if directive == "CRYST1":
            values = [line[field] for field in _CRYST1_FIELDS]
            try:
                cell_volume = unit_cell_volume(*(float(v) for v in values))
            except ValueError:
                # The Java prints "CRYST1 line could not be parsed" and
                # carries on with no cell volume at all.
                cell_volume = None

        elif directive == "HETATM":
            # Water is solvent, and is counted by the solvent model rather
            # than atom by atom.
            if line[_ATOM_RESIDUE_NAME].strip().upper() == "HOH":
                continue
            symbol = line[_ELEMENT_SYMBOL].strip().upper()
            if not symbol:
                continue
            occupancy = line[_OCCUPANCY].strip()
            try:
                amount = float(occupancy) if occupancy else 1.0
            except ValueError:
                # "occupancy column does not contain a valid number".
                amount = 1.0
            hetero_atoms[symbol] = hetero_atoms.get(symbol, 0.0) + amount

        elif directive == "SEQRES":
            for name in _seqres_names(line):
                try:
                    residue = database.by_key(name)
                except KeyError:
                    # "could not decipher PDB three letter code".
                    continue
                if residue.residue_type is ResidueType.PROTEIN:
                    amino_acids += 1
                elif residue.residue_type is ResidueType.RNA:
                    rna += 1
                else:
                    dna += 1
                for symbol, number in residue.atoms.items():
                    residue_atoms[symbol] = residue_atoms.get(symbol, 0.0) + number

        elif directive == "REMARK":
            if line[_SMTRY_FLAG] == "SMTRY1":
                crystallographic_operators += 1

        elif directive == "MTRIX1":
            # A flag of "1" marks an operator already applied to the
            # coordinates, so it must not be counted again.
            if line[_MTRIX_FLAG] != "1":
                non_crystallographic_operators += 1

    # Residue atoms and heteroatoms both belong to the macromolecule as far
    # as the occurrence maps are concerned; heteroatoms are additionally
    # tracked on their own, because the solvent estimate treats them
    # differently.
    macromolecular = dict(residue_atoms)
    for symbol, amount in hetero_atoms.items():
        macromolecular[symbol] = macromolecular.get(symbol, 0.0) + amount

    return _PdbRecords(
        macromolecular=macromolecular,
        hetero_atoms=hetero_atoms,
        residues=Macromolecule(
            amino_acids=amino_acids, rna_nucleotides=rna, dna_nucleotides=dna
        ),
        crystallographic_operators=crystallographic_operators,
        non_crystallographic_operators=non_crystallographic_operators,
        cell_volume=cell_volume,
    )


def _seqres_names(line: str) -> list[str]:
    """Split a SEQRES line into three-character residue names."""
    rest = line[_SEQRES_START:]
    names = []
    while len(rest) > _SEQRES_RESIDUE_LENGTH:
        name = rest[:_SEQRES_RESIDUE_LENGTH]
        if name.strip():
            names.append(name)
        rest = rest[_SEQRES_STRIDE:]
    return names


def composition_from_pdb(
    path: str | Path,
    solvent_concentrations: Mapping[str, Array] | None = None,
) -> Composition:
    """Read a unit cell composition from a PDB file.

    Ports ``CoefCalcFromPDB``, minus its ability to download an entry from
    RCSB by four-letter code. Six record types are read:

    ``CRYST1``
        Unit cell dimensions.
    ``SEQRES``
        The sequence, from which residue compositions are looked up. Note
        that this is the *stated* sequence, not the modelled one, so
        disordered residues absent from the coordinates still count.
    ``HETATM``
        Everything that is not a standard residue, weighted by occupancy.
        Waters are skipped, being handled by the solvent model.
    ``REMARK ... SMTRY1``
        One per crystallographic symmetry operator.
    ``MTRIX1``
        One per non-crystallographic operator, unless flagged as already
        applied to the coordinates.

    The number of monomers in the cell is the product of the two operator
    counts. The Z value on the ``CRYST1`` line is deliberately ignored: the
    Java's comment says it "is often mis-calculated in the PDB ... when there
    is non-crystallographic symmetry involved".

    Parameters
    ----------
    path : str | Path
        Path to a PDB file.
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
        If the file has no usable ``CRYST1`` record, since without a cell
        volume there is nothing to divide by.

    Notes
    -----
    Python slices do not raise on a short string, so a file whose lines are
    not padded to 80 columns reads the missing columns as blank. The Java
    raises ``IndexOutOfBoundsException`` and, because the catch sits outside
    the read loop, abandons the rest of the file. The two agree on any
    well-formed PDB file, which is padded.
    """
    records = _pdb_records(Path(path).read_text("utf-8", errors="replace"))

    if records.cell_volume is None:
        raise ValueError(
            f"{path} has no usable CRYST1 record, so the unit cell volume is "
            f"unknown and no composition can be built from it"
        )

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
#: A CIF numeric value may carry a parenthesised standard uncertainty.
_CIF_NUMBER = re.compile(r"[0-9]")


def composition_from_cif(path: str | Path) -> Composition:
    """Read a unit cell composition from a small-molecule CIF file.

    Ports ``CoefCalcFromCIF``, which RADDOSE-3D selects for ``AbsCoefCalc
    EXPSM``. Exactly two data items are read: ``_chemical_formula_sum`` for
    the contents and ``_cell_volume`` for the cell. There is no solvent
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
    """
    text = Path(path).read_text("utf-8", errors="replace")

    formula: str | None = None
    cell_volume: Any = None
    formula_units = 1

    for line in text.splitlines():
        space = line.find(" ")
        if space < 0:
            continue
        directive = line[:space].strip()

        if directive == _CIF_FORMULA_KEY:
            first = line.find("'")
            last = line.rfind("'")
            if first >= 0 and last > first:
                formula = line[first + 1 : last]
        elif directive == _CIF_FORMULA_UNITS_KEY:
            formula_units = _parse_formula_units(line[space:].strip())
        elif directive == _CIF_VOLUME_KEY:
            digit = _CIF_NUMBER.search(line)
            if digit is None:
                continue
            value = line[digit.start() :]
            # A parenthesised standard uncertainty follows the value.
            bracket = value.find("(")
            if bracket >= 0:
                value = value[:bracket]
            try:
                cell_volume = float(value.strip())
            except ValueError:
                # "Not a valid unit cell". The Java leaves the volume at
                # whatever it already was.
                continue

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
