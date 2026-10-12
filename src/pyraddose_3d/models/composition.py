"""Sample composition: how absorption coefficients are determined.

Maps onto RADDOSE-3D's ``AbsCoefCalc`` directive. Each variant needs a
different set of parameters, so these are a discriminated union rather than
one model with many optional fields -- an incomplete composition then fails at
load time rather than part way through a run.

A field has a default only where RADDOSE-3D has one, and it is RADDOSE-3D's.
The Java's residue counts are uninitialised ``int`` fields, so an omitted
count is zero; the unit cell and the monomer count have no usable default and
are required. These used to default to insulin's values, so a configuration
that left out ``num_residues`` was silently insulin.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from pyraddose_3d.models._base import RD3DModel

__all__ = [
    "AverageComposition",
    "CifComposition",
    "Composition",
    "PdbComposition",
    "SaxsComposition",
    "SequenceComposition",
    "SmallMoleculeComposition",
    "UnitCellComposition",
]

#: Element symbol paired with a count or concentration.
ElementAmounts = dict[str, float]


class _CellMixin(RD3DModel):
    """Unit cell shared by the composition variants that need one."""

    unit_cell: tuple[float, float, float]
    unit_cell_angles: tuple[float, float, float] | None = None


class AverageComposition(RD3DModel):
    """Generic "average protein" coefficients (``AbsCoefCalc Average``).

    Takes no composition parameters at all; useful as a control case.
    """

    method: Literal["average"] = "average"


class UnitCellComposition(_CellMixin):
    """Composition from unit cell contents (``AbsCoefCalc RD3D``).

    This is the standard MX route.

    Attributes
    ----------
    num_monomers : int
        Monomers in the unit cell.
    num_residues : int
        Amino acid residues per monomer.
    num_rna, num_dna : int
        Nucleotides per monomer.
    num_carb : int
        Carbohydrate residues per monomer.
    protein_heavy_atoms : dict[str, float]
        Heavy atoms added to the protein part of each monomer, by symbol.
    solvent_heavy_conc : dict[str, float]
        Heavy element concentrations in the solvent, in mmol/l. Oxygen and
        lighter elements should not be given.
    solvent_fraction : float | None
        Fraction of the unit cell occupied by solvent. If ``None``,
        RADDOSE-3D computes it from the cell contents.
    """

    method: Literal["unit_cell"] = "unit_cell"
    num_monomers: Annotated[int, Field(gt=0)]
    num_residues: Annotated[int, Field(ge=0)] = 0
    num_rna: Annotated[int, Field(ge=0)] = 0
    num_dna: Annotated[int, Field(ge=0)] = 0
    num_carb: Annotated[int, Field(ge=0)] = 0
    protein_heavy_atoms: ElementAmounts = Field(default_factory=dict)
    solvent_heavy_conc: ElementAmounts = Field(default_factory=dict)
    solvent_fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class SmallMoleculeComposition(_CellMixin):
    """Composition of a small-molecule crystal (``AbsCoefCalc SmallMole``).

    Attributes
    ----------
    atoms : dict[str, float]
        Atoms per unit cell asymmetric unit, by element symbol.
    num_monomers : int
        Monomers in the unit cell.
    """

    method: Literal["small_molecule"] = "small_molecule"
    atoms: ElementAmounts
    num_monomers: Annotated[int, Field(gt=0)]


class SaxsComposition(_CellMixin):
    """Composition of a SAXS sample (``AbsCoefCalc SAXS``).

    Attributes
    ----------
    protein_concentration : float
        Protein concentration in grams per litre.
    num_residues : int
        Residues per monomer. At least one of the residue counts must be
        non-zero, or the monomer has no mass.
    num_carb : int
        Carbohydrate residues per monomer.
    """

    method: Literal["saxs"] = "saxs"
    protein_concentration: Annotated[float, Field(gt=0.0)]
    num_residues: Annotated[int, Field(ge=0)] = 0
    num_rna: Annotated[int, Field(ge=0)] = 0
    num_dna: Annotated[int, Field(ge=0)] = 0
    num_carb: Annotated[int, Field(ge=0)] = 0
    protein_heavy_atoms: ElementAmounts = Field(default_factory=dict)
    solvent_heavy_conc: ElementAmounts = Field(default_factory=dict)
    solvent_fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class SequenceComposition(_CellMixin):
    """Composition read from a sequence file (``AbsCoefCalc Sequence``).

    Attributes
    ----------
    sequence_file : str
        Path to a FASTA file, resolved relative to the working directory of
        the run.
    num_carb : int
        Carbohydrate residues per monomer, which a sequence cannot express.
    """

    method: Literal["sequence"] = "sequence"
    sequence_file: str
    num_monomers: Annotated[int, Field(gt=0)]
    num_carb: Annotated[int, Field(ge=0)] = 0
    protein_heavy_atoms: ElementAmounts = Field(default_factory=dict)
    solvent_heavy_conc: ElementAmounts = Field(default_factory=dict)
    solvent_fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class PdbComposition(RD3DModel):
    """Composition read from a PDB entry (``AbsCoefCalc EXP`` or ``PDB``).

    ``EXP`` is the original keyword: the grammar's token *named* PDB matches
    the literal "EXP", exactly as the token named CIF matches "EXPSM".
    RADDOSE-3D has accepted ``PDB`` as well since 2026.

    The unit cell, symmetry and solvent fraction all come from the file, so
    none of the usual cell parameters apply here.

    Attributes
    ----------
    pdb : str
        Either a path ending in ``.pdb``, read locally, or a four-character
        PDB code, which the reference implementation downloads from RCSB.
        Prefer a local path: a downloading test is not reproducible.
    """

    method: Literal["pdb"] = "pdb"
    pdb: str
    solvent_heavy_conc: ElementAmounts = Field(default_factory=dict)


class CifComposition(RD3DModel):
    """Composition read from a small-molecule CIF (``AbsCoefCalc EXPSM`` or ``CIF``).

    ``EXPSM`` is the original keyword: the grammar's token *named* CIF matches
    the literal "EXPSM". RADDOSE-3D has accepted ``CIF`` as well since 2026.

    This is the *small-molecule* route. It needs ``_chemical_formula_sum`` and
    ``_cell_volume``, so a protein mmCIF (with dotted tags such as
    ``_cell.length_a``) will not work.

    Attributes
    ----------
    cif : str
        Path to the CIF file.
    """

    method: Literal["cif"] = "cif"
    cif: str
    solvent_heavy_conc: ElementAmounts = Field(default_factory=dict)


Composition = Annotated[
    AverageComposition
    | UnitCellComposition
    | PdbComposition
    | CifComposition
    | SmallMoleculeComposition
    | SaxsComposition
    | SequenceComposition,
    Field(discriminator="method"),
]
