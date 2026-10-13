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

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyraddose_3d.backend._array import array_namespace, promote
from pyraddose_3d.backend.elements import ATOMIC_MASS_UNIT, element_database
from pyraddose_3d.backend.residues import ResidueType, residue_database

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyraddose_3d.backend.elements import Element

__all__ = [
    "Chains",
    "Composition",
    "Macromolecule",
    "estimate_solvent_fraction",
    "parse_sequence",
    "protein_density",
    "read_sequence_file",
    "saxs_monomer_count",
    "saxs_sequence_monomer_count",
    "sequence_composition",
    "sequence_residue_counts",
    "small_molecule_composition",
    "unit_cell_composition",
]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""

Chains = tuple[tuple[ResidueType, str], ...]
"""A sequence file's residues: each line of codes, with what kind they are."""

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

#: Protein density from Fischer et al. (2004), Protein Science 13, 2825-2828:
#: ``rho = 1.410 + 0.145 exp(-M/13)``, with M the molecular weight of one
#: molecule in kDa. Small proteins pack less efficiently and are
#: correspondingly denser, which a single constant cannot express.
FISCHER_DENSITY_LIMIT = 1.410
FISCHER_DENSITY_RANGE = 0.145
FISCHER_DECAY_KDA = 13.0
_RNA_NUCLEOTIDE_MASS = 321.0
_RNA_DENSITY = 1.3
_DNA_NUCLEOTIDE_MASS = 312.0
_DNA_DENSITY = 1.35
_CARBOHYDRATE_AVERAGE_MASS = 162.0
_CARBOHYDRATE_DENSITY = 1.54
_HETATM_DENSITY = 1.35

#: Heteroatoms above this atomic number are left out of the solvent estimate,
#: so that a few heavy atoms cannot dominate it. Inclusive: copper counts as
#: light. RADDOSE-3D used ``<``, which excluded copper and disagreed both with
#: the constant's own description and with its only other use; a
#: "TODO: Is this < or <= ?!" stood against it. Fixed in both.
LIGHT_ATOM_MAX_NUMBER = 29

#: Atoms contributed by one residue of each kind. Averages over composition,
#: not the formula of any particular residue.
_PER_AMINO_ACID = {"C": 5.0, "N": 1.35, "O": 1.5, "H": 8.0}
_PER_RNA_NUCLEOTIDE = {"C": 9.5, "N": 3.75, "O": 7.0, "H": 11.25, "P": 1.0}
_PER_DNA_NUCLEOTIDE = {"C": 9.75, "N": 4.0, "O": 6.0, "H": 11.75, "P": 1.0}
_PER_CARBOHYDRATE = {"C": 6.0, "O": 5.0, "H": 11.0}

#: Average residue masses used only by the SAXS monomer-count calculation.
#: They differ from the ones above -- 339.5 and 327.0 rather than 321.0 and
#: 312.0 -- because CoefCalcSAXS carries its own copies.
_SAXS_AMINO_ACID_MASS = 110.0
_SAXS_RNA_MASS = 339.5
_SAXS_DNA_MASS = 327.0
_SAXS_CARBOHYDRATE_MASS = 180.0

_ANGSTROM_CUBED_TO_LITRE = 1e-27


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


def protein_density(amino_acids: int) -> float:
    """Density of a globular protein of the given size.

    From Fischer et al. (2004), Protein Science 13, 2825-2828:
    ``rho = 1.410 + 0.145 exp(-M/13)``, with M the molecular weight of one
    molecule in kDa. Small proteins pack less efficiently than large ones and
    are correspondingly denser, approaching 1.410 g/cm^3 as they grow.

    Parameters
    ----------
    amino_acids : int
        Residues in **one molecule**, not in the whole cell. The caller
        multiplies by the number of monomers separately.

    Returns
    -------
    float
        Density in grams per cubic centimetre.
    """
    molecular_weight_kda = _AMINO_ACID_AVERAGE_MASS * amino_acids / 1000.0
    return FISCHER_DENSITY_LIMIT + FISCHER_DENSITY_RANGE * math.exp(
        -molecular_weight_kda / FISCHER_DECAY_KDA
    )


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
        Atoms with atomic number above ``LIGHT_ATOM_MAX_NUMBER`` are
        ignored, so that a handful of heavy atoms cannot dominate.

    Returns
    -------
    Array
        Solvent fraction. Not clamped: a value below zero means the residue
        counts do not fit in the cell, and is worth seeing rather than
        hiding.

    Notes
    -----
    Protein density comes from :func:`protein_density`, which follows Fischer
    et al. (2004). RADDOSE-3D used a flat 1.35 g/cm^3 with that formula
    commented out beneath it; it was adopted in both codebases at once, so
    the two still agree. For a 129-residue protein it raises the density from
    1.35 to 1.459 and so the solvent fraction from 0.709 to 0.731.

    The densities used for RNA, DNA, carbohydrate and heteroatoms remain
    constants, as in the Java -- Fischer's fit is for globular proteins.
    """
    database = element_database()
    monomers = macromolecule.monomers

    def occupied(mass: float, count: int, density: float) -> Array:
        total = ATOMIC_MASS_UNIT * mass * count * monomers
        return total / (cell_volume * density * _ANGSTROMS_TO_ML)

    protein = occupied(
        _AMINO_ACID_AVERAGE_MASS,
        macromolecule.amino_acids,
        protein_density(macromolecule.amino_acids),
    )
    rna = occupied(_RNA_NUCLEOTIDE_MASS, macromolecule.rna_nucleotides, _RNA_DENSITY)
    dna = occupied(_DNA_NUCLEOTIDE_MASS, macromolecule.dna_nucleotides, _DNA_DENSITY)
    carbohydrate = occupied(
        _CARBOHYDRATE_AVERAGE_MASS,
        macromolecule.carbohydrate_residues,
        _CARBOHYDRATE_DENSITY,
    )

    hetatm_mass: Array = 0.0
    for symbol, count in (hetero_atoms or {}).items():
        element = database.by_symbol(symbol)
        if element.atomic_number <= LIGHT_ATOM_MAX_NUMBER:
            hetatm_mass = hetatm_mass + count * element.atomic_weight_in_grams
    hetatm = hetatm_mass / (cell_volume * _HETATM_DENSITY * _ANGSTROMS_TO_ML)

    return 1 - protein - rna - dna - hetatm - carbohydrate


def _add_solvent(
    add: Any,
    cell_volume: Array,
    solvent_fraction: Array,
    solvent_concentrations: Mapping[str, Array] | None,
) -> None:
    """Add solute and water atoms to a cell being assembled.

    Ports ``calculateSolventWater``. Solutes are counted first, because they
    displace water rather than adding to it.

    Parameters
    ----------
    add : Any
        Callback taking a chemical symbol and an amount.
    cell_volume : Array
        Cubic Angstroms.
    solvent_fraction : Array
        Fraction of the cell that is solvent.
    solvent_concentrations : Mapping[str, Array] | None
        Solute concentrations in millimoles per litre.
    """
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

    _add_solvent(add, cell_volume, solvent_fraction, solvent_concentrations)

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


def small_molecule_composition(
    cell_volume: Array,
    atoms: Mapping[str, Array],
    monomers: int = 1,
) -> Composition:
    """Assemble a small-molecule unit cell from explicit atom counts.

    Ports ``CoefCalcSmallMolecules``. There is no residue model and no
    solvent: a small-molecule crystal is its atoms and nothing else.

    Parameters
    ----------
    cell_volume : Array
        Unit cell volume in cubic Angstroms.
    atoms : Mapping[str, Array]
        Atoms per monomer, keyed by chemical symbol.
    monomers : int, optional
        Copies per unit cell. Zero is treated as one, as the Java does.

    Returns
    -------
    Composition
        Elements and counts per unit cell.

    Notes
    -----
    RADDOSE-3D accepts ``SolventHeavyConc`` alongside ``AbsCoefCalc
    SmallMole`` and then **silently ignores it**. The concentrations are
    collected into ``solventConcentration``, but the only thing that reads
    that map is ``calculateSolventWater``, which ``CoefCalcSmallMolecules``
    guards behind a local ``boolean fillRestWithWater = false`` -- a constant,
    with the comment "There are vacuums in small molecule crystals". So no
    solvent atom ever reaches the cell. This port therefore takes no solvent
    argument at all, rather than accepting one and discarding it.
    """
    database = element_database()
    if monomers == 0:
        monomers = 1

    counts: dict[str, Array] = {}
    for symbol, per_monomer in atoms.items():
        name = database.by_symbol(symbol).name
        counts[name] = counts.get(name, 0.0) + per_monomer * monomers

    elements = sorted(
        (database.by_symbol(s) for s in counts), key=lambda e: e.atomic_number
    )
    xp, values = promote(*(counts[e.name] for e in elements))
    return Composition(
        elements=tuple(elements),
        counts=xp.stack(values) if values else xp.asarray([]),
        cell_volume=cell_volume,
    )


def saxs_monomer_count(
    cell_volume: Array,
    protein_concentration: Array,
    macromolecule: Macromolecule,
) -> int:
    """How many monomers a SAXS sample volume contains.

    A SAXS "unit cell" is a box of solution rather than a crystallographic
    cell, so the number of molecules in it follows from the concentration.

    Parameters
    ----------
    cell_volume : Array
        Volume of the box in cubic Angstroms.
    protein_concentration : Array
        Concentration in grams per litre.
    macromolecule : Macromolecule
        Residue counts for one monomer, used to get its molecular weight.

    Returns
    -------
    int
        Monomers in the box, never fewer than one.

    Raises
    ------
    ValueError
        If the residue counts give a non-positive molecular weight.

    Notes
    -----
    ``CoefCalcSAXS`` used to weigh each nucleic acid with the other's average
    mass: ``calculateNumMonomers`` was declared ``(residues, DNA, RNA)`` and
    called ``(residues, RNA, DNA)``. The visible symptom was that equal counts
    of RNA and DNA gave different monomer numbers, with RNA -- the heavier
    residue -- giving *more* molecules per gram rather than fewer.

    Found during this port and fixed in both, so the two still agree. See
    RADDOSE-3D ``TEST-TRIAGE.md`` item 9 and ``CoefCalcSaxsMonomerTest``.
    """
    molecular_weight = (
        _SAXS_AMINO_ACID_MASS * macromolecule.amino_acids
        + _SAXS_RNA_MASS * macromolecule.rna_nucleotides
        + _SAXS_DNA_MASS * macromolecule.dna_nucleotides
        + _SAXS_CARBOHYDRATE_MASS * macromolecule.carbohydrate_residues
    )
    if molecular_weight <= 0:
        raise ValueError(
            "SAXS requires a positive monomer molecular weight. Set at least "
            "one of amino_acids, rna_nucleotides, dna_nucleotides or "
            "carbohydrate_residues to a non-zero value."
        )

    return _monomers_in(cell_volume, protein_concentration, molecular_weight)


def saxs_sequence_monomer_count(
    cell_volume: Array,
    protein_concentration: Array,
    chains: Chains,
    carbohydrate_residues: int = 0,
) -> int:
    """How many monomers a SAXS sample volume contains, weighed by sequence.

    Ports ``CoefCalcFromSequenceSAXS.calculateNumMonomers``. As
    :func:`saxs_monomer_count`, but the monomer's mass is the sum of its
    residues' real masses rather than a count times an average.

    Parameters
    ----------
    cell_volume : Array
        Volume of the box in cubic Angstroms.
    protein_concentration : Array
        Concentration in grams per litre.
    chains : Chains
        The monomer's residues, as :func:`parse_sequence` returns them.
    carbohydrate_residues : int, optional
        Sugar residues per monomer, weighed as ``CoefCalcSAXS`` weighs them.

    Returns
    -------
    int
        Monomers in the box, never fewer than one.

    Raises
    ------
    ValueError
        If the sequence has no residues, and so the monomer no mass.
    """
    database = residue_database()
    molecular_weight = 0.0
    for residue_type, residues in chains:
        for character in residues:
            if not character.isspace():
                residue = database.by_letter(character, residue_type)
                molecular_weight += residue.molecular_weight
    if molecular_weight <= 0:
        raise ValueError(
            "SAXS from a sequence needs a monomer with mass, and the sequence "
            "file has no residues"
        )
    molecular_weight += _SAXS_CARBOHYDRATE_MASS * carbohydrate_residues
    return _monomers_in(cell_volume, protein_concentration, molecular_weight)


def _monomers_in(
    cell_volume: Array, protein_concentration: Array, molecular_weight: float
) -> int:
    """Molecules of a given mass in a box of solution, at least one.

    Rounded half up, as ``Math.round``; Python's ``round`` would take a half
    to the even neighbour.
    """
    molarity = protein_concentration / molecular_weight
    volume_litres = _ANGSTROM_CUBED_TO_LITRE * cell_volume
    count = math.floor(float(molarity * volume_litres * AVOGADRO) + 0.5)
    return max(count, 1)


def parse_sequence(text: str) -> Chains:
    """Split a sequence file into lines of residue codes, each with its type.

    Ports ``CoefCalcFromSequence.parseSequenceFile``. A line starting ``>``
    is a header, and says what the residues after it are: DNA if it contains
    ``ISDNA``, RNA if it contains ``ISRNA``, and protein otherwise, in any
    case. Residues before any header are protein. A file can therefore hold
    several chains of different kinds. Blank lines are skipped.

    Every nucleotide code is also an amino acid code, so a DNA or RNA chain
    whose header lacks its marker reads as protein without error. As
    RADDOSE-3D does, a chain of :data:`NUCLEOTIDE_LOOKALIKE_LENGTH` or more
    residues read as protein but made only of A, C, G, T and U draws a
    warning.

    Parameters
    ----------
    text : str
        The file's contents.

    Returns
    -------
    Chains
        One entry per line of residues, in file order.

    Raises
    ------
    ValueError
        If a character is not a residue code of the type its header gives.
    """
    chains: list[tuple[ResidueType, str]] = []
    residue_type = ResidueType.PROTEIN
    protein_chain: list[str] = []

    def warn_if_nucleotides() -> None:
        chain = "".join(protein_chain)
        protein_chain.clear()
        if len(chain) >= NUCLEOTIDE_LOOKALIKE_LENGTH and set(chain) <= set("ACGTU"):
            warnings.warn(
                f"a chain of {len(chain)} residues was read as protein, but "
                "every residue is A, C, G, T or U; if it is DNA or RNA, put "
                "ISDNA or ISRNA in its '>' header line",
                stacklevel=3,
            )

    # As BufferedReader.readLine: a line ends at \n, \r or \r\n.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for number, line in enumerate(lines, start=1):
        # String.trim, which strips every character up to the space.
        if not line.strip(_JAVA_TRIMMED):
            continue
        if line.startswith(">"):
            warn_if_nucleotides()
            header = line.upper()
            if "ISDNA" in header:
                residue_type = ResidueType.DNA
            elif "ISRNA" in header:
                residue_type = ResidueType.RNA
            else:
                residue_type = ResidueType.PROTEIN
            continue
        try:
            sequence_residue_counts(line, residue_type)
        except ValueError as error:
            raise ValueError(f"line {number} of the sequence file: {error}") from None
        if residue_type is ResidueType.PROTEIN:
            protein_chain.extend(c.upper() for c in line if not c.isspace())
        chains.append((residue_type, line))
    warn_if_nucleotides()
    return tuple(chains)


#: ``CoefCalcFromSequence.NUCLEOTIDE_LOOKALIKE_LENGTH``: chains this long or
#: longer, read as protein but made only of nucleotide codes, are reported.
#: Shorter ones could well be peptides.
NUCLEOTIDE_LOOKALIKE_LENGTH = 10


#: The characters Java's ``String.trim`` removes: everything up to the space.
_JAVA_TRIMMED = "".join(chr(code) for code in range(33))


def read_sequence_file(path: str | Path) -> Chains:
    """Read a sequence file; see :func:`parse_sequence`.

    Parameters
    ----------
    path : str | Path
        A FASTA-style file of single-letter residue codes.

    Returns
    -------
    Chains
        One entry per line of residues, in file order.
    """
    return parse_sequence(Path(path).read_text("ascii"))


def sequence_residue_counts(
    sequence: str, residue_type: ResidueType = ResidueType.PROTEIN
) -> dict[str, float]:
    """Atom counts for one copy of a sequence.

    Ports the sequence-parsing half of ``CoefCalcFromSequence``: rather than
    approximating every residue as an average one, look each up and add its
    real composition.

    Parameters
    ----------
    sequence : str
        Single-letter residue codes. Whitespace is ignored.
    residue_type : ResidueType, optional
        Whether the sequence is protein, RNA or DNA. The same letter means
        different residues in each.

    Returns
    -------
    dict[str, float]
        Atoms per copy, keyed by chemical symbol.

    Raises
    ------
    ValueError
        If a character does not name a residue of that type. Stopping is
        deliberate: silently dropping it would give a polymer one residue
        short, and a dose for a molecule the caller did not describe.
    """
    database = residue_database()
    counts: dict[str, float] = {}
    for position, character in enumerate(sequence, start=1):
        # Whitespace is skipped rather than looked up: a sequence written on
        # Windows carries a carriage return at the end of every line.
        if character.isspace():
            continue
        try:
            residue = database.by_letter(character, residue_type)
        except KeyError:
            raise ValueError(
                f"{character!r} at position {position} of the sequence is not "
                f"a recognised {_TYPE_NAMES[residue_type]} residue code."
            ) from None
        for symbol, number in residue.atoms.items():
            counts[symbol] = counts.get(symbol, 0.0) + number
    return counts


#: ``CoefCalcFromSequence.residueTypeName``, for error messages.
_TYPE_NAMES = {
    ResidueType.PROTEIN: "protein",
    ResidueType.RNA: "RNA",
    ResidueType.DNA: "DNA",
}


def sequence_composition(
    cell_volume: Array,
    sequence: str | Chains,
    *,
    residue_type: ResidueType = ResidueType.PROTEIN,
    monomers: int = 1,
    heavy_protein_atoms: Mapping[str, Array] | None = None,
    solvent_concentrations: Mapping[str, Array] | None = None,
    solvent_fraction: Array | None = None,
    carbohydrate_residues: int = 0,
) -> Composition:
    """Assemble a unit cell from an actual sequence.

    Ports ``CoefCalcFromSequence``. Where :func:`unit_cell_composition`
    approximates every residue as an average one, this looks each up and adds
    its real atoms, so a cysteine-rich protein gets its sulphur.

    Parameters
    ----------
    cell_volume : Array
        Unit cell volume in cubic Angstroms.
    sequence : str | Chains
        Single-letter residue codes, whitespace ignored, or a sequence file's
        chains as :func:`parse_sequence` returns them.
    residue_type : ResidueType, optional
        Whether a ``str`` sequence is protein, RNA or DNA. Chains carry their
        own.
    monomers : int, optional
        Copies per unit cell.
    heavy_protein_atoms : Mapping[str, Array], optional
        Extra atoms bound to the macromolecule, per monomer.
    solvent_concentrations : Mapping[str, Array], optional
        Solute concentrations in millimoles per litre.
    solvent_fraction : Array, optional
        Estimated from the residue counts when not given.
    carbohydrate_residues : int, optional
        Sugar residues per monomer, added as average compositions since a
        sequence file cannot express them.

    Returns
    -------
    Composition
        Elements and counts per unit cell, ordered by atomic number.

    Notes
    -----
    Order matters and follows the Java. Solvent water is computed *before*
    the macromolecular atoms are multiplied up by ``monomers``, and the
    multiplication applies only to the macromolecule -- so solvent is counted
    once per cell while protein is counted once per copy.

    ``carbohydrate_residues`` contributes both atoms and volume, so it also
    reaches the solvent-fraction estimate. RADDOSE-3D recorded the count in
    ``addCarbs``, which ran after the estimate had already used it, so the
    estimate saw zero however many were specified; fixed in both.
    """
    database = element_database()
    chains = ((residue_type, sequence),) if isinstance(sequence, str) else sequence
    residue_counts: dict[str, float] = {}
    for kind, codes in chains:
        for symbol, number in sequence_residue_counts(codes, kind).items():
            residue_counts[symbol] = residue_counts.get(symbol, 0.0) + number

    per_monomer: dict[str, Array] = {}

    def add_macromolecular(symbol: str, amount: Array) -> None:
        name = database.by_symbol(symbol).name
        per_monomer[name] = per_monomer.get(name, 0.0) + amount

    for symbol, amount in (heavy_protein_atoms or {}).items():
        add_macromolecular(symbol, amount)
    for symbol, amount in residue_counts.items():
        add_macromolecular(symbol, amount)
    for symbol, per_one in _PER_CARBOHYDRATE.items():
        if carbohydrate_residues:
            add_macromolecular(symbol, per_one * carbohydrate_residues)

    residues = _residue_totals(chains, monomers, carbohydrate_residues)
    if solvent_fraction is None:
        solvent_fraction = estimate_solvent_fraction(cell_volume, residues)

    counts: dict[str, Array] = {}

    def add(symbol: str, amount: Array) -> None:
        name = database.by_symbol(symbol).name
        counts[name] = counts.get(name, 0.0) + amount

    _add_solvent(add, cell_volume, solvent_fraction, solvent_concentrations)

    # Only the macromolecule scales with the number of copies; the solvent
    # fills one cell however many copies are in it.
    for name, amount in per_monomer.items():
        add(name, amount * monomers)

    elements = sorted(
        (database.by_symbol(s) for s in counts), key=lambda e: e.atomic_number
    )
    xp, values = promote(*(counts[e.name] for e in elements))
    return Composition(
        elements=tuple(elements),
        counts=xp.stack(values),
        cell_volume=cell_volume,
        solvent_fraction=solvent_fraction,
    )


def _residue_totals(
    chains: Sequence[tuple[ResidueType, str]],
    monomers: int,
    carbohydrate_residues: int = 0,
) -> Macromolecule:
    """Count residues in a sequence, as a :class:`Macromolecule`.

    Carbohydrates cannot be expressed in a sequence file and are supplied
    separately, but they occupy space in the cell like anything else and so
    belong in the solvent estimate. RADDOSE-3D recorded them only in
    ``addCarbs``, which ran after the estimate had already been made; fixed in
    both.
    """
    lengths = dict.fromkeys(ResidueType, 0)
    for residue_type, residues in chains:
        lengths[residue_type] += sum(1 for c in residues if not c.isspace())
    return Macromolecule(
        monomers=monomers,
        amino_acids=lengths[ResidueType.PROTEIN],
        rna_nucleotides=lengths[ResidueType.RNA],
        dna_nucleotides=lengths[ResidueType.DNA],
        carbohydrate_residues=carbohydrate_residues,
    )
