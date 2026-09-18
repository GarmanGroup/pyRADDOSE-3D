"""Loader for the CoefCalc reference fixture.

Shared by ``test_composition.py``, which checks that the right atoms come out
of a description of the crystal, and ``test_coefficients.py``, which takes
those atoms as given and checks the arithmetic on top of them.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass, field

REFERENCE = pathlib.Path(__file__).parent / "data" / "coefficients.tsv"


@dataclass
class Case:
    """One reference composition: its inputs, and what the Java made of them."""

    id: str
    cell: tuple[float, ...] = ()
    cell_volume: float = 0.0
    density: float = 0.0
    solvent_fraction: float = 0.0
    monomers: int = 0
    amino_acids: int = 0
    rna_nucleotides: int = 0
    dna_nucleotides: int = 0
    #: None when RADDOSE-3D was left to estimate the solvent fraction.
    given_solvent_fraction: float | None = None
    heavy_protein_atoms: dict[str, float] = field(default_factory=dict)
    solvent_concentrations: dict[str, float] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    counts: list[float] = field(default_factory=list)
    coefficients: list[tuple[float, ...]] = field(default_factory=list)

    @property
    def estimates_solvent_fraction(self) -> bool:
        """Whether the solvent fraction was computed rather than supplied."""
        return self.given_solvent_fraction is None


def load_cases() -> dict[str, Case]:
    """Parse the reference fixture into cases, keyed by id."""
    cases: dict[str, Case] = {}
    with REFERENCE.open(encoding="ascii") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            tag, case_id = row[0], row[1]
            case = cases.setdefault(case_id, Case(case_id))
            if tag == "CASE":
                case.cell = tuple(float(v) for v in row[2:8])
            elif tag == "INPUT":
                case.monomers = int(row[2])
                case.amino_acids = int(row[3])
                case.rna_nucleotides = int(row[4])
                case.dna_nucleotides = int(row[5])
                case.given_solvent_fraction = (
                    None if row[6] == "auto" else float(row[6])
                )
            elif tag == "HAPROT":
                case.heavy_protein_atoms[row[2]] = float(row[3])
            elif tag == "HASOLV":
                case.solvent_concentrations[row[2]] = float(row[3])
            elif tag == "CELLVOL":
                case.cell_volume = float(row[2])
            elif tag == "DENSITY":
                case.density = float(row[2])
            elif tag == "SOLFRAC":
                case.solvent_fraction = float(row[2])
            elif tag == "ATOMS":
                case.symbols.append(row[3])
                case.counts.append(float(row[4]))
            elif tag == "COEF":
                case.coefficients.append(tuple(float(v) for v in row[2:7]))
    return cases


CASES = load_cases()
CASE_IDS = sorted(CASES)
