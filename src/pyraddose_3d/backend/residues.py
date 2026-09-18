"""Residue compositions, for building a crystal from its sequence.

Ports ``Residue.java`` and ``ResidueDatabase.java``. Where
:mod:`pyraddose_3d.backend.composition` can approximate a protein as "so many
average amino acids", this gives the actual atom counts of each residue, so a
sequence file yields a real composition rather than an average one.

The table is vendored as ``data/residues.tsv``, dumped from the Java rather
than transcribed: ``ResidueDatabase.java`` is 815 lines of hand-written
constants, and re-keying those by hand would be a poor use of anyone's
afternoon and an excellent source of typos.

Residues are keyed by a three-character PDB-style identifier -- ``"ALA"`` for
alanine, ``"  A"`` (right-justified in three columns) for adenine. Sequence
files use single letters, which mean different residues depending on whether
the sequence is protein, RNA or DNA, so the letter-to-key conversion is part
of the vendored data too.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import IntEnum
from importlib import resources
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Residue", "ResidueDatabase", "ResidueType", "residue_database"]

_DATA_PACKAGE: Final = "pyraddose_3d.data"
_DATA_FILE: Final = "residues.tsv"


class ResidueType(IntEnum):
    """Which kind of polymer a residue belongs to.

    The values are the Java's, and appear in the vendored table.
    """

    PROTEIN = 1
    RNA = 2
    DNA = 3


@dataclass(frozen=True)
class Residue:
    """One residue's atomic composition.

    Attributes
    ----------
    key : str
        Three-character identifier, as used in a PDB file. Nucleotides are
        right-justified, so adenine is ``"  A"``.
    identifier : str
        The identifier the Java stores on the residue itself. Usually equal
        to ``key``.
    residue_type : ResidueType
        Protein, RNA or DNA.
    carbons, hydrogens, oxygens, nitrogens, sulphurs, seleniums, phosphoruses : int
        Atoms per residue.
    molecular_weight : float
        Residue mass in unified atomic mass units.
    """

    key: str
    identifier: str
    residue_type: ResidueType
    carbons: int
    hydrogens: int
    oxygens: int
    nitrogens: int
    sulphurs: int
    seleniums: int
    phosphoruses: int
    molecular_weight: float

    @property
    def atoms(self) -> dict[str, int]:
        """Atom counts keyed by chemical symbol, omitting absent elements.

        Returns
        -------
        dict[str, int]
            Symbols as the element database spells them.
        """
        counts = {
            "C": self.carbons,
            "H": self.hydrogens,
            "O": self.oxygens,
            "N": self.nitrogens,
            "S": self.sulphurs,
            "SE": self.seleniums,
            "P": self.phosphoruses,
        }
        return {symbol: n for symbol, n in counts.items() if n}


class ResidueDatabase:
    """Lookup of :class:`Residue` by three-character key or sequence letter.

    Use :func:`residue_database` for the shared instance.
    """

    def __init__(self) -> None:
        text = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE).read_text("ascii")
        self._by_key: dict[str, Residue] = {}
        # (residue type, upper-case letter) -> three-character key.
        self._letters: dict[tuple[ResidueType, str], str] = {}

        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] == "RES":
                residue = Residue(
                    key=fields[1],
                    identifier=fields[2],
                    residue_type=ResidueType(int(fields[3])),
                    carbons=int(fields[4]),
                    hydrogens=int(fields[5]),
                    oxygens=int(fields[6]),
                    nitrogens=int(fields[7]),
                    sulphurs=int(fields[8]),
                    seleniums=int(fields[9]),
                    phosphoruses=int(fields[10]),
                    molecular_weight=float(fields[11]),
                )
                self._by_key[residue.key] = residue
            elif fields[0] == "LETTER" and fields[4] == "OK":
                self._letters[ResidueType(int(fields[1])), fields[2]] = fields[3]

    def __len__(self) -> int:
        """Number of distinct residues in the table."""
        return len(self._by_key)

    def __iter__(self) -> Iterator[Residue]:
        """Iterate the residues in key order."""
        return iter([self._by_key[k] for k in sorted(self._by_key)])

    def by_key(self, key: str) -> Residue:
        """Look a residue up by its three-character identifier.

        Parameters
        ----------
        key : str
            Three-character key, e.g. ``"ALA"`` or ``"  A"``.

        Returns
        -------
        Residue
            The matching residue.

        Raises
        ------
        KeyError
            If the key is not tabulated.
        """
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"no residue keyed {key!r} in {_DATA_FILE}") from None

    def by_letter(self, letter: str, residue_type: ResidueType) -> Residue:
        """Look a residue up by its single-letter sequence code.

        Parameters
        ----------
        letter : str
            One character. Case-insensitive.
        residue_type : ResidueType
            Which polymer the sequence is, since the same letter means
            different residues in a protein and in DNA.

        Returns
        -------
        Residue
            The matching residue.

        Raises
        ------
        KeyError
            If the letter does not name a residue of that type. The Java
            prints a warning and then dereferences the null, so an
            unrecognised character in a sequence file crashes with a
            ``NullPointerException`` one line later; raising here says what
            actually went wrong.
        """
        try:
            key = self._letters[residue_type, letter.upper()]
        except KeyError:
            raise KeyError(
                f"{letter!r} does not name a {residue_type.name.lower()} residue"
            ) from None
        return self._by_key[key]


@functools.lru_cache(maxsize=1)
def residue_database() -> ResidueDatabase:
    """Return the shared :class:`ResidueDatabase`, parsing it on first call.

    Returns
    -------
    ResidueDatabase
        The singleton database.
    """
    return ResidueDatabase()
