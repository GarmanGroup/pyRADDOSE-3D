"""X-ray cross-sections for the chemical elements.

Ports ``Element.java`` and ``ElementDatabase.java``, which read a single
87-row by 80-column file, ``constants/MuCalcConstants.txt``, with no header
and no column names anywhere but a Java enum in another class. It holds
Pathikrit Bandyopadhyay's parameterisation of the McMaster tables: for each
of 87 elements, absorption edge energies and the coefficients of a cubic
polynomial in ``log(E)`` whose exponential gives a cross-section.

That file is split here into named, topical tables under ``data/``, each
with a header row and each narrow enough to read. Three of them are all this
module needs:

``elements.tsv``
    Atomic number, symbol, weight, mass and density.
``absorption_edges.tsv``
    The K, L1-L3 and M1-M5 edge energies in keV.
``cross_section_coefficients.tsv``
    One row per element per shell, four coefficients each.

Two more, ``fluorescence.tsv`` and ``edge_ratios.tsv``, are read by escape
modelling (:mod:`~pyraddose_3d.backend.escape`). The last two,
``electron_ionisation.tsv`` and ``elsepa_elastic.tsv``, carry the rest of the
original 80 columns and are read by nothing yet; they belong to the Monte
Carlo subprogram's electron transport.
``tests/backend/data/build_element_tables.py`` is the script that produced
all seven, kept so the derivation is auditable.

Energies are in keV and cross-sections in Barns/Atom throughout, matching the
Java.

Differentiability
-----------------
:meth:`Element.absorption_coefficients` is differentiable in ``energy`` and
runs on either backend. Absorption edges make it genuinely piecewise --
the cross-section really does jump by an order of magnitude at an edge -- so
the derivative is defined within a shell and undefined exactly at an edge.
Branch selection therefore uses ``where`` rather than Python ``if``, which
both vectorises over an array of energies and keeps the gradient intact.

Polynomial coefficients are held as plain Python floats, not arrays. A float
multiplied by a tensor is a tensor and by an ndarray an ndarray, so the
constants never force a backend.

Faithfulness to the Java
------------------------
Three oddities are reproduced deliberately, because the conformance corpus
pins current behaviour and a port error must not be confusable with an
intended fix:

1. At exactly ``E = 1`` keV the Java sums the coefficients instead of
   evaluating the polynomial, which introduces a discontinuity. Its own
   comment calls this "actually wrong ... However, this is how Pathikrit
   Bandyopadhyay chose to implement it, so it stays."
2. The L-edge jump corrections are applied to every element. The Java guards
   them with ``atomicNumber <= LIGHT_ATOM_MAX_NUM``, but that guard is
   commented out, above a comment reading "I don't think this is doing it
   correctly. It's not just for light atoms."
3. Missing M and N coefficients become zero rather than an error, so
   ``exp(0) = 1`` barn is returned for a shell that has no parameterisation.
   Those branches are unreachable for the elements concerned.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from pyraddose_3d.backend._array import array_namespace, is_torch

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["CrossSections", "Element", "ElementDatabase", "element_database"]

# --- the vendored tables ------------------------------------------------
_DATA_PACKAGE: Final = "pyraddose_3d.data"
_ELEMENTS_FILE: Final = "elements.tsv"
_EDGES_FILE: Final = "absorption_edges.tsv"
_COEFFICIENTS_FILE: Final = "cross_section_coefficients.tsv"

#: Shells and scattering types that have a polynomial. A shell an element has
#: no parameterisation for is absent from the coefficients table, and becomes
#: four zeros here -- see "Faithfulness to the Java" (3).
_SHELLS: Final = ("K", "L", "M", "N", "coherent", "incoherent")
_NO_COEFFICIENTS: Final = (0.0, 0.0, 0.0, 0.0)

# --- physical constants -------------------------------------------------
#: Atomic mass unit in grams, as spelled in Element.java.
ATOMIC_MASS_UNIT: Final = 1.66e-24

#: LJ_1 and LJ_2 from the original Fortran: McMaster tabulates against the L1
#: edge, so cross-sections between the L3/L2/L1 edges need the edge jumps
#: divided out.
_LJ_1: Final = 1.160
_LJ_2: Final = 1.41

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""


@dataclass(frozen=True)
class CrossSections:
    """X-ray cross-sections for one element, in Barns/Atom.

    Attributes
    ----------
    photoelectric : Array
        Photoelectric absorption. Does not contribute to scattering.
    coherent : Array
        Coherent (elastic, Rayleigh) scattering.
    compton : Array
        Incoherent (Compton) scattering.
    total : Array
        Sum of the three.
    """

    photoelectric: Array
    coherent: Array
    compton: Array
    total: Array


@dataclass(frozen=True)
class Element:
    """One chemical element's X-ray cross-section parameterisation.

    Attributes
    ----------
    name : str
        Chemical symbol, e.g. ``"S"``.
    atomic_number : int
        Number of protons.
    atomic_weight : float
        Atomic weight in unified atomic mass units.
    edge_k, edge_l, edge_m : float | None
        Absorption edge energies in keV. ``edge_k`` is always present;
        ``edge_m`` is absent for the 15 lightest elements.
    l2, l3 : float
        L2 and L3 edge energies in keV, used for the L-edge jump correction.
    coefficients : dict[str, tuple[float, ...]]
        Four-term polynomial coefficients keyed by shell (``"K"``, ``"L"``,
        ``"M"``, ``"N"``) and by scattering type (``"coherent"``,
        ``"incoherent"``).
    """

    name: str
    atomic_number: int
    atomic_weight: float
    edge_k: float
    edge_l: float | None
    edge_m: float | None
    l2: float
    l3: float
    coefficients: dict[str, tuple[float, ...]]

    @property
    def atomic_weight_in_grams(self) -> float:
        """Atomic weight in grams.

        Returns
        -------
        float
            ``atomic_weight`` converted by the atomic mass unit.
        """
        return self.atomic_weight * ATOMIC_MASS_UNIT

    @property
    def edges(self) -> tuple[float, ...]:
        """Energies at which the photoelectric cross-section jumps, in keV.

        The K, L1, L2, L3 and M edges -- every energy at which
        :meth:`absorption_coefficients` switches polynomial or applies an
        L-edge correction. Smooth between them, and not differentiable
        across them, so an optimiser over energy has to treat them as walls.

        Returns
        -------
        tuple[float, ...]
            Ascending; edges the data leaves blank or zero are omitted.
        """
        candidates = (self.edge_k, self.edge_l, self.l2, self.l3, self.edge_m)
        return tuple(sorted({edge for edge in candidates if edge}))

    def _bax(self, shell: str, energy: Array, log_energy: Array) -> Array:
        """Evaluate one polynomial: ``exp(sum_i c_i log(E)^i)``.

        Named for the Java, which calls this quantity ``bax`` after the
        original Fortran.

        Parameters
        ----------
        shell : str
            Key into :attr:`coefficients`.
        energy : Array
            Photon energy in keV. Needed only for the ``E == 1`` special case.
        log_energy : Array
            ``log(energy)``, passed in so it is computed once per call.

        Returns
        -------
        Array
            Cross-section contribution in Barns/Atom.
        """
        xp = array_namespace(energy)
        c_0, c_1, c_2, c_3 = self.coefficients[shell]
        total = c_0 + c_1 * log_energy + c_2 * log_energy**2 + c_3 * log_energy**3
        # See "Faithfulness to the Java" (1) in the module docstring. Note
        # that the polynomial above is already well defined at E = 1 -- it
        # evaluates to c_0 -- so this is a deliberate discontinuity, not a
        # guard against log(1) = 0.
        total = xp.where(energy == 1.0, c_0 + c_1 + c_2 + c_3, total)
        return xp.exp(total)

    def _photoelectric(self, energy: Array, log_energy: Array) -> Array:
        """Photoelectric cross-section, selecting the shell by energy.

        Parameters
        ----------
        energy : Array
            Photon energy in keV.
        log_energy : Array
            ``log(energy)``.

        Returns
        -------
        Array
            Photoelectric cross-section in Barns/Atom.
        """
        xp = array_namespace(energy)

        # The Java chain is:
        #     if   (E > K) or (L is None)  -> K
        #     elif (E > L3) or (M is None) -> L
        #     elif (E > M)                 -> M
        #     else                         -> N
        # `L is None` never happens: every row of MuCalcConstants.txt has an
        # L edge. `M is None` does, for the 15 lightest elements, and then
        # everything below the K edge uses the L polynomial.
        above_k = energy > self.edge_k
        in_k = self._bax("K", energy, log_energy)
        in_l = self._bax("L", energy, log_energy)

        if self.edge_m is None:
            # Below the K edge it is always L; M and N are unreachable.
            result = xp.where(above_k, in_k, in_l)
        else:
            in_n = self._bax("N", energy, log_energy)
            in_m = xp.where(
                energy > self.edge_m, self._bax("M", energy, log_energy), in_n
            )
            below_k = xp.where(energy > self.l3, in_l, in_m)
            result = xp.where(above_k, in_k, below_k)

        # McMaster tabulates against the L1 edge, so between L3 and L1 the
        # edge jumps have to be divided back out. Applied to every element --
        # see "Faithfulness to the Java" (2).
        in_l3_l2 = (energy > self.l3) & (energy < self.l2)
        result = xp.where(in_l3_l2, result / (_LJ_1 * _LJ_2), result)
        if self.edge_l is not None:
            in_l2_l1 = (energy > self.l2) & (energy < self.edge_l)
            result = xp.where(in_l2_l1, result / _LJ_1, result)
        return result

    def absorption_coefficients(self, energy: Array) -> CrossSections:
        """Photoelectric, coherent and Compton cross-sections at ``energy``.

        Differentiable in ``energy``, and vectorised over it: pass an array
        or tensor to evaluate many energies at once.

        Parameters
        ----------
        energy : Array
            Photon energy in keV. A Python float, a NumPy array or scalar, or
            a torch tensor. A tensor result carries a gradient.

        Returns
        -------
        CrossSections
            Cross-sections in Barns/Atom.
        """
        xp = array_namespace(energy)
        if not is_torch(energy):
            # A bare Python float has no `where`, so promote it. Tensors are
            # left alone: torch.asarray on one warns about requires_grad, and
            # there is nothing to promote.
            energy = np.asarray(energy)
        log_energy = xp.log(energy)

        photoelectric = self._photoelectric(energy, log_energy)

        # Both guards are per-element, not per-energy, so they stay as plain
        # Python conditionals: an element with no parameterisation for a
        # scattering type contributes exactly zero, not exp(0) = 1.
        zero = xp.zeros_like(photoelectric)
        coherent = (
            self._bax("coherent", energy, log_energy)
            if self.coefficients["coherent"][0] != 0
            else zero
        )
        compton = (
            self._bax("incoherent", energy, log_energy)
            if self.coefficients["incoherent"][0] != 0
            else zero
        )

        return CrossSections(
            photoelectric=photoelectric,
            coherent=coherent,
            compton=compton,
            total=photoelectric + coherent + compton,
        )


def _parse_optional(field: str) -> float | None:
    """Parse one field, treating the empty string as absent."""
    return None if field == "" else float(field)


def read_table(file_name: str) -> list[dict[str, str]]:
    """Read one vendored table into a list of rows keyed by column name.

    Comment lines and blank lines are skipped; the first line that is
    neither is the header.

    Parameters
    ----------
    file_name : str
        Name of a ``.tsv`` in :mod:`pyraddose_3d.data`.

    Returns
    -------
    list[dict[str, str]]
        One dict per data row, values unparsed.

    Raises
    ------
    ValueError
        If a row does not have exactly as many fields as the header. The
        single wide file this replaced had no such guarantee, and
        gadolinium's row carried two stray trailing tabs that nothing
        noticed for as long as the file has existed.
    """
    text = resources.files(_DATA_PACKAGE).joinpath(file_name).read_text("ascii")
    header: list[str] | None = None
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if header is None:
            header = fields
            continue
        if len(fields) != len(header):
            raise ValueError(
                f"{file_name}:{number} has {len(fields)} fields, but the "
                f"header declares {len(header)}"
            )
        rows.append(dict(zip(header, fields, strict=True)))
    if header is None:
        raise ValueError(f"{file_name} has no header row")
    return rows


def _read_coefficients() -> dict[int, dict[str, tuple[float, ...]]]:
    """Read the polynomial coefficients, keyed by atomic number then shell.

    Returns
    -------
    dict[int, dict[str, tuple[float, ...]]]
        Every element maps to all of :data:`_SHELLS`; a shell absent from
        the table becomes four zeros.
    """
    found: dict[int, dict[str, tuple[float, ...]]] = {}
    for row in read_table(_COEFFICIENTS_FILE):
        terms = tuple(float(row[f"c{i}"]) for i in range(4))
        found.setdefault(int(row["z"]), {})[row["shell"]] = terms
    return {
        z: {shell: shells.get(shell, _NO_COEFFICIENTS) for shell in _SHELLS}
        for z, shells in found.items()
    }


class ElementDatabase:
    """Lookup of :class:`Element` by chemical symbol or atomic number.

    Ports ``ElementDatabase.java``, which is a singleton reading the same
    file. Use :func:`element_database` to get the shared instance rather than
    constructing one per call; parsing is cheap but not free, and the
    elements are immutable.
    """

    def __init__(self) -> None:
        edges = {int(row["z"]): row for row in read_table(_EDGES_FILE)}
        coefficients = _read_coefficients()

        self._by_number: dict[int, Element] = {}
        self._by_symbol: dict[str, Element] = {}
        for row in read_table(_ELEMENTS_FILE):
            number = int(row["z"])
            edge = edges[number]
            element = Element(
                name=row["symbol"],
                atomic_number=number,
                atomic_weight=float(row["atomic_weight"]),
                edge_k=float(edge["k_kev"]),
                edge_l=_parse_optional(edge["l1_kev"]),
                edge_m=_parse_optional(edge["m1_kev"]),
                l2=float(edge["l2_kev"]),
                l3=float(edge["l3_kev"]),
                coefficients=coefficients[number],
            )
            self._by_number[number] = element
            self._by_symbol[element.name.lower()] = element

    def __len__(self) -> int:
        """Number of elements in the database."""
        return len(self._by_number)

    def __iter__(self) -> Iterator[Element]:
        """Iterate the elements in atomic-number order."""
        return iter([self._by_number[z] for z in sorted(self._by_number)])

    def by_number(self, atomic_number: int) -> Element:
        """Look an element up by atomic number.

        Parameters
        ----------
        atomic_number : int
            Number of protons.

        Returns
        -------
        Element
            The matching element.

        Raises
        ------
        KeyError
            If no element with that atomic number is tabulated.
        """
        try:
            return self._by_number[atomic_number]
        except KeyError:
            raise KeyError(
                f"no element with atomic number {atomic_number} in "
                f"{_ELEMENTS_FILE} (it covers Z = 1 to {max(self._by_number)})"
            ) from None

    def by_symbol(self, symbol: str) -> Element:
        """Look an element up by chemical symbol, case-insensitively.

        Parameters
        ----------
        symbol : str
            Chemical symbol, e.g. ``"S"`` or ``"fe"``.

        Returns
        -------
        Element
            The matching element.

        Raises
        ------
        KeyError
            If the symbol is not tabulated.
        """
        try:
            return self._by_symbol[symbol.lower()]
        except KeyError:
            raise KeyError(f"no element named {symbol!r} in {_ELEMENTS_FILE}") from None


@functools.lru_cache(maxsize=1)
def element_database() -> ElementDatabase:
    """Return the shared :class:`ElementDatabase`, parsing it on first call.

    Returns
    -------
    ElementDatabase
        The singleton database.
    """
    return ElementDatabase()
