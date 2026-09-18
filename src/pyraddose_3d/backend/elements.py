"""X-ray cross-sections for the chemical elements.

Ports ``Element.java`` and ``ElementDatabase.java``. Both read the same
vendored constants file, ``data/MuCalcConstants.txt``, which holds Pathikrit
Bandyopadhyay's parameterisation of the McMaster tables: for each of 87
elements, absorption edge energies and the coefficients of a cubic polynomial
in ``log(E)`` whose exponential gives a cross-section.

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

# --- layout of MuCalcConstants.txt --------------------------------------
# Zero-based column indices, transcribed from ElementDatabase.DatabaseFields.
# The file is tab-separated with 80 columns; only those the X-ray path needs
# are named here. The rest carry fluorescence yields, edge ratios and ELSEPA
# elastic cross-sections, used only by the Monte Carlo paths.
_ATOMIC_NUMBER: Final = 0
_ELEMENT_NAME: Final = 1
_EDGE_K: Final = 2
_EDGE_L: Final = 3
_EDGE_M: Final = 4
_ATOMIC_WEIGHT: Final = 23
_L2: Final = 36
_L3: Final = 37

# First column of each four-term polynomial in log(E).
_POLYNOMIALS: Final[dict[str, int]] = {
    "K": 5,
    "L": 9,
    "M": 13,
    "N": 17,
    "coherent": 24,
    "incoherent": 28,
}
_POLYNOMIAL_TERMS: Final = 4

_DATA_PACKAGE: Final = "pyraddose_3d.data"
_DATA_FILE: Final = "MuCalcConstants.txt"

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


def _parse_line(line: str) -> Element:
    """Build an :class:`Element` from one tab-separated row of the database."""
    columns = line.split("\t")

    coefficients: dict[str, tuple[float, ...]] = {}
    for shell, first in _POLYNOMIALS.items():
        raw = columns[first : first + _POLYNOMIAL_TERMS]
        # Missing M and N coefficients become zero -- see "Faithfulness to
        # the Java" (3).
        coefficients[shell] = tuple(0.0 if c == "" else float(c) for c in raw)

    return Element(
        name=columns[_ELEMENT_NAME],
        atomic_number=int(columns[_ATOMIC_NUMBER]),
        atomic_weight=float(columns[_ATOMIC_WEIGHT]),
        edge_k=float(columns[_EDGE_K]),
        edge_l=_parse_optional(columns[_EDGE_L]),
        edge_m=_parse_optional(columns[_EDGE_M]),
        l2=float(columns[_L2]),
        l3=float(columns[_L3]),
        coefficients=coefficients,
    )


class ElementDatabase:
    """Lookup of :class:`Element` by chemical symbol or atomic number.

    Ports ``ElementDatabase.java``, which is a singleton reading the same
    file. Use :func:`element_database` to get the shared instance rather than
    constructing one per call; parsing is cheap but not free, and the
    elements are immutable.
    """

    def __init__(self) -> None:
        text = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE).read_text("ascii")
        self._by_number: dict[int, Element] = {}
        self._by_symbol: dict[str, Element] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            element = _parse_line(line)
            self._by_number[element.atomic_number] = element
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
                f"{_DATA_FILE} (it covers Z = 1 to {max(self._by_number)})"
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
            raise KeyError(f"no element named {symbol!r} in {_DATA_FILE}") from None


@functools.lru_cache(maxsize=1)
def element_database() -> ElementDatabase:
    """Return the shared :class:`ElementDatabase`, parsing it on first call.

    Returns
    -------
    ElementDatabase
        The singleton database.
    """
    return ElementDatabase()
