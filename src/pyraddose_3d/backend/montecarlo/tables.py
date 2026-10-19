"""The atomic tables RADDOSE-3D's Monte Carlo and XFEL subprograms read.

Auger and fluorescence transitions, as ``MC.populateAugerLinewidths`` and
``populateFluorescenceLinewidths`` hold them, and the overvoltage
coefficients of electron impact ionisation, as
``CoefCalcCompute.populateCrossSectionCoefficients`` does. All are vendored
from RADDOSE-3D's ``constants/`` by
``tests/backend/data/build_monte_carlo_tables.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from pyraddose_3d.backend.elements import read_table

__all__ = [
    "AUGER_ELEMENTS",
    "PackedTransitions",
    "Transitions",
    "auger_shells",
    "auger_transitions",
    "fluorescence_transitions",
    "overvoltage_high",
    "overvoltage_low",
    "packed_auger",
    "packed_fluorescence",
]

#: The elements whose vacancies relax by Auger emission or fluorescence:
#: ``MC.augerElements``. Phosphorus is among them, with tables, but the
#: cascade's own list of elements leaves it out, so RADDOSE-3D never relaxes
#: it.
AUGER_ELEMENTS = (
    6,
    7,
    8,
    11,
    12,
    14,
    15,
    16,
    17,
    19,
    20,
    25,
    26,
    27,
    28,
    29,
    30,
    33,
    34,
)

#: Slots per vacancy: RADDOSE-3D's arrays have 65, zero past the table's rows.
_SLOTS = 65

#: Shells per element in ``lowOvervoltages`` and ``highOvervoltages``.
_OVERVOLTAGE_SHELLS = 9


@dataclass(frozen=True)
class Transitions:
    """The transitions that fill one vacancy in one element.

    Each array has RADDOSE-3D's 65 slots, zero past the table's rows: a
    random number past the last cumulative probability selects a slot of
    zeros there too.

    Attributes
    ----------
    linewidth_ev : numpy.ndarray
        Natural linewidth of each transition, eV.
    probability : numpy.ndarray
        Its relative probability, as tabulated.
    cumulative : numpy.ndarray
        Running sum of the probabilities, scaled by ``1 / total`` as the
        Java scales it: by multiplying by the reciprocal, not dividing.
    energy_kev : numpy.ndarray
        Energy of the emitted electron or photon, keV.
    exit_shell : numpy.ndarray | None
        For Auger transitions, the vacancy the emitted electron leaves.
    drop_shell : numpy.ndarray
        The vacancy the electron that fills this one leaves.
    total : float
        The sum of the probabilities.
    rows : int
        How many transitions the table has; the slots after them are zero.
    """

    linewidth_ev: np.ndarray
    probability: np.ndarray
    cumulative: np.ndarray
    energy_kev: np.ndarray
    exit_shell: np.ndarray | None
    drop_shell: np.ndarray
    total: float
    rows: int


def auger_shells(z: int) -> int:
    """``MC.getNumAugerShells``: the highest vacancy that relaxes, 0 = K.

    Shells 0 to this one have tables; for C, N and O only K does.
    """
    if z == 11:
        return 1
    if z == 12:
        return 2
    if 12 < z <= 20:
        return 3
    if 25 <= z <= 30:
        return 6
    if 33 <= z <= 34:
        return 8
    return 0


def _transitions(name: str, auger: bool) -> dict[tuple[int, int], Transitions]:
    rows: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in read_table(name):
        rows.setdefault((int(row["z"]), int(row["shell"])), []).append(row)
    tables = {}
    for key, entries in rows.items():

        def column(field: str, entries: list[dict[str, str]] = entries) -> np.ndarray:
            values = np.zeros(_SLOTS)
            values[: len(entries)] = [float(e[field]) for e in entries]
            return values

        probability = column("probability")
        # The Java's running sum, left to right, as np.cumsum adds.
        cumulative = np.cumsum(probability)
        total = float(cumulative[len(entries) - 1])
        cumulative[len(entries) :] = 0.0
        cumulative = cumulative * (1 / total)
        tables[key] = Transitions(
            linewidth_ev=column("linewidth_ev"),
            probability=probability,
            cumulative=cumulative,
            energy_kev=column("energy_kev"),
            exit_shell=column("exit_shell").astype(np.int64) if auger else None,
            drop_shell=column("drop_shell").astype(np.int64),
            total=total,
            rows=len(entries),
        )
    return tables


@dataclass(frozen=True)
class PackedTransitions:
    """One kind of transition's tables, every element's at once.

    The arrays of :class:`Transitions`, indexed by atomic number, vacancy
    and slot: ``(Z + 1, 9, 65)`` up to the heaviest element tabulated, zero
    for an element or vacancy with no table. A vacancy's tables are a
    gather, as an array library on a GPU wants them, rather than a lookup
    in a dict for each vacancy.

    Attributes
    ----------
    cumulative, energy_kev, linewidth_ev : numpy.ndarray
        As in :class:`Transitions`.
    exit_shell, drop_shell : numpy.ndarray
        Integers; ``exit_shell`` is zero for fluorescence, which has none.
    """

    cumulative: np.ndarray
    energy_kev: np.ndarray
    linewidth_ev: np.ndarray
    exit_shell: np.ndarray
    drop_shell: np.ndarray


def _pack(tables: dict[tuple[int, int], Transitions]) -> PackedTransitions:
    shape = (max(z for z, _ in tables) + 1, max(s for _, s in tables) + 1, _SLOTS)
    packed = {
        name: np.zeros(shape, dtype=np.int64 if "shell" in name else float)
        for name in PackedTransitions.__dataclass_fields__
    }
    for (z, vacancy), table in tables.items():
        for name, array in packed.items():
            value = getattr(table, name)
            if value is not None:
                array[z, vacancy] = value
    for array in packed.values():
        array.setflags(write=False)
    return PackedTransitions(**packed)


@cache
def auger_transitions() -> dict[tuple[int, int], Transitions]:
    """Auger transitions, keyed by (atomic number, vacancy)."""
    return _transitions("auger_transitions.tsv", auger=True)


@cache
def fluorescence_transitions() -> dict[tuple[int, int], Transitions]:
    """Fluorescence transitions, keyed by (atomic number, vacancy)."""
    return _transitions("fluorescence_transitions.tsv", auger=False)


@cache
def packed_auger() -> PackedTransitions:
    """:func:`auger_transitions`, packed."""
    return _pack(auger_transitions())


@cache
def packed_fluorescence() -> PackedTransitions:
    """:func:`fluorescence_transitions`, packed."""
    return _pack(fluorescence_transitions())


def _overvoltage(name: str, width: int) -> np.ndarray:
    table = np.zeros((95, _OVERVOLTAGE_SHELLS, width))
    for row in read_table(name):
        values = [v for k, v in row.items() if k not in ("z", "shell", "label")]
        table[int(row["z"]), int(row["shell"])] = [float(v) for v in values]
    return table


@cache
def overvoltage_low() -> np.ndarray:
    """``lowOvervoltages[Z][shell]``: the binding energy in eV, then a1-a5."""
    return _overvoltage("overvoltage_low.tsv", 6)


@cache
def overvoltage_high() -> np.ndarray:
    """``highOvervoltages[Z][shell]``: b-, b+, A and g1-g4."""
    return _overvoltage("overvoltage_high.tsv", 7)
