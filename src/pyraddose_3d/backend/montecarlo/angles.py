"""ELSEPA's electron angular tables, from their two data packages.

RADDOSE-3D's Monte Carlo and XFEL subprograms draw each elastic deflection
from a table per element: the differential cross-section at each tabulated
energy, angle by angle. ``getAngleFileData`` reads ``below_20000/{Z}.csv``
for electrons up to 20 keV and ``above_20000/{Z}.csv`` above. At 100 MB
they are too much for the core package, so they come in two of their own,
installed with the ``montecarlo`` extra:

- ``pyraddose-3d-elsepa-low``, every element to 20 keV;
- ``pyraddose-3d-elsepa-high``, 21 to 300 keV, for Z = 1 to 52.

Each table is read when first asked for, as the Java reads it, and kept.
"""

from __future__ import annotations

import lzma
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np

__all__ = [
    "HIGH_ENERGY_KEV",
    "MAX_Z",
    "AngularTable",
    "MissingTablesError",
    "angular_table",
    "deflection_angles",
]

#: Above this electron energy, in keV, the high-energy tables are used.
HIGH_ENERGY_KEV = 20.0

#: high_energy -> the last element with a table: up to 20 keV plutonium,
#: above it tellurium.
MAX_Z = {False: 94, True: 52}

#: high_energy -> (distribution, import name).
_PACKAGES = {
    False: ("pyraddose-3d-elsepa-low", "pyraddose_3d_elsepa_low"),
    True: ("pyraddose-3d-elsepa-high", "pyraddose_3d_elsepa_high"),
}


class MissingTablesError(ImportError):
    """A table's package is not installed, or not built."""


@dataclass(frozen=True)
class AngularTable:
    """One element's elastic scattering, at each tabulated energy.

    Attributes
    ----------
    energies_kev : numpy.ndarray
        The tabulated electron energies, keV, ascending and distinct: the
        keys of the Java's ``TreeMap``.
    probabilities : numpy.ndarray
        One row per energy: the differential cross-section at each angle,
        which the Java sums to a cumulative distribution.
    angles_deg : numpy.ndarray
        The angles the file's header gives each column, in degrees. Below
        20 keV the Java uses the column index, the same. Above it, the Java
        works the angle out from the index by its own formula.
    """

    energies_kev: np.ndarray
    probabilities: np.ndarray
    angles_deg: np.ndarray


@cache
def angular_table(z: int, high_energy: bool) -> AngularTable:
    """The table for element ``z``, above 20 keV or up to it.

    Parameters
    ----------
    z : int
        Atomic number.
    high_energy : bool
        Whether the electron is above 20 keV.

    Returns
    -------
    AngularTable
        As ``getAngleFileData`` builds it: rows in energy order, and where an
        energy appears twice, the later row. Only hydrogen's high-energy
        table repeats any, 135 of them, each the same as before.

    Raises
    ------
    MissingTablesError
        If the package holding the table is not installed or not built.
    ValueError
        If ELSEPA has no table for this element.
    """
    distribution, name = _PACKAGES[high_energy]
    if not 1 <= z <= MAX_Z[high_energy]:
        raise ValueError(
            f"ELSEPA's tables {'above' if high_energy else 'up to'} 20 keV run "
            f"from Z = 1 to {MAX_Z[high_energy]}; there is none for Z = {z}"
        )
    try:
        tables = resources.files(name) / "tables"
    except ModuleNotFoundError:
        raise MissingTablesError(
            f"the Monte Carlo and XFEL subprograms need ELSEPA's electron "
            f"angular tables, in the package {distribution}; install "
            f"pyRADDOSE-3D with its montecarlo extra"
        ) from None
    path = tables / f"{z}.csv.xz"
    if not path.is_file():
        raise MissingTablesError(
            f"{distribution} is installed without its tables; build them from "
            f"a RADDOSE-3D checkout with pyRADDOSE-3D's "
            f"tests/backend/data/build_elsepa_packages.py"
        )

    header, *lines = lzma.decompress(path.read_bytes()).decode("ascii").splitlines()
    rows: dict[float, list[float]] = {}
    for line in lines:
        fields = line.split(",")
        # A TreeMap: a repeated energy replaces the earlier row.
        rows[float(fields[0])] = [float(v) for v in fields[1:]]
    energies = sorted(rows)
    return AngularTable(
        energies_kev=np.array(energies),
        probabilities=np.array([rows[e] for e in energies]),
        angles_deg=np.array([float(v) for v in header.split(",")[1:]]),
    )


def column_weights(degrees: np.ndarray) -> np.ndarray:
    """Each column's share of the solid angle, up to a constant.

    The tables give the differential cross-section per unit solid angle at
    each column's angle, so the chance of a deflection near theta is that
    value times sin theta times the column's span of angles: half the span
    to its neighbours. RADDOSE-3D took the values themselves as the chances
    until #46, which made small deflections far too likely.
    """
    radians = np.radians(degrees)
    before = np.concatenate([radians[:1], radians[:-1]])
    after = np.concatenate([radians[1:], radians[-1:]])
    return np.asarray(np.sin(radians) * (after - before) / 2)


@dataclass(frozen=True)
class _Sampler:
    """One table, ready to sample: each row's running probabilities."""

    energies_kev: np.ndarray
    cumulative: np.ndarray
    angles: np.ndarray


@cache
def _sampler(z: int, high_energy: bool) -> _Sampler:
    table = angular_table(z, high_energy)
    weighted = table.probabilities * column_weights(table.angles_deg)[None, :]
    running = np.cumsum(weighted, axis=1)
    return _Sampler(
        energies_kev=table.energies_kev,
        cumulative=running / running[:, -1:],
        angles=np.radians(table.angles_deg),
    )


def _nearest_rows(energies: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """``returnNearestEnergy``: TreeMap floor or ceiling, the floor on a tie."""
    below = np.searchsorted(energies, energy, side="right") - 1
    above = np.searchsorted(energies, energy, side="left")
    below = np.where(below < 0, above, below)
    above = np.where(above >= len(energies), below, above)
    nearer_below = np.abs(energy - energies[below]) <= np.abs(energy - energies[above])
    return np.where(nearer_below, below, above)


def deflection_angles(
    z: np.ndarray, energy: np.ndarray, uniform: np.ndarray
) -> np.ndarray:
    """``getPrimaryElasticScatteringAngle`` for many electrons, in radians.

    Parameters
    ----------
    z : numpy.ndarray
        The atomic number of the element each electron scattered from.
    energy : numpy.ndarray
        Each electron's energy, keV, from 0.05 to 300.
    uniform : numpy.ndarray
        A uniform random number in [0, 1) for each.

    Returns
    -------
    numpy.ndarray
        The polar deflection: the first of the table's angles whose running
        chance, at the tabulated energy nearest the electron's, reaches the
        random number.
    """
    angles = np.empty_like(energy)
    high = energy > HIGH_ENERGY_KEV
    for key in np.unique(np.stack([z, high.astype(np.int64)]), axis=1).T:
        element, high_energy = int(key[0]), bool(key[1])
        chosen = np.flatnonzero((z == element) & (high == high_energy))
        sampler = _sampler(element, high_energy)
        rows = _nearest_rows(sampler.energies_kev, energy[chosen])
        column = _first_reaching(sampler.cumulative, rows, uniform[chosen])
        angles[chosen] = sampler.angles[column]
    return angles


def _first_reaching(
    cumulative: np.ndarray, rows: np.ndarray, uniform: np.ndarray
) -> np.ndarray:
    """In each chosen row, the first column whose value is at least ``uniform``.

    A binary search in every row at once: about ten passes for 606 columns.
    """
    low = np.zeros(len(rows), dtype=np.int64)
    high = np.full(len(rows), cumulative.shape[1] - 1, dtype=np.int64)
    while True:
        open_ = low < high
        if not open_.any():
            return low
        middle = (low + high) // 2
        reaches = cumulative[rows, middle] >= uniform
        high = np.where(open_ & reaches, middle, high)
        low = np.where(open_ & ~reaches, middle + 1, low)
