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

__all__ = ["MAX_Z", "AngularTable", "MissingTablesError", "angular_table"]

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
