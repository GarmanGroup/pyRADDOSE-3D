"""Electron quantities for the Monte Carlo, for many electrons at once.

At every step ``trackPhotoelectron`` asks ``CoefCalcCompute`` for the
stopping power, the elastic mean free path and which element scattered the
electron. MicroED asks the same of one energy at a time, in
:mod:`pyraddose_3d.backend.microed`; these answer for an array of energies,
from the same tables and in the same order of operations, so they agree
with MicroED's to the last bit or two.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES

if TYPE_CHECKING:
    from pyraddose_3d.backend.microed import ElectronMaterial

__all__ = [
    "elastic_cross_sections",
    "elastic_element_cdf",
    "elastic_mean_free_path",
    "stopping_power",
]

_ELECTRON_MASS = 9.10938356e-31
_SPEED_OF_LIGHT = 299792458.0
_AVOGADRO = 6.022e23
#: Elements whose mean excitation energy the table gives for the condensed
#: phase already; the rest are multiplied by 1.13.
_CONDENSED_PHASE = frozenset({1, 6, 7, 8, 9, 17})
#: ELSEPA's cross-sections cover 0.05 to 300 keV.
_ELSEPA_RANGE = (0.05, 300.0)


def stopping_power(material: ElectronMaterial, energy: np.ndarray) -> np.ndarray:
    """``calcStoppingPower``, in keV/nm, for each energy in keV.

    As :func:`pyraddose_3d.backend.microed.stopping_power`, term by term.
    """
    m = _ELECTRON_MASS
    csquared = _SPEED_OF_LIGHT * _SPEED_OF_LIGHT
    vo = energy * KEV_TO_JOULES
    beta_squared = 1 - np.power(m * csquared / (vo + m * csquared), 2)
    gamma = 1 / np.power(1 - beta_squared, 0.5)
    kinetic_mev = (((gamma - 1) * m * csquared) / KEV_TO_JOULES) / 1000

    mean_ln_i = np.zeros_like(energy)
    mean_z = 0.0
    sum_a = 0.0
    sum_z = 0.0
    for atom in material.atoms:
        a = atom.atomic_weight
        z = atom.z
        fraction = (atom.count * a) / material.molecular_weight
        sum_z += z * atom.count
        sum_a += a * atom.count
        j = atom.shells.mean_excitation
        if z not in _CONDENSED_PHASE:
            j *= 1.13
        k = 0.7344 * math.pow(z, 0.0367)
        j_star = j / (1 + k * (j / (energy * 1000)))
        mean_z += fraction * z
        mean_ln_i += fraction * (z / a) * np.log(j_star)

    mean_ln_i = mean_ln_i / (sum_z / sum_a)
    mean_j = np.exp(mean_ln_i)
    mean_j = mean_j / (1 + 0.85 * mean_j / (energy * 1000))
    mean_j = (mean_j / 1000) * KEV_TO_JOULES
    e = energy * KEV_TO_JOULES
    f_beta = (
        np.log((m * csquared * e * beta_squared) / (2 * (1 - beta_squared)))
        - (2 * np.power(1 - beta_squared, 0.5) - 1 + beta_squared) * math.log(2)
        + 1
        - beta_squared
        + (1.0 / 8.0) * (1 - np.power(1 - beta_squared, 0.5))
    )
    power = (
        (0.153536 / beta_squared)
        * (sum_z / sum_a)
        * (f_beta - 2 * np.log(mean_j) - 0.1832)
    )
    power = power * 1000 * material.density / 1e7
    return power + ((kinetic_mev * mean_z) / 800) * power


def elastic_cross_sections(
    material: ElectronMaterial, energy: np.ndarray
) -> np.ndarray:
    """Each element's elastic cross-section per atom, in nm^2.

    ``(elements, energies)``: ELSEPA's, linear between its energies, from
    0.05 to 300 keV where it has a value; Langmore and Smith's formula
    otherwise.
    """
    m = _ELECTRON_MASS
    csquared = 3e8 * 3e8
    vo = energy * KEV_TO_JOULES
    beta_squared = 1 - np.power(m * csquared / (vo + m * csquared), 2)
    in_range = (energy <= _ELSEPA_RANGE[1]) & (energy >= _ELSEPA_RANGE[0])
    rows = []
    for atom in material.atoms:
        z = float(atom.z)
        formula = ((1.4e-6 * math.pow(z, 1.5)) / beta_squared) * (
            1 - (0.26 * z) / (137 * np.power(beta_squared, 0.5))
        )
        table = atom.elastic
        tabulated = _elsepa_at(table.energies, table.cross_sections, energy)
        rows.append(np.where(in_range & (tabulated > 0.0), tabulated, formula))
    return np.stack(rows)


def _elsepa_at(
    energies: tuple[float, ...], values: tuple[float, ...], energy: np.ndarray
) -> np.ndarray:
    """``ElementEM.getElasticCoefficient``: TreeMap floor and ceiling, linear."""
    grid = np.asarray(energies)
    table = np.asarray(values)
    clipped = np.clip(energy, grid[0], grid[-1])
    below = np.searchsorted(grid, clipped, side="right") - 1
    above = np.searchsorted(grid, clipped, side="left")
    before, after = grid[below], grid[above]
    span = after - before
    across = np.where(span != 0, (clipped - before) / np.where(span != 0, span, 1), 0.0)
    low, high = table[below], table[above]
    return low + ((high - low) * across)


def elastic_mean_free_path(
    material: ElectronMaterial, cross_sections: np.ndarray
) -> np.ndarray:
    """``getElectronElasticMFPL``, in nm, from :func:`elastic_cross_sections`."""
    part = np.zeros(cross_sections.shape[1])
    for atom, sigma in zip(material.atoms, cross_sections, strict=True):
        a = atom.elastic.atomic_weight
        fraction = (atom.count * a) / material.molecular_weight
        part += (fraction * sigma) / a
    return 1 / (_AVOGADRO * (material.density / 1e21) * part)


def elastic_element_cdf(
    material: ElectronMaterial, cross_sections: np.ndarray
) -> np.ndarray:
    """``getElasticProbs``: which element scatters, as running fractions.

    ``(elements, energies)``, in atomic-number order: each element's
    cross-section times its atoms per nm^3, over their sum.
    """
    per_volume = np.array(
        [atom.count / (material.cell_volume / 1000) for atom in material.atoms]
    )
    weighted = cross_sections * per_volume[:, None]
    total = np.zeros(cross_sections.shape[1])
    for row in weighted:
        total += row
    return np.cumsum(weighted / total, axis=0)
