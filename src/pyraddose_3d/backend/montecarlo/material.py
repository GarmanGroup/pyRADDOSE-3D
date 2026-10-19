"""Electron quantities for the Monte Carlo, for many electrons at once.

At every step ``trackPhotoelectron`` asks ``CoefCalcCompute`` for the
stopping power, the elastic mean free path and which element scattered the
electron. MicroED asks the same of one energy at a time, in
:mod:`pyraddose_3d.backend.microed`; these answer for an array of energies,
from the same tables and in the same order of operations, so they agree
with MicroED's to the last bit or two.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Atoms:
    """A material's atoms' constants, one row per atom, in its order.

    Attributes
    ----------
    excitation, k : numpy.ndarray
        The mean excitation energy, eV, in the condensed phase, and the
        constant of its correction at low energy.
    ln_i_weight : numpy.ndarray
        Each atom's weight in the mean of ln I: its mass fraction times Z/A.
    mean_z, z_per_a : float
        The mass-weighted mean Z, and the sum of Z over the sum of A.
    formula_scale, screening : numpy.ndarray
        Langmore and Smith's 1.4e-6 Z^1.5 and 0.26 Z.
    elsepa_energies, elsepa : numpy.ndarray
        ELSEPA's energies, keV, which every element shares, and each atom's
        cross-sections at them, nm^2.
    elastic_fraction, elastic_weight : numpy.ndarray
        The mass fraction and atomic weight by ELSEPA's own atomic weights.
    per_volume : numpy.ndarray
        Atoms per nm^3.
    """

    excitation: np.ndarray
    k: np.ndarray
    ln_i_weight: np.ndarray
    mean_z: float
    z_per_a: float
    formula_scale: np.ndarray
    screening: np.ndarray
    elsepa_energies: np.ndarray
    elsepa: np.ndarray
    elastic_fraction: np.ndarray
    elastic_weight: np.ndarray
    per_volume: np.ndarray


@functools.lru_cache(maxsize=8)
def _atoms(material: ElectronMaterial) -> _Atoms:
    """Gathered once per material, each as the loops over atoms had it."""
    excitation, k, ln_i_weight = [], [], []
    mean_z = sum_a = sum_z = 0.0
    for atom in material.atoms:
        a = atom.atomic_weight
        z = atom.z
        fraction = (atom.count * a) / material.molecular_weight
        sum_z += z * atom.count
        sum_a += a * atom.count
        j = atom.shells.mean_excitation
        if z not in _CONDENSED_PHASE:
            j *= 1.13
        excitation.append(j)
        k.append(0.7344 * math.pow(z, 0.0367))
        ln_i_weight.append(fraction * (z / a))
        mean_z += fraction * z
    grids = {atom.elastic.energies for atom in material.atoms}
    if len(grids) != 1:
        raise ValueError("ELSEPA's cross-sections should share one grid of energies")
    elastic_weight = [atom.elastic.atomic_weight for atom in material.atoms]
    return _Atoms(
        excitation=np.array(excitation),
        k=np.array(k),
        ln_i_weight=np.array(ln_i_weight),
        mean_z=mean_z,
        z_per_a=sum_z / sum_a,
        formula_scale=np.array(
            [1.4e-6 * math.pow(float(atom.z), 1.5) for atom in material.atoms]
        ),
        screening=np.array([0.26 * float(atom.z) for atom in material.atoms]),
        elsepa_energies=np.asarray(grids.pop()),
        elsepa=np.array([atom.elastic.cross_sections for atom in material.atoms]),
        elastic_fraction=np.array(
            [
                (atom.count * a) / material.molecular_weight
                for atom, a in zip(material.atoms, elastic_weight, strict=True)
            ]
        ),
        elastic_weight=np.array(elastic_weight),
        per_volume=np.array(
            [atom.count / (material.cell_volume / 1000) for atom in material.atoms]
        ),
    )


def _in_turn(terms: np.ndarray) -> np.ndarray:
    """The sum over atoms, the first axis, in turn as a loop adds, not pairwise."""
    return np.asarray(np.add.accumulate(terms, axis=0)[-1])


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

    atoms = _atoms(material)
    j, k = atoms.excitation[:, None], atoms.k[:, None]
    j_star = j / (1 + k * (j / (energy * 1000)))
    mean_ln_i = _in_turn(atoms.ln_i_weight[:, None] * np.log(j_star))
    mean_z = atoms.mean_z

    mean_ln_i = mean_ln_i / atoms.z_per_a
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
        * atoms.z_per_a
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
    atoms = _atoms(material)
    # Infinite for an electron at rest: an Auger electron of an empty
    # transition slot, which stops before it takes a step, as in the Java.
    with np.errstate(divide="ignore", invalid="ignore"):
        formula = (atoms.formula_scale[:, None] / beta_squared) * (
            1 - atoms.screening[:, None] / (137 * np.power(beta_squared, 0.5))
        )
    tabulated = _elsepa_at(atoms.elsepa_energies, atoms.elsepa, energy)
    return np.where(in_range & (tabulated > 0.0), tabulated, formula)


def _elsepa_at(grid: np.ndarray, table: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """``ElementEM.getElasticCoefficient``: TreeMap floor and ceiling, linear.

    ``table`` holds a row of values on ``grid`` for each atom.
    """
    clipped = np.clip(energy, grid[0], grid[-1])
    below = np.searchsorted(grid, clipped, side="right") - 1
    above = np.searchsorted(grid, clipped, side="left")
    before, after = grid[below], grid[above]
    span = after - before
    across = np.where(span != 0, (clipped - before) / np.where(span != 0, span, 1), 0.0)
    low, high = table[:, below], table[:, above]
    return low + ((high - low) * across)


def elastic_mean_free_path(
    material: ElectronMaterial, cross_sections: np.ndarray
) -> np.ndarray:
    """``getElectronElasticMFPL``, in nm, from :func:`elastic_cross_sections`."""
    atoms = _atoms(material)
    part = _in_turn(
        (atoms.elastic_fraction[:, None] * cross_sections)
        / atoms.elastic_weight[:, None]
    )
    return 1 / (_AVOGADRO * (material.density / 1e21) * part)


def elastic_element_cdf(
    material: ElectronMaterial, cross_sections: np.ndarray
) -> np.ndarray:
    """``getElasticProbs``: which element scatters, as running fractions.

    ``(elements, energies)``, in atomic-number order: each element's
    cross-section times its atoms per nm^3, over their sum.
    """
    weighted = cross_sections * _atoms(material).per_volume[:, None]
    total = _in_turn(weighted)
    return np.cumsum(weighted / total, axis=0)
