"""The GOS model's inelastic collisions, for ``Subprogram GOS`` and XFEL.

What ``trackPhotoelectron`` asks of ``CoefCalcCompute`` and of itself at an
inelastic collision, for many electrons at once:

- **How often.** :func:`~pyraddose_3d.backend.microed.inelastic_shells` gives
  each subshell's inverse mean free path, split into distant longitudinal,
  distant transverse and close collisions. The Java recomputes it, about 10^5
  operations, whenever an electron's energy changes; here it is computed once
  per material on a grid of energies (:class:`ShellTable`) and interpolated,
  exact at the grid points. Inner shells are taken instead from Bethe's
  cross-sections with the overvoltage tables (:func:`bethe_shells`), exact.
- **Which shell, what kind, how much energy.** :func:`energy_loss_distant`,
  :func:`recoil_distant` and :func:`sample_k`, as the Java draws them.
- **Which way.** The primary's deflection and the secondary's direction, from
  the energy and momentum transferred.

Energies are in keV unless a name says eV; ``Q``, the recoil energy, is in
joules, as the Java keeps it.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.microed import distant_integrals, inelastic_shells
from pyraddose_3d.backend.montecarlo import tables

if TYPE_CHECKING:
    from pyraddose_3d.backend.microed import ElectronMaterial

__all__ = [
    "ShellTable",
    "bethe_shells",
    "energy_loss_distant",
    "primary_angle_close",
    "primary_angle_distant",
    "qak",
    "recoil_distant",
    "sample_k",
    "secondary_angle_close",
    "secondary_angle_distant",
    "shell_table",
    "wak",
]

_ELECTRON_MASS = 9.10938356e-31
_SPEED_OF_LIGHT = 299792458.0
_MC2 = _ELECTRON_MASS * _SPEED_OF_LIGHT * _SPEED_OF_LIGHT
#: The Bohr radius, nm.
_BOHR = 5.2917721067e-2
#: ``MC.Wcc``: zero.
_WCC = 0.0
#: ``samplek`` gives up after this many rejections, keeping the last k.
_SAMPLE_K_TRIES = 10000
#: Grid points per factor of e in energy, besides each subshell's edges.
_PER_E_FOLD = 400
#: How many of each distant integral's first steps the grid brackets.
_STEPS = 100
#: Energies whose distant integrals are computed together: a thousand-point
#: walk for each of some 25 subshells, so about 13 MB an array at 64.
_ENERGIES_AT_ONCE = 64


@dataclass(frozen=True)
class ShellTable:
    """:func:`~pyraddose_3d.backend.microed.inelastic_shells` on a grid.

    Attributes
    ----------
    energies : numpy.ndarray
        The grid, keV, ascending: geometric, with every subshell's binding
        energy and the energy where its resonance changes form added.
    values : numpy.ndarray
        ``(energies, elements, 10, 4)``, nm^-1: ``SUBSHELL_ROWS`` rows.
    """

    energies: np.ndarray
    values: np.ndarray

    def _bracket(self, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each energy's grid points either side, and how far it lies between."""
        energy = np.asarray(energy, dtype=float)
        upper = np.clip(
            np.searchsorted(self.energies, energy, side="right"),
            1,
            len(self.energies) - 1,
        )
        lower = upper - 1
        e0, e1 = self.energies[lower], self.energies[upper]
        return lower, upper, (energy - e0) / (e1 - e0)

    def at(self, energy: np.ndarray) -> np.ndarray:
        """Linearly between grid points: ``(len(energy), elements, rows, 4)``."""
        lower, upper, weight = self._bracket(energy)
        return np.asarray(
            self.values[lower]
            + weight[:, None, None, None] * (self.values[upper] - self.values[lower])
        )

    def rows(
        self, energy: np.ndarray, element: np.ndarray, row: np.ndarray
    ) -> np.ndarray:
        """One row per energy, ``(len(energy), 4)``: :meth:`at`'s, read alone."""
        lower, upper, weight = self._bracket(energy)
        below = self.values[lower, element, row]
        return np.asarray(
            below + weight[:, None] * (self.values[upper, element, row] - below)
        )

    def column(self, energy: np.ndarray, row: np.ndarray, column: int) -> np.ndarray:
        """One column of each element's row ``row[element]``, at each energy.

        ``(len(energy), elements)``: :meth:`at`'s, read alone.
        """
        lower, upper, weight = self._bracket(energy)
        element = np.arange(len(row))
        below = self.values[lower[:, None], element, row, column]
        above = self.values[upper[:, None], element, row, column]
        return np.asarray(below + weight[:, None] * (above - below))


def _edges(material: ElectronMaterial) -> list[float]:
    """Where a subshell's terms switch on or change form, keV."""
    from pyraddose_3d.backend.microed import _wk

    edges = []
    for atom in material.atoms:
        _, inner = atom.valence
        for index in range(inner + 1):
            uk = atom.subshell_binding(index) * 1000
            wk = _wk(material, material.sternheimer_a, atom, index)
            edges.extend([uk / 1000, (3 * wk - 2 * uk) / 1000])
    return edges


def _steps(material: ElectronMaterial) -> list[float]:
    """Where the distant integral gains its first trapezia, keV.

    ``integrateDist`` walks from the binding energy Uk to W_dis in steps of
    W_dis / 1000, and its sum steps up as each new point fits. Just above
    the edge W_dis is E itself, so the k-th point fits from E = Uk / (1 -
    k / 1000). The first steps are the large ones: the terms switch on, the
    longitudinal one below zero, within 0.2% of the edge.
    """
    steps: list[float] = []
    for atom in material.atoms:
        _, inner = atom.valence
        for index in range(inner + 1):
            uk = atom.subshell_binding(index)
            steps.extend(uk / (1 - k / 1000) for k in range(1, _STEPS + 1))
    return steps


@functools.lru_cache(maxsize=4)
def shell_table(
    material: ElectronMaterial, lowest: float, highest: float
) -> ShellTable:
    """Tabulate the GOS model's subshells between two energies, keV.

    Exact at the grid points. Between them, measured on insulin from 0.05
    to 12 keV: the inverse mean free path within 5 x 10^-5, and 99% of
    subshells' values within 3 x 10^-4. The rest lie within 0.2% above an
    edge, where the Java's trapezium rule for the distant terms moves in
    steps no grid follows; they are within 4%.

    Kept for the next run of the same material, as building it takes a
    couple of seconds; its arrays are read-only, so that no run can change
    the next one's.
    """
    count = max(2, math.ceil(math.log(highest / lowest) * _PER_E_FOLD) + 1)
    grid = set(np.geomspace(lowest, highest, count).tolist())
    for edge in _edges(material) + _steps(material):
        if lowest < edge < highest:
            # Either side of it, so no interval straddles the change.
            grid.update((edge * (1 - 1e-9), edge * (1 + 1e-9)))
    energies = np.array(sorted(grid))
    # The distant integrals, most of the work, a batch of energies at a time.
    integrals = np.concatenate(
        [
            distant_integrals(material, energies[i : i + _ENERGIES_AT_ONCE])
            for i in range(0, len(energies), _ENERGIES_AT_ONCE)
        ]
    )
    values = np.stack(
        [
            inelastic_shells(material, float(e), row)
            for e, row in zip(energies, integrals, strict=True)
        ]
    )
    energies.setflags(write=False)
    values.setflags(write=False)
    return ShellTable(energies, values)


def _shell_count(z: int) -> int:
    """``getNumberOfShells``: the shells Bethe's cross-sections cover."""
    if z <= 10:
        return 1
    if z == 11:
        return 2
    if 12 <= z <= 19:
        return 4
    if 20 <= z <= 22:
        return 5
    if z == 23:
        return 6
    if 24 <= z <= 32:
        return 7
    return 9


def bethe_shells(material: ElectronMaterial, energy: np.ndarray) -> np.ndarray:
    """``betheIonisationxSection``: each inner shell's inverse mean free path.

    Gryzinski-like at overvoltages under 16, the plane-wave Born
    approximation above, with the overvoltage tables' coefficients; zero
    below a shell's binding energy, and zero for hydrogen.

    Returns
    -------
    numpy.ndarray
        ``(len(energy), elements, 9)``, nm^-1: the Java's ``shellXSections``,
        each shell's cross-section times the atoms per nm^3.
    """
    energy = np.asarray(energy, dtype=float)
    low, high = tables.overvoltage_low(), tables.overvoltage_high()
    vo = energy * KEV_TO_JOULES
    beta_squared = 1 - np.power(_MC2 / (vo + _MC2), 2)
    x = np.power(vo * (vo + 2 * _MC2), 0.5) / _MC2
    result = np.zeros((len(energy), len(material.atoms), 9))
    for column, atom in enumerate(material.atoms):
        if atom.z == 1:
            continue
        per_volume = atom.count / (material.cell_volume / 1000)
        for i in range(_shell_count(atom.z)):
            lo, hi = low[atom.z, i], high[atom.z, i]
            binding = lo[0] / 1000
            u = energy / binding
            with np.errstate(divide="ignore", invalid="ignore"):
                below_16 = (
                    4
                    * math.pi
                    * math.pow(_BOHR, 2)
                    * ((u - 1) / np.power(u, 2))
                    * np.power(
                        lo[1]
                        + u * lo[2]
                        + lo[3] / (1 + u)
                        + lo[4] / np.power(1 + u, 3)
                        + lo[5] / np.power(1 + u, 5),
                        2,
                    )
                )
                pwba = (
                    4
                    * math.pi
                    * math.pow(_BOHR, 2)
                    * (hi[2] / beta_squared)
                    * (
                        (np.log(np.power(x, 2)) - beta_squared) * (1 + (1 / x) * hi[3])
                        + hi[4]
                        + hi[5] * np.power(1 - beta_squared, 0.25)
                        + hi[6] * (1 / x)
                    )
                )
                above_16 = pwba * (u / (u + hi[0]))
            sigma = np.where(u < 16, below_16, above_16)
            result[:, column, i] = np.where(u >= 1, sigma * per_volume, 0.0)
    return result


def wak(energy: np.ndarray, wk: np.ndarray, uk: np.ndarray) -> np.ndarray:
    """``WkToWak``, eV."""
    return np.where(energy * 1000 > 3 * wk - 2 * uk, wk, (energy * 1000 + 2 * uk) / 3)


def qak(energy: np.ndarray, wk: np.ndarray, uk: np.ndarray) -> np.ndarray:
    """``getQak``, eV."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            energy * 1000 > 3 * wk - 2 * uk,
            uk,
            uk * (energy * 1000 / (3 * wk - 2 * uk)),
        )


def energy_loss_distant(
    rng: np.random.Generator, w_dis: np.ndarray, uk: np.ndarray
) -> np.ndarray:
    """``getEnergyLossDistant``: W from the triangle on [Uk, W_dis], eV."""
    return np.asarray(
        w_dis - np.power(rng.random(len(w_dis)) * np.power(w_dis - uk, 2), 0.5)
    )


def _q_minus(energy: np.ndarray, wak_ev: np.ndarray) -> np.ndarray:
    """``getQminusModified``, J."""
    e = energy * KEV_TO_JOULES
    w = (wak_ev / 1000) * KEV_TO_JOULES
    with np.errstate(invalid="ignore"):
        value = (
            np.power(
                np.power(
                    np.power(e * (e + 2 * _MC2), 0.5)
                    - np.power((e - w) * (e - w + 2 * _MC2), 0.5),
                    2,
                )
                + math.pow(_MC2, 2),
                0.5,
            )
            - _MC2
        )
    return np.where(w > e, 0.0, value)


def recoil_distant(
    rng: np.random.Generator, energy: np.ndarray, wak_ev: np.ndarray, qak_ev: np.ndarray
) -> np.ndarray:
    """``getRecoilEnergyDistant``: Q, J; zero where Qak is."""
    q_minus = _q_minus(energy, wak_ev)
    q_s = q_minus / (1 + q_minus / (2 * _MC2))
    q_ak = (qak_ev / 1000) * KEV_TO_JOULES
    u = rng.random(len(energy))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (
            q_s
            * 1
            / (np.power((q_s / q_ak) * (1 + q_ak / (2 * _MC2)), u) - (q_s / (2 * _MC2)))
        )
    return np.asarray(np.where(qak_ev == 0, 0.0, q))


def _close_a(energy: np.ndarray) -> np.ndarray:
    """``getClosea``."""
    vo = energy * KEV_TO_JOULES
    return np.asarray(np.power(vo / (vo + _MC2), 2))


def sample_k(
    rng: np.random.Generator, energy: np.ndarray, qk_ev: np.ndarray
) -> np.ndarray:
    """``samplek``: the fraction of E + Uk a close collision transfers.

    By rejection, as the Java draws it; after 10,000 rejections the last k is
    kept, as the Java keeps it.
    """
    energy = np.asarray(energy, dtype=float)
    a = _close_a(energy)
    kc = np.maximum(qk_ev, _WCC) / (energy * 1000 + qk_ev)
    k = np.zeros(len(energy))
    # With Qk zero every trial is k = 0, whose density is infinite: always
    # rejected, so the Java keeps k = 0 after 10,000 tries.
    pending = np.flatnonzero(kc > 0)
    for _ in range(_SAMPLE_K_TRIES):
        if not len(pending):
            break
        ap, kcp = a[pending], kc[pending]
        zeta = rng.random(len(pending)) * (1.0 + 5.0 * ap * kcp / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            trial = np.where(
                zeta < 1,
                kcp / (1 - zeta * (1 - 2 * kcp)),
                kcp + (zeta - 1) * (1 - 2 * kcp) / (5 * ap * kcp),
            )
            pdf = (
                np.power(trial, -2.0)
                + np.power(1 - trial, -2.0)
                - 1 / (trial * (1 - trial))
                + ap * (1 + 1 / (trial * (1 - trial)))
            ) * ((trial - kcp >= 0) & (0.5 - trial >= 0))
            accept = rng.random(len(pending)) * (1 + 5 * ap * np.power(trial, 2)) < (
                np.power(trial, 2) * pdf
            )
        k[pending] = trial
        pending = pending[~accept]
    return k


def primary_angle_distant(
    energy: np.ndarray, q: np.ndarray, wak_ev: np.ndarray
) -> np.ndarray:
    """``getGOSPrimaryScatterLong``: the primary's deflection, radians."""
    e = energy * KEV_TO_JOULES
    w = (wak_ev / 1000) * KEV_TO_JOULES
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = (
            e * (e + 2 * _MC2) + (e - w) * (e - w + 2 * _MC2) - q * (q + 2 * _MC2)
        )
        denominator = 2 * np.power(
            e * (e + 2 * _MC2) * (e - w) * (e - w + 2 * _MC2), 0.5
        )
        return np.asarray(np.arccos(numerator / denominator))


def primary_angle_close(energy: np.ndarray, w_kev: np.ndarray) -> np.ndarray:
    """``getGOSPrimaryScatterClose``, radians."""
    e = energy * KEV_TO_JOULES
    w = w_kev * KEV_TO_JOULES
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(
            np.arccos(
                np.power(((e - w) / e) * ((e + 2 * _MC2) / (e - w + 2 * _MC2)), 0.5)
            )
        )


def secondary_angle_close(energy: np.ndarray, w_kev: np.ndarray) -> np.ndarray:
    """``secondaryScatterClose``: the secondary's angle to the primary."""
    w = w_kev * KEV_TO_JOULES
    e = energy * KEV_TO_JOULES
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(
            np.arccos(np.power((w / e) * ((e + 2 * _MC2) / (w + 2 * _MC2)), 0.5))
        )


def secondary_angle_distant(
    energy: np.ndarray, wak_ev: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """``secondaryScatterDistant``: the secondary's angle to the primary."""
    vo = energy * KEV_TO_JOULES
    beta_squared = 1 - np.power(_MC2 / (vo + _MC2), 2)
    w = (wak_ev / 1000) * KEV_TO_JOULES
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.power(
            ((np.power(w, 2) / beta_squared) / (q * (q + 2 * _MC2)))
            * np.power(
                1 + (q * (q + 2 * _MC2) - np.power(w, 2)) / (2 * w * (vo + _MC2)), 2
            ),
            0.5,
        )
        return np.asarray(np.arccos(cos_theta))
