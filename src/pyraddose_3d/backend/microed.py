"""RADDOSE-3D's MicroED engine: ``Subprogram EMED``, or ``EMSP``.

Ports ``MicroED.CalculateEM`` and the electron physics it calls in
``CoefCalcCompute``. Both keywords run the same calculation, and it is
deterministic: no random numbers, no Monte Carlo. RADDOSE-3D's MicroED Monte
Carlo, and its lookup of NIST's ESTAR stopping powers through a web browser,
are both commented out in the Java and so are not here either.

What it computes, for a thin crystal in an electron beam:

- **The dose**, by stopping power. The crystal is cut along the beam into as
  many 5 nm slices as fit, at least one, which share its thickness equally.
  Each slice takes the stopping power at the energy its electrons enter with,
  and the dose reported is the mean over the slices.
- **How many electrons scatter**: elastically, exactly once elastically, and
  inelastically, from the elastic mean free path (ELSEPA cross-sections) and
  the inelastic one (a generalised oscillator strength, GOS, model). An
  electron scattered elastically once and never inelastically is
  *productive*: it carries diffraction information.
- **The information coefficient**, productive electrons per MGy, and a grid
  search for the accelerating voltage and the thickness that would maximise
  it.

Every formula is transcribed in the Java's evaluation order, constants
included -- some methods take the speed of light as 3e8 and others as
299792458. The sums over elements run in atomic-number order, where the Java
iterates a ``HashSet``, so the two can differ in the last bit; nothing else
should differ.

Binding energies
----------------
The GOS model reads every subshell's binding energy up to the valence one.
RADDOSE-3D's tables used to give zero for one of them for Na, Ca, Sc, Ti, V,
Cr, Ga and Ge, and nothing for aluminium's M1. They now come from the X-ray
Data Booklet, Table 1-1, and, for Al 3s and Cr 3d, which it does not list,
from Carlson's *Photoelectron and Auger Spectroscopy* (1975) as Geant4
tabulates them. Should a table still lack one, RADDOSE-3D reports NaN for
the inelastic mean free path and everything it feeds -- the inelastic and
productive counts, the information coefficient and the best voltage and
thickness -- and warns, and so does this port (:func:`lacking_bindings`).

This follows RADDOSE-3D ``master`` at ``82ac582``, with its MicroED fixes
(#40) and its stopping power's charge summed as a double (#42). Before #40,
the distant-collision integral mixed two shell models, and the electron
physics saw part of the crystal, or none of it, for PDB, sequence and CIF
compositions (``UPSTREAM_FIXES.md``).
"""

from __future__ import annotations

import csv
import dataclasses
import io
import math
import warnings
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES

if TYPE_CHECKING:
    from pyraddose_3d.backend.composition import Composition

__all__ = [
    "SUBSHELL_ROWS",
    "ElectronMaterial",
    "MicroEDResult",
    "csda_range",
    "elastic_mean_free_path",
    "electron_material",
    "inelastic_mean_free_path",
    "inelastic_shells",
    "microed",
    "stopping_power",
]

_DATA_PACKAGE = "pyraddose_3d.data"

#: Electron rest mass in kg, as most of the Java's methods spell it.
_ELECTRON_MASS = 9.10938356e-31
_SPEED_OF_LIGHT = 299792458.0
#: ``CoefCalcCompute.AVOGADRO_NUM``.
_AVOGADRO = 6.022e23
#: The electron's charge in esu, for the GOS model's cross-sections.
_ELEMENTARY_CHARGE_ESU = 4.80320425e-10
_PLANCK = 6.62607004e-34

#: ``CoefCalcCompute.Wbins``: points in each distant-collision integral.
_W_BINS = 1000
#: ``CoefCalcCompute.fcbCutoff``: a subshell bound more weakly than this
#: joins the conduction band. At zero, none does, so the plasmon term never
#: contributes.
_FCB_CUTOFF = 0.0
#: Electrons in each subshell, K, L1, L2, L3, M1, ..., N1.
_SUBSHELL_ELECTRONS = (2, 2, 2, 4, 2, 2, 4, 10, 2, 30)
#: Elements whose mean excitation energy the table already gives for the
#: condensed phase; every other one is multiplied by 1.13.
_CONDENSED_PHASE = frozenset({1, 6, 7, 8, 9, 17})

#: ``MicroED.sliceThickness``, in nm.
_SLICE_THICKNESS = 5.0
#: ``MicroED.energyArray``: accelerating voltages the coarse search samples,
#: in kV.
_ENERGY_GRID = (
    10.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
    1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0,
)  # fmt: skip
#: ``MicroED.thicknessArray``: thicknesses the coarse search samples, in nm.
_THICKNESS_GRID = (
    10.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0,
    1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
    2000.0,
)  # fmt: skip
#: The fine searches' steps: 10 kV and 5 nm.
_ENERGY_STEP = 10
_THICKNESS_STEP = 5
#: Below this energy, in keV, an electron has stopped.
_STOPPED = 0.05


# --- Tables ---------------------------------------------------------------


def _read_table(name: str, delimiter: str) -> list[list[str]]:
    """The rows of a vendored table, comment lines left out."""
    text = resources.files(_DATA_PACKAGE).joinpath(name).read_text("ascii")
    return [
        row
        for row in csv.reader(io.StringIO(text), delimiter=delimiter)
        if row and not row[0].startswith("#")
    ]


def _keyed(name: str) -> dict[int, dict[str, str]]:
    """A tab-separated table with a header row, keyed by atomic number."""
    header, *rows = _read_table(name, "\t")
    return {int(row[0]): dict(zip(header, row, strict=True)) for row in rows}


def _number(value: str) -> float | None:
    """A table value, with a blank read as absent, as the Java reads null."""
    return None if value == "" else float(value)


@dataclass(frozen=True)
class _ElasticTable:
    """One element's ELSEPA elastic cross-sections (``ElementEM``).

    Attributes
    ----------
    atomic_weight : float
        The table's own atomic weight, which can differ from the element
        table's: 1.01 for hydrogen rather than 1.008.
    energies : tuple[float, ...]
        Electron energies in keV: 0.05 to 1 in steps of 0.05, then 2 to 300 in
        steps of 1, each computed as the Java computes its map keys.
    cross_sections : tuple[float, ...]
        Elastic cross-section per atom at each energy, in nm^2.
    """

    atomic_weight: float
    energies: tuple[float, ...]
    cross_sections: tuple[float, ...]

    def at(self, energy: float) -> float:
        """``ElementEM.getElasticCoefficient``: linear between grid points.

        Zero outside 0.05 to 300 keV, where the caller falls back to the
        Langmore-Smith formula.
        """
        if not 0.05 <= energy <= 300:
            return 0.0
        # TreeMap.floorKey and ceilingKey.
        below = bisect_right(self.energies, energy) - 1
        above = bisect_left(self.energies, energy)
        before, after = self.energies[below], self.energies[above]
        across = 0.0
        if after != before:
            across = (energy - before) / (after - before)
        low, high = self.cross_sections[below], self.cross_sections[above]
        return low + ((high - low) * across)


def _elsepa_energies() -> tuple[float, ...]:
    """``ElementDatabaseEM``'s map keys for columns 3 to 321."""
    energies = []
    for i in range(3, 322):
        if i >= 23:
            energies.append(float(i - 21) * 1.0)
        else:
            energies.append(float(i - 2) * 0.05)
    return tuple(energies)


@cache
def _elsepa() -> dict[int, _ElasticTable]:
    """``constants/fullelsepa.csv``: every element's elastic cross-sections."""
    energies = _elsepa_energies()
    tables = {}
    for row in _read_table("elsepa_cross_sections.csv", ","):
        tables[int(row[0])] = _ElasticTable(
            atomic_weight=float(row[2]),
            energies=energies,
            cross_sections=tuple(float(v) for v in row[3:322]),
        )
    return tables


@dataclass(frozen=True)
class _Shells:
    """What the GOS model knows of one element's electrons.

    Attributes
    ----------
    mean_excitation : float
        Mean excitation energy, I, in eV.
    subshells : tuple[float | None, ...]
        Binding energies in keV, as ``getShellBindingSubshell`` returns them
        for subshells 0 to 9 -- K, L1, L2, L3, M1, M2, M3, M4, M5, N1. L1, M1
        and N1 are binding energies; the others are absorption edges.
    shells : tuple[float | None, ...]
        As ``getShellBinding`` returns them for shells 0 to 2: the K, L1 and
        M1 absorption edges, a missing M1 edge read as zero.
    """

    mean_excitation: float
    subshells: tuple[float | None, ...]
    shells: tuple[float | None, ...]


@cache
def _shell_table() -> dict[int, _Shells]:
    edges = _keyed("absorption_edges.tsv")
    electron = _keyed("electron_ionisation.tsv")
    table = {}
    for z, row in electron.items():
        edge = edges[z]
        table[z] = _Shells(
            mean_excitation=float(row["mean_excitation_energy_ev"]),
            subshells=(
                _number(edge["k_kev"]),
                _number(row["l1_binding_kev"]),
                _number(edge["l2_kev"]),
                _number(edge["l3_kev"]),
                _number(row["m1_binding_kev"]),
                _number(edge["m2_kev"]),
                _number(edge["m3_kev"]),
                _number(edge["m4_kev"]),
                _number(edge["m5_kev"]),
                _number(row["n1_binding_kev"]),
            ),
            shells=(
                _number(edge["k_kev"]),
                _number(edge["l1_kev"]),
                _number(edge["m1_kev"]) or 0.0,
            ),
        )
    return table


# --- The material ---------------------------------------------------------


@dataclass(frozen=True)
class _Atom:
    """One element of the material, with everything the physics reads."""

    symbol: str
    z: int
    atomic_weight: float
    count: float
    shells: _Shells
    elastic: _ElasticTable

    @property
    def valence(self) -> tuple[int, int]:
        """``getNumValenceElectronsSubshells``: (valence electrons, inner subshells).

        Elements from Z = 35 have no inner subshells: all their electrons
        count as valence. RADDOSE-3D's model stops there; reproduced.
        """
        z = self.z
        if 2 < z <= 10:
            return z - 2, 1
        if z == 11:
            return z - 4, 2
        if z == 12:
            return z - 6, 3
        if 13 <= z <= 19:
            return z - 10, 4
        if 20 <= z <= 22:
            return z - 12, 5
        if z == 23:
            return z - 14, 6
        if 24 <= z <= 32:
            return z - 18, 7
        if 33 <= z <= 34:
            return z - 28, 9
        return z, 0

    @property
    def valence_shells(self) -> tuple[int, int]:
        """``getNumValenceElectrons``, the coarser shell model."""
        z = self.z
        if 2 < z <= 10:
            return z - 2, 1
        if 10 < z <= 28:
            return z - 10, 2
        if 28 < z <= 60:
            return z - 28, 3
        if z > 60:
            return z - 60, 4
        return z, 0

    def subshell_binding(self, index: int) -> float:
        """``getShellBindingSubshell``, in keV."""
        if index >= len(self.shells.subshells):
            return 0.0
        value = self.shells.subshells[index]
        if value is None:
            raise ValueError(
                f"{self.symbol} has no binding energy for subshell {index}"
            )
        return value

    def shell_binding(self, index: int) -> float:
        """``getShellBinding``, in keV."""
        if index >= len(self.shells.shells):
            return 0.0
        value = self.shells.shells[index]
        return 0.0 if value is None else value


@dataclass(frozen=True)
class ElectronMaterial:
    """A crystal as the electron physics sees it.

    Attributes
    ----------
    atoms : tuple
        Each element with its atoms per unit cell, ordered by atomic number.
    cell_volume : float
        Unit cell volume in cubic Angstroms.
    density : float
        Density in g/cm^3.
    molecular_weight : float
        Mass of the unit cell's contents in atomic mass units.
    plasma_energy : float
        ``getPlasmaEnergyAll``: the free-electron plasma energy of the
        material, in eV. It depends on nothing but the material, so it is
        computed once rather than for every subshell as the Java does.
    total_charge : float
        Sum of atomic number times atoms per cell.
    sternheimer_a : float
        ``getSturnheimera``: the factor scaling every binding energy in the
        GOS model. It depends on nothing but the material, so it is found
        once rather than for every inelastic mean free path.
    """

    atoms: tuple[_Atom, ...]
    cell_volume: float
    density: float
    molecular_weight: float
    plasma_energy: float
    total_charge: float
    sternheimer_a: float = math.nan


def electron_material(composition: Composition, density: Any) -> ElectronMaterial:
    """Prepare a composition for the electron physics.

    Parameters
    ----------
    composition : Composition
        The unit cell's contents.
    density : Any
        Its density in g/cm^3, as
        :func:`pyraddose_3d.backend.coefficients.density` gives it.

    Returns
    -------
    ElectronMaterial
        The material, with each element's tables attached.

    """
    shells = _shell_table()
    elsepa = _elsepa()
    counts = np.asarray(composition.counts, dtype=float)
    atoms = []
    for element, count in zip(composition.elements, counts, strict=True):
        z = element.atomic_number
        atoms.append(
            _Atom(
                symbol=element.name,
                z=z,
                atomic_weight=element.atomic_weight,
                count=float(count),
                shells=shells[z],
                elastic=elsepa[z],
            )
        )

    cell_volume = float(composition.cell_volume)
    molecular_weight = 0.0
    total_charge = 0.0
    electron_density = 0.0
    for atom in atoms:
        molecular_weight += atom.count * atom.atomic_weight
        total_charge += atom.z * atom.count
        # Electrons per cm^3.
        electron_density += (atom.z * atom.count) / (cell_volume / 1e24)
    hbar_squared = math.pow(_PLANCK / (2 * math.pi), 2)
    e_squared = math.pow(_ELEMENTARY_CHARGE_ESU, 2) / 1000
    plasma = math.pow(
        4 * math.pi * hbar_squared * electron_density * e_squared / _ELECTRON_MASS,
        0.5,
    )
    plasma = (plasma / KEV_TO_JOULES) * 1000
    material = ElectronMaterial(
        atoms=tuple(atoms),
        cell_volume=cell_volume,
        density=float(density),
        molecular_weight=molecular_weight,
        plasma_energy=plasma,
        total_charge=total_charge,
    )
    if lacking_bindings(material):
        return material
    return dataclasses.replace(material, sternheimer_a=_sternheimer_a(material))


def lacking_bindings(material: ElectronMaterial) -> tuple[str, ...]:
    """``getElementsLackingBindingEnergies``: what the GOS model cannot use.

    Parameters
    ----------
    material : ElectronMaterial
        The crystal.

    Returns
    -------
    tuple[str, ...]
        The elements present, in atomic-number order, for which the tables
        give a zero or blank binding energy for a subshell up to the valence
        one. Empty when the GOS model has every binding energy it reads.
    """
    lacking = []
    for atom in sorted(material.atoms, key=lambda atom: atom.z):
        _, inner = atom.valence
        if any(b is None or b <= 0 for b in atom.shells.subshells[: inner + 1]):
            lacking.append(atom.symbol)
    return tuple(lacking)


# --- Stopping power -------------------------------------------------------


def stopping_power(material: ElectronMaterial, energy: float) -> float:
    """``calcStoppingPower``: the electronic and radiative stopping power.

    Bethe-Rohrlich, with a mean excitation energy averaged by Bragg's rule,
    a fixed density-effect correction of 0.1832, and a rough radiative term.

    Parameters
    ----------
    material : ElectronMaterial
        The crystal.
    energy : float
        Electron kinetic energy in keV.

    Returns
    -------
    float
        Stopping power in keV/nm.
    """
    m = _ELECTRON_MASS
    csquared = _SPEED_OF_LIGHT * _SPEED_OF_LIGHT
    vo = energy * KEV_TO_JOULES
    beta_squared = 1 - math.pow(m * csquared / (vo + m * csquared), 2)
    gamma = 1 / math.pow((1 - beta_squared), 0.5)
    kinetic = (gamma - 1) * m * csquared
    kinetic_mev = (kinetic / KEV_TO_JOULES) / 1000

    mean_ln_i = 0.0
    mean_z = 0.0
    sum_a = 0.0
    # Truncated to a long in RADDOSE-3D until 2026-10, which made the
    # stopping power depend on the order elements were summed in.
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
        mean_ln_i += fraction * (z / a) * math.log(j_star)

    mean_ln_i = mean_ln_i / (sum_z / sum_a)
    mean_j = math.exp(mean_ln_i)
    # Joy and Luo's correction for low energies.
    mean_j = mean_j / (1 + 0.85 * mean_j / (energy * 1000))
    mean_j = (mean_j / 1000) * KEV_TO_JOULES
    delta = 0.1832
    e = energy * KEV_TO_JOULES
    f_beta = (
        math.log((m * csquared * (e) * beta_squared) / (2 * (1 - beta_squared)))
        - (2 * math.pow((1 - beta_squared), 0.5) - 1 + beta_squared) * math.log(2)
        + 1
        - beta_squared
        + (1.0 / 8.0) * (1 - math.pow(1 - beta_squared, 0.5))
    )
    power = (
        (0.153536 / beta_squared)
        * (sum_z / sum_a)
        * (f_beta - 2 * math.log(mean_j) - delta)
    )
    power = power * 1000 * material.density / 1e7
    radiative = ((kinetic_mev * mean_z) / 800) * power
    return power + radiative


def csda_range(material: ElectronMaterial, energy: float) -> float:
    """``MicroED.getCSDArange``: distance to slow to 0.05 keV, in nm.

    A hundred equal energy steps, each over the distance the stopping power
    at its start gives.
    """
    step = energy / 100
    distance = 0.0
    while energy > _STOPPED:
        distance += step / stopping_power(material, energy)
        energy -= step
    return distance


# --- Elastic scattering ---------------------------------------------------


def elastic_mean_free_path(material: ElectronMaterial, energy: float) -> float:
    """``getElectronElasticMFPL``: the elastic mean free path, in nm.

    ELSEPA cross-sections between 0.05 and 300 keV; Langmore and Smith's
    formula outside that range, or where ELSEPA has no value.
    """
    m = _ELECTRON_MASS
    csquared = 3e8 * 3e8
    vo = energy * KEV_TO_JOULES
    beta_squared = 1 - math.pow(m * csquared / (vo + m * csquared), 2)
    part = 0.0
    for atom in material.atoms:
        z = float(atom.z)
        cross_section = ((1.4e-6 * math.pow(z, 1.5)) / beta_squared) * (
            1 - (0.26 * z) / (137 * math.pow(beta_squared, 0.5))
        )
        a = atom.elastic.atomic_weight
        fraction = (atom.count * a) / material.molecular_weight
        if energy <= 300 and energy >= 0.05:
            tabulated = atom.elastic.at(energy)
            if tabulated > 0.0:
                cross_section = tabulated
        part += (fraction * cross_section) / a
    return 1 / (_AVOGADRO * (material.density / 1e21) * part)


# --- Inelastic scattering: the GOS model ----------------------------------


def _wk(material: ElectronMaterial, a: float, atom: _Atom, index: int) -> float:
    """``getWkMolecule``: the resonance energy of one subshell, in eV."""
    fk = _SUBSHELL_ELECTRONS[index]
    valence, inner = atom.valence
    if index == inner:
        fk = valence
    return math.pow(
        math.pow(a * atom.subshell_binding(index) * 1000, 2)
        + (2.0 / 3.0)
        * ((fk * atom.count) / (material.total_charge))
        * math.pow(material.plasma_energy, 2),
        0.5,
    )


def _wak(energy: float, wk: float, uk: float) -> float:
    if energy * 1000 > 3 * wk - 2 * uk:
        return wk
    return (energy * 1000 + 2 * uk) / 3


def _qak(energy: float, wk: float, uk: float) -> float:
    if energy * 1000 > 3 * wk - 2 * uk:
        return uk
    return uk * (energy * 1000 / (3 * wk - 2 * uk))


def _q_minus(energy: float, wak: float) -> float:
    """``getQminusModified``, in J."""
    m = _ELECTRON_MASS
    csquared = _SPEED_OF_LIGHT * _SPEED_OF_LIGHT
    e = energy * KEV_TO_JOULES
    w = (wak / 1000) * KEV_TO_JOULES
    if w > e:
        return 0.0
    return (
        math.pow(
            math.pow(
                math.pow(e * (e + 2 * m * csquared), 0.5)
                - math.pow((e - w) * (e - w + 2 * m * csquared), 0.5),
                2,
            )
            + math.pow(m * csquared, 2),
            0.5,
        )
        - m * csquared
    )


def _wcb(material: ElectronMaterial) -> float:
    """``getWcbAll``: the conduction band's plasmon energy, in eV.

    Zero in practice: no shell is bound more weakly than ``_FCB_CUTOFF``.
    """
    sum_fcb = 0.0
    for atom in material.atoms:
        valence, inner = atom.valence_shells
        for index in range(inner + 1):
            if atom.shell_binding(index) * 1000 < _FCB_CUTOFF:
                sum_fcb += valence * atom.count
    return math.pow(sum_fcb / material.total_charge, 0.5) * material.plasma_energy


def _z_ln_i(material: ElectronMaterial) -> float:
    """``getZlnI``: the material's total charge times ln(I)."""
    sum_z = 0.0
    sum_a = 0.0
    mean_ln_i = 0.0
    for atom in material.atoms:
        a = atom.atomic_weight
        z = atom.z
        fraction = (atom.count * a) / material.molecular_weight
        sum_z += z * atom.count
        sum_a += a * atom.count
        j = atom.shells.mean_excitation
        if z not in _CONDENSED_PHASE:
            j *= 1.13
        mean_ln_i += fraction * (z / a) * math.log(j)
    mean_ln_i = mean_ln_i / (sum_z / sum_a)
    return sum_z * mean_ln_i


def _check_mean_i(material: ElectronMaterial, a: float) -> tuple[float, float]:
    """``checkMeanI``: Z ln(I), and what the resonance energies give for it."""
    z_ln_i = _z_ln_i(material)
    sum_fcb = 0.0
    sum_fk_ln_wk = 0.0
    for atom in material.atoms:
        valence, inner = atom.valence
        for index in range(inner + 1):
            fk = _SUBSHELL_ELECTRONS[index]
            if index == inner:
                uk = atom.subshell_binding(index) * 1000
                if uk > _FCB_CUTOFF:
                    fk = valence
                else:
                    sum_fcb += valence
                    fk = 0
            wk = _wk(material, a, atom, index)
            if wk > 0:
                sum_fk_ln_wk += atom.count * fk * math.log(wk)
    fcb_ln_wcb = 0.0
    wcb = _wcb(material)
    if wcb > 0.0:
        fcb_ln_wcb = sum_fcb * math.log(wcb)
    return z_ln_i, fcb_ln_wcb + sum_fk_ln_wk


def _sternheimer_a(material: ElectronMaterial) -> float:
    """``getSturnheimera``: the factor scaling every binding energy.

    Stepped by 0.1 from 1.5 until the resonance energies reproduce the mean
    excitation energy, then interpolated back if it went up. The Java passes
    the electron's energy in, but nothing it computes depends on it.
    """
    a = 1.5
    too_high = False
    check = _check_mean_i(material, a)
    last = check
    if check[1] > check[0]:
        a = 1.5
        while check[1] > check[0]:
            if a == 0.1:
                too_high = True
                break
            last = check
            a -= 0.1
            check = _check_mean_i(material, a)
    else:
        while check[1] < check[0]:
            if a >= 3:
                too_high = True
                break
            last = check
            a += 0.1
            check = _check_mean_i(material, a)
    if not too_high and a > 1.5:
        fraction_back = (check[1] - check[0]) / (check[1] - last[1])
        a -= 0.1 * fraction_back
    return a


def _close_analytical(w_in: float, energy: float, a: float) -> float:
    """``solveCloseAnalytical`` for n = 0, the cross-section."""
    e = energy * KEV_TO_JOULES
    w = (w_in / 1000) * KEV_TO_JOULES
    if e < w:
        return 0.0
    if w == 0.0:
        # -1/0 + log(E/0): minus infinity plus infinity, NaN in Java.
        return math.nan
    return (
        -1 / w
        + 1 / (e - w)
        + (((1 - a) / e) * math.log((e - w) / w))
        + a * w / math.pow(e, 2)
    )


def _close_integral(energy: float, uk: float, qak: float) -> float:
    """``doCloseIntegral``: close collisions, integrated analytically."""
    energy = energy + uk / 1000
    w_max = 1000 * energy / 2
    vo = energy * KEV_TO_JOULES
    m = _ELECTRON_MASS
    csquared = _SPEED_OF_LIGHT * _SPEED_OF_LIGHT
    a = math.pow(vo / (vo + m * csquared), 2)
    return _close_analytical(w_max, energy, a) - _close_analytical(qak, energy, a)


def _distant_integral(energy: float, binding: float, wk: float) -> float:
    """``integrateDist`` for n = 0: distant collisions, by the trapezium rule.

    The distribution, ``getpDisW``, is a triangle from the subshell's binding
    energy, ``binding`` in keV, to W_dis. The Java walks W from the binding
    energy in steps of W_dis / 1000 while W does not exceed W_dis, adding as
    it goes. The steps are accumulated and the trapezia summed in the same
    order, so the sums are the Java's.
    """
    uk = binding * 1000
    wak = _wak(energy, wk, uk)
    w_dis = ((3 * wak - 2 * uk) / 1000) * KEV_TO_JOULES
    step = w_dis / (_W_BINS)
    w_start = binding * KEV_TO_JOULES
    if not w_start <= w_dis:
        return 0.0
    count = int((w_dis - w_start) / step) + 3
    w = np.add.accumulate(np.concatenate(([w_start], np.full(count, step))))
    w = w[w <= w_dis]

    # getpDisW, which converts the binding energy by way of eV, so its lower
    # limit can be an ulp away from the walk's start.
    uk_j = (uk / 1000) * KEV_TO_JOULES
    inside = (w >= uk_j) & (w < w_dis)
    density = np.where(
        inside,
        (2 / math.pow(w_dis - uk_j, 2)) * (w_dis - w),
        0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(w == 0, 0.0, np.power(w, -1.0) * density)
    if y.size < 2:
        return 0.0
    trapezia = step * ((y[1:] + y[:-1]) / 2)
    return float(np.add.accumulate(trapezia)[-1])


#: Rows of ``populateGOSInel``'s table per element: K, L1, ..., the valence.
SUBSHELL_ROWS = 10


def inelastic_shells(material: ElectronMaterial, energy: float) -> np.ndarray:
    """``populateGOSInel`` for n = 0: each subshell's inverse mean free paths.

    For every element, in the material's order, and every subshell up to its
    valence one: distant longitudinal, distant transverse and close
    collisions, and their sum, in nm^-1 -- the Java's ``GOSinelastic`` map,
    with an element of no atoms left at zero. Rows past an element's valence
    subshell are zero, as the Java leaves them.

    Parameters
    ----------
    material : ElectronMaterial
        The crystal or the surrounding.
    energy : float
        Electron kinetic energy in keV.

    Returns
    -------
    numpy.ndarray
        ``(elements, SUBSHELL_ROWS, 4)``.
    """
    a = material.sternheimer_a
    m = _ELECTRON_MASS
    csquared = _SPEED_OF_LIGHT * _SPEED_OF_LIGHT
    vo = energy * KEV_TO_JOULES
    beta_squared = 1 - math.pow(m * csquared / (vo + m * csquared), 2)
    v_squared = beta_squared * csquared
    constant = (
        2 * math.pi * (math.pow(_ELEMENTARY_CHARGE_ESU, 4) / 1e18) / (m * v_squared)
    )
    delta_f = 0.0
    per_cell = material.cell_volume / 1000
    table = np.zeros((len(material.atoms), SUBSHELL_ROWS, 4))
    for row, atom in enumerate(material.atoms):
        if not atom.count > 0:
            continue
        valence, inner = atom.valence
        for index in range(inner + 1):
            fk = _SUBSHELL_ELECTRONS[index]
            if index == inner:
                fk = valence
            wk = _wk(material, a, atom, index)
            if not wk > 0:
                continue
            uk = atom.subshell_binding(index) * 1000
            if uk < _FCB_CUTOFF:
                fk = 0
            wak = _wak(energy, wk, uk)
            qak = _qak(energy, wk, uk)
            q_minus = _q_minus(energy, wak)
            integral = _distant_integral(energy, atom.subshell_binding(index), wk)
            if energy * 1000 > uk and uk >= _FCB_CUTOFF:
                longitudinal = 0.0
                if qak > 0 and q_minus > 0:
                    q = (qak / 1000) * KEV_TO_JOULES
                    longitudinal = (
                        1e18
                        * constant
                        * (
                            atom.count
                            * fk
                            * math.log(
                                (q / q_minus)
                                * (
                                    (q_minus + 2 * m * csquared)
                                    / (q + 2 * m * csquared)
                                )
                            )
                            * integral
                        )
                    ) / per_cell
                transverse = (
                    1e18
                    * constant
                    * (
                        atom.count
                        * fk
                        * (math.log(1 / (1 - beta_squared)) - beta_squared - delta_f)
                        * integral
                    )
                ) / per_cell
                close = (
                    1e18
                    * constant
                    * (atom.count * fk * _close_integral(energy, uk, qak))
                ) / per_cell
                table[row, index] = (
                    longitudinal,
                    transverse,
                    close,
                    longitudinal + transverse + close,
                )
    # The plasmon term: zero, since nothing joins the conduction band.
    wcb = _wcb(material)
    if wcb > 0 and energy * 1000 > wcb:
        raise NotImplementedError("a conduction-band plasmon term is never reached")
    return table


def inelastic_mean_free_path(material: ElectronMaterial, energy: float) -> float:
    """``getGOSInel``: the inelastic mean free path, in nm.

    The generalised oscillator strength model: for every subshell of every
    element, distant collisions -- longitudinal and transverse -- and close
    ones, each a cross-section per atom times the atoms per cell over the
    cell's volume (:func:`inelastic_shells`), summed in the Java's order.

    Parameters
    ----------
    material : ElectronMaterial
        The crystal.
    energy : float
        Electron kinetic energy in keV.

    Returns
    -------
    float
        The mean free path in nm; NaN for a crystal containing an element
        in :func:`lacking_bindings`.
    """
    if lacking_bindings(material):
        return math.nan
    total = 0.0
    for shells in inelastic_shells(material, energy):
        for row in shells:
            total += float(row[3])
    return 1 / total


# --- The calculation ------------------------------------------------------


@dataclass(frozen=True)
class MicroEDResult:
    """What ``MicroED.CalculateEM`` reports.

    Attributes
    ----------
    energy : float
        Beam energy in keV.
    dose : float
        Mean dose over the slices of the exposed volume, in MGy.
    slice_doses : tuple[float, ...]
        Dose of each slice, from the entrance face, in MGy. The slices share
        the thickness equally, at least 5 nm each, and there is always at
        least one. Slices after the electrons have stopped are zero.
    slice_width : float
        Their width, in nm.
    elastic, single_elastic, inelastic, productive : float
        Electrons scattered elastically at least once, exactly once,
        inelastically, and productive: once elastically, never inelastically.
    information_coefficient : float
        Productive electrons per MGy.
    best_energy : float
        The accelerating voltage the grid search finds best, in kV.
    best_thickness : float
        The thickness the grid search finds best, in nm.
    density : float
        Crystal density in g/cm^3.
    elastic_mean_free_path, inelastic_mean_free_path : float
        At the beam energy, in nm.
    csda_range : float
        Continuous-slowing-down range at the beam energy, in nm.
    lacking_bindings : tuple[str, ...]
        The elements the inelastic model has no binding energies for. When
        there are any, every value the inelastic model feeds is NaN.
    """

    energy: float
    dose: float
    slice_doses: tuple[float, ...]
    slice_width: float
    elastic: float
    single_elastic: float
    inelastic: float
    productive: float
    information_coefficient: float
    best_energy: float
    best_thickness: float
    density: float
    elastic_mean_free_path: float
    inelastic_mean_free_path: float
    csda_range: float
    lacking_bindings: tuple[str, ...] = ()


def _exposed_area(
    dimensions_nm: tuple[float, float, float],
    aperture: tuple[float, float],
    circular: bool,
) -> float:
    """The exposed area in um^2: the beam, cut to the crystal on each axis."""
    exposed_x = (
        aperture[0]
        if dimensions_nm[0] / 1000 > aperture[0]
        else dimensions_nm[0] / 1000
    )
    exposed_y = (
        aperture[1]
        if dimensions_nm[1] / 1000 > aperture[1]
        else dimensions_nm[1] / 1000
    )
    if not circular:
        return exposed_x * exposed_y
    return math.pi * ((exposed_x / 2) * (exposed_y / 2))


def _exposed_mass(material: ElectronMaterial, area: float, thickness: float) -> float:
    """Mass of the exposed volume, in kg."""
    volume = area * (thickness / 1000) * 1e-15
    return ((material.density * 1000) * volume) / 1000


def _information_coefficient(
    material: ElectronMaterial,
    energy: float,
    electrons: float,
    mass: float,
    thickness: float,
) -> float:
    """``MicroED.getInfoCoef``: productive electrons per MGy, in one step."""
    power = stopping_power(material, energy)
    deposited = electrons * (power * thickness) * KEV_TO_JOULES
    dose = (deposited / mass) / 1e06
    elastic = elastic_mean_free_path(material, energy)
    inelastic = inelastic_mean_free_path(material, energy)
    inelastic_probability = 1 - math.exp(-thickness / inelastic)
    single = (
        electrons
        * math.exp(-thickness / elastic)
        * (math.pow(thickness / elastic, 1) / 1)
    )
    productive = single * (1 - inelastic_probability)
    return productive / dose


def _best_energy(
    material: ElectronMaterial, electrons: float, mass: float, thickness: float
) -> float:
    """``MicroED.getOptimalEnergy``: a coarse scan, then 10 kV steps."""
    scanned: list[float] = []
    exit_i = 0
    for i, energy in enumerate(_ENERGY_GRID):
        exit_i = i
        scanned.append(
            _information_coefficient(material, energy, electrons, mass, thickness)
        )
        if i > 1 and scanned[i] < scanned[i - 1]:
            break
    start = _ENERGY_GRID[exit_i - 2]
    end = _ENERGY_GRID[exit_i]
    steps = (int(end - start) // _ENERGY_STEP) + 1
    previous = 0.0
    energy = 0.0
    for i in range(steps):
        energy = start + i * _ENERGY_STEP
        coefficient = _information_coefficient(
            material, energy, electrons, mass, thickness
        )
        if i > 0 and coefficient < previous:
            break
        previous = coefficient
    return energy - _ENERGY_STEP


def _best_thickness(
    material: ElectronMaterial, electrons: float, area: float, energy: float
) -> float:
    """``MicroED.getOptimalThickness``: a coarse scan, then 5 nm steps."""
    scanned: list[float] = []
    exit_i = 0
    for i, thickness in enumerate(_THICKNESS_GRID):
        exit_i = i
        mass = _exposed_mass(material, area, thickness)
        scanned.append(
            _information_coefficient(material, energy, electrons, mass, thickness)
        )
        if i > 1 and scanned[i] < scanned[i - 1]:
            break
    start = _THICKNESS_GRID[exit_i - 2]
    end = _THICKNESS_GRID[exit_i]
    steps = (int(end - start) // _THICKNESS_STEP) + 1
    previous = 0.0
    thickness = 0.0
    for i in range(steps):
        thickness = start + i * _THICKNESS_STEP
        mass = _exposed_mass(material, area, thickness)
        coefficient = _information_coefficient(
            material, energy, electrons, mass, thickness
        )
        if i > 0 and coefficient < previous:
            break
        previous = coefficient
    return thickness - _THICKNESS_STEP


def microed(
    material: ElectronMaterial,
    *,
    dimensions_nm: tuple[float, float, float],
    aperture: tuple[float, float],
    circular: bool,
    energy: float,
    exposure: float,
) -> MicroEDResult:
    """Run RADDOSE-3D's MicroED calculation.

    Parameters
    ----------
    material : ElectronMaterial
        The crystal's contents.
    dimensions_nm : tuple[float, float, float]
        The crystal's extent along x, y and z, in nm. z is along the beam,
        so it is the thickness.
    aperture : tuple[float, float]
        The beam's size in x and y, in um: its collimation.
    circular : bool
        Whether the aperture is an ellipse rather than a rectangle.
    energy : float
        Electron energy in keV.
    exposure : float
        Fluence in electrons per square Angstrom.

    Returns
    -------
    MicroEDResult
        The dose, the scattering counts and the grid searches.
    """
    lacking = lacking_bindings(material)
    if lacking:
        warnings.warn(lacking_bindings_warning(lacking), stacklevel=2)
    thickness = dimensions_nm[2]
    slices = max(1, int(thickness / _SLICE_THICKNESS))

    range_nm = csda_range(material, energy)

    # MicroED.EMEquationWay: the scattering counts.
    area = _exposed_area(dimensions_nm, aperture, circular)
    electrons = exposure * (area * 1e08)
    elastic_lambda = elastic_mean_free_path(material, energy)
    elastic_probability = 1 - math.exp(-thickness / elastic_lambda)
    elastic = elastic_probability * electrons
    single_elastic = (
        electrons
        * math.exp(-thickness / elastic_lambda)
        * (math.pow(thickness / elastic_lambda, 1) / 1)
    )
    inelastic_lambda = inelastic_mean_free_path(material, energy)
    inelastic_probability = 1 - math.exp(-thickness / inelastic_lambda)
    inelastic = inelastic_probability * electrons
    productive = single_elastic * (1 - inelastic_probability)

    # MicroED.EMStoppingPowerWay: the dose, slice by slice.
    mass_per_slice = _exposed_mass(material, area, thickness) / slices
    slice_doses = [0.0] * slices
    total = 0.0
    remaining = energy
    for i in range(slices):
        power = stopping_power(material, remaining)
        per_electron = power * (thickness / slices)
        remaining -= per_electron
        stopped = remaining < _STOPPED
        if stopped:
            per_electron += remaining
        deposited = electrons * per_electron * KEV_TO_JOULES
        slice_doses[i] = (deposited / mass_per_slice) / 1e06
        total += slice_doses[i]
        if stopped:
            break
    dose = total / slices

    information = productive / dose
    # Both searches stop where the information coefficient falls, which a NaN
    # never does, so without the inelastic model there is nothing to search.
    best_energy = best_thickness = math.nan
    if not lacking:
        best_energy = _best_energy(
            material, electrons, _exposed_mass(material, area, thickness), thickness
        )
        best_thickness = _best_thickness(material, electrons, area, energy)

    return MicroEDResult(
        energy=energy,
        dose=dose,
        slice_doses=tuple(slice_doses),
        slice_width=thickness / slices,
        elastic=elastic,
        single_elastic=single_elastic,
        inelastic=inelastic,
        productive=productive,
        information_coefficient=information,
        best_energy=best_energy,
        best_thickness=best_thickness,
        density=material.density,
        elastic_mean_free_path=elastic_lambda,
        inelastic_mean_free_path=inelastic_lambda,
        csda_range=range_nm,
        lacking_bindings=lacking,
    )


def lacking_bindings_warning(lacking: tuple[str, ...]) -> str:
    """What RADDOSE-3D prints for elements in :func:`lacking_bindings`.

    It follows ``Warning: `` in RADDOSE-3D's output.
    """
    return (
        f"the element tables have no binding energy for one or more "
        f"of the subshells of {', '.join(lacking)}, which the inelastic (GOS) "
        f"model needs. The inelastic and productive counts, the information "
        f"coefficient and the optimal voltage and thickness are reported as "
        f"NaN. The dose and the elastic counts are unaffected."
    )
