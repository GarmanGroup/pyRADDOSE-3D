"""How much of the diffracted beam a detector's sensor absorbs.

Not part of RADDOSE-3D. Its diffraction efficiency counts the photons the
crystal scatters elastically; whether a detector records them depends on the
sensor, and above all on the energy. A thin silicon sensor stops nearly
every 8 keV photon and lets most 25 keV photons straight through, so an
energy that is best for the crystal can be poor for the experiment.

The fraction absorbed is ``1 - exp(-(mu/rho) rho t)``, with ``mu/rho`` from
the photoelectric cross-sections of the same McMaster parameterisation the
crystal uses: photoabsorption is what turns a photon into a signal, and
scattering mostly does not. Treat it as an upper bound on the detector's
quantum efficiency -- it leaves out charge sharing, fluorescence escape,
counting thresholds and dead area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyraddose_3d.backend._array import array_namespace
from pyraddose_3d.backend.composition import AVOGADRO
from pyraddose_3d.backend.elements import element_database

__all__ = ["SENSORS", "Sensor", "sensor_absorption", "sensor_edges"]

Array = Any

#: Barns to square centimetres.
_BARNS_TO_CM2 = 1e-24
#: Micrometres to centimetres.
_UM_TO_CM = 1e-4


@dataclass(frozen=True)
class Sensor:
    """A sensor material.

    Attributes
    ----------
    formula : dict[str, float]
        Element symbol to atoms per formula unit.
    density : float
        Density in g/cm^3.
    """

    formula: dict[str, float]
    density: float


#: Common sensor materials, by the name a configuration uses.
SENSORS: dict[str, Sensor] = {
    "Si": Sensor({"Si": 1.0}, 2.329),
    "CdTe": Sensor({"Cd": 1.0, "Te": 1.0}, 5.85),
    "GaAs": Sensor({"Ga": 1.0, "As": 1.0}, 5.3176),
}


def _mass_absorption(sensor: Sensor, energy: Array) -> Array:
    """Photoelectric mass attenuation coefficient, cm^2/g."""
    database = element_database()
    grams_per_mole = 0.0
    cross_section: Array = 0.0
    for symbol, count in sensor.formula.items():
        element = database.by_symbol(symbol)
        grams_per_mole += count * element.atomic_weight
        photoelectric = element.absorption_coefficients(energy).photoelectric
        cross_section = cross_section + count * photoelectric
    return cross_section * _BARNS_TO_CM2 * AVOGADRO / grams_per_mole


def sensor_absorption(
    energy: Array, material: str = "Si", thickness: float = 450.0
) -> Array:
    """The fraction of photons a sensor absorbs.

    Parameters
    ----------
    energy : Array
        Photon energy in keV. A tensor result carries a gradient.
    material : str
        A key of :data:`SENSORS`.
    thickness : float
        Sensor thickness in micrometres.

    Returns
    -------
    Array
        Between 0 and 1, the shape of ``energy``.
    """
    sensor = SENSORS[material]
    mu = _mass_absorption(sensor, energy) * sensor.density
    xp = array_namespace(mu)
    return -xp.expm1(-mu * thickness * _UM_TO_CM)


def sensor_edges(material: str) -> tuple[float, ...]:
    """Energies at which a sensor's absorption jumps, in keV.

    Parameters
    ----------
    material : str
        A key of :data:`SENSORS`.

    Returns
    -------
    tuple[float, ...]
        Ascending.
    """
    database = element_database()
    edges = {
        edge for symbol in SENSORS[material].formula
        for edge in database.by_symbol(symbol).edges
    }  # fmt: skip
    return tuple(sorted(edges))
