"""Attenuation of the beam by a sample container.

Ports ``ContainerMixture`` and ``ContainerElemental``. A container -- a loop,
a capillary, a sheet of film -- sits in the beam ahead of the crystal and
absorbs some of it, so the crystal sees ``flux * (1 - f)`` with

    f = 1 - exp(-(mu/rho) * rho * t)

where mu/rho is the container's mass attenuation coefficient at the beam
energy, rho its density and t its thickness.

RADDOSE-3D looks mu/rho up in NIST's tables, downloading them on every run.
They are vendored here instead, extracted with the Java's own parsing, so
the rows are the ones it interpolates between. See
``tests/backend/data/build_nist_tables.py``.

The interpolation is reproduced exactly, including two things a fresh
implementation would do differently:

- It is **linear in energy and in mu/rho**, not log-log, although mu/rho
  falls by orders of magnitude between table rows.
- It keeps the **last** row at or below the beam energy. At an absorption
  edge the tables list the same energy twice, below then above the edge, so
  a beam exactly at an edge interpolates from the value above it.
"""

from __future__ import annotations

import functools
import math
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend._array import array_namespace, to_namespace
from pyraddose_3d.backend.elements import element_database

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "container_transmission",
    "elemental_mass_attenuation",
    "known_mixtures",
    "mixture_mass_attenuation",
]

Array = Any

#: ``ContainerSemiTransparent.MICRONS_TO_CENTIMETERS``.
_MICRONS_TO_CENTIMETERS = 1e-4
#: ``ContainerSemiTransparent.MEV_TO_KEV``.
_MEV_TO_KEV = 1e3


@functools.cache
def _tables(file_name: str) -> dict[str, list[tuple[float, float]]]:
    """Rows of one vendored table, keyed by element number or material."""
    text = resources.files("pyraddose_3d.data").joinpath(file_name).read_text("ascii")
    tables: dict[str, list[tuple[float, float]]] = {}
    header_seen = False
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if not header_seen:
            header_seen = True
            continue
        key, energy_mev, mu_rho = line.split("\t")
        # Parsed and scaled exactly as the Java does: Double.parseDouble of
        # the text, then times MEV_TO_KEV.
        tables.setdefault(key, []).append(
            (float(energy_mev) * _MEV_TO_KEV, float(mu_rho))
        )
    return tables


def known_mixtures() -> frozenset[str]:
    """The compound and mixture names NIST tabulates."""
    return frozenset(_tables("nist_mixtures.tsv"))


def _interpolate(rows: list[tuple[float, float]], energy: Array) -> Array:
    """The Java's walk through a table, then its linear interpolation.

    The walk is discrete -- which pair of rows brackets the energy -- so it
    is taken on a plain float, and the interpolation then uses the energy as
    given, so a gradient reaches it.
    """
    at = float(to_namespace(np, energy))
    previous_energy, previous_mu = 0.0, 0.0
    next_energy, next_mu = 0.0, 0.0
    for next_energy, next_mu in rows:
        if next_energy > at:
            break
        previous_energy, previous_mu = next_energy, next_mu
    return previous_mu + (next_mu - previous_mu) * (
        (energy - previous_energy) / (next_energy - previous_energy)
    )


def mixture_mass_attenuation(material: str, energy: Array) -> Array:
    """Mass attenuation coefficient of a named NIST compound or mixture.

    Parameters
    ----------
    material : str
        A name NIST tabulates, e.g. ``"water"`` or ``"mylar"``.
    energy : Array
        Photon energy in keV.

    Returns
    -------
    Array
        mu/rho in cm^2/g.

    Raises
    ------
    KeyError
        If NIST does not tabulate the material.
    """
    tables = _tables("nist_mixtures.tsv")
    if material not in tables:
        raise KeyError(
            f"NIST does not tabulate {material!r}; known: {', '.join(sorted(tables))}"
        )
    return _interpolate(tables[material], energy)


def elemental_mass_attenuation(elements: Mapping[str, float], energy: Array) -> Array:
    """Mass attenuation coefficient of a container given by its elements.

    Each element's mu/rho is interpolated from its own NIST table, and they
    are combined by mass fraction, using RADDOSE-3D's atomic weights.

    Parameters
    ----------
    elements : Mapping[str, float]
        Element symbol to number of atoms, as in a formula.
    energy : Array
        Photon energy in keV.

    Returns
    -------
    Array
        mu/rho in cm^2/g.
    """
    database = element_database()
    tables = _tables("nist_elements.tsv")

    coefficients, weights = [], []
    total_weight = 0.0
    for symbol, count in elements.items():
        element = database.by_symbol(symbol)
        coefficients.append(_interpolate(tables[str(element.atomic_number)], energy))
        weights.append(element.atomic_weight)
        total_weight += element.atomic_weight * count

    # Accumulated in the order the elements were given, as the Java does.
    total: Array = 0.0
    for count, weight, coefficient in zip(
        elements.values(), weights, coefficients, strict=True
    ):
        total = total + ((count * weight) / total_weight) * coefficient
    return total


def container_transmission(
    mass_attenuation: Array, density: float, thickness: float
) -> Array:
    """The fraction of the beam that gets through the container.

    Parameters
    ----------
    mass_attenuation : Array
        mu/rho in cm^2/g.
    density : float
        Container density in g/cm^3.
    thickness : float
        Container thickness in micrometres.

    Returns
    -------
    Array
        ``1 - f``, computed from ``f`` as ``BeamGaussian`` does rather than
        as ``exp(-x)`` directly; the two differ in the last bit.
    """
    mass_thickness = density * (_MICRONS_TO_CENTIMETERS * thickness)
    exponent = -mass_attenuation * mass_thickness
    if isinstance(exponent, float):
        attenuated = 1 - math.exp(exponent)
    else:
        attenuated = 1 - array_namespace(exponent).exp(exponent)
    return 1 - attenuated
