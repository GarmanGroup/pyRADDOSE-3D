"""Electrons, followed step by step: ``trackPhotoelectron`` in simple mode.

RADDOSE-3D follows one electron to its end before starting the next. Here
every electron advances one step per pass, and the passes go on until the
last has stopped or left. Each step, as the Java takes it:

- the electron moves ``s`` along its direction, ``s`` drawn from the elastic
  mean free path at its energy;
- if it is still inside, it deposits ``s`` times the stopping power at the
  point it reached, scatters elastically off an element drawn by its share
  of the elastic cross-section, through ELSEPA's angle for that element at
  the nearest tabulated energy, and loses that energy;
- if not, it deposits the stopping power times the distance to the
  nearest face plane at the point it left from, and is no longer followed;
- below 0.05 keV it deposits what is left -- negative, if the last step
  overshot, so that the energy deposited adds up to the energy it had.

The random number the Java draws each step to choose an inelastic event,
which in simple mode never happens, is not drawn.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pyraddose_3d.backend.montecarlo.angles import deflection_angles
from pyraddose_3d.backend.montecarlo.geometry import exit_distance, inside
from pyraddose_3d.backend.montecarlo.material import (
    elastic_cross_sections,
    elastic_element_cdf,
    elastic_mean_free_path,
    stopping_power,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyraddose_3d.backend.geometry import Mesh
    from pyraddose_3d.backend.microed import ElectronMaterial
    from pyraddose_3d.backend.montecarlo.geometry import Placement

__all__ = ["STOPPED_KEV", "new_directions", "track"]

#: Below this energy, keV, an electron has stopped.
STOPPED_KEV = 0.05


def new_directions(
    direction: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """``getNewDirectionVector``: each direction turned by theta, then phi.

    The Java's formula, which is not a rotation by theta about the old
    direction (``UPSTREAM_FIXES.md``), normalised as it normalises it.
    """
    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        an = -(x / y)
        am = 1 / np.power(1 + an * an, 0.5)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    v1 = an * sin_theta
    v2 = an * am * sin_theta
    v3 = np.cos(phi)
    v4 = np.sin(phi)
    ca = (x * cos_theta) + (v1 * v3) + (y * v2 * v4)
    cb = (y * cos_theta) + (v4 * (z * v1 - x * v2))
    cc = (z * cos_theta) + (v2 * v3) - (y * v1 * v4)
    turned = np.stack([ca, cb, cc], axis=1)
    magnitude = np.sqrt(np.sum(np.power(turned, 2), axis=1, keepdims=True))
    return turned / magnitude


def track(
    rng: np.random.Generator,
    material: ElectronMaterial,
    mesh: Mesh,
    position: np.ndarray,
    direction: np.ndarray,
    energy: np.ndarray,
    placement: Placement,
    deposit: Callable[[np.ndarray, np.ndarray, np.ndarray], None],
) -> int:
    """Follow every electron until it stops or leaves the crystal.

    Parameters
    ----------
    rng : numpy.random.Generator
        The run's random numbers.
    material : ElectronMaterial
        The crystal, for the electron physics.
    mesh : Mesh
        The crystal, unrotated, um.
    position : numpy.ndarray
        ``(N, 3)``: where each electron starts, nm, lab frame.
    direction : numpy.ndarray
        ``(N, 3)``: unit directions.
    energy : numpy.ndarray
        Starting energies, keV.
    placement : Placement
        How the crystal stood for each electron's photon.
    deposit : Callable
        Called as ``deposit(points_nm, which, energy_kev)`` with the
        electrons' indices into the arrays passed in.

    Returns
    -------
    int
        How many passes it took: the most steps any electron made.
    """
    z_of = np.array([atom.z for atom in material.atoms])
    which = np.arange(len(energy))
    previous = np.array(position, dtype=float)
    direction = np.array(direction, dtype=float)
    energy = np.array(energy, dtype=float)

    # ``startingEnergy < 0.05``: the whole energy, where it was freed.
    stopped = energy < STOPPED_KEV
    if stopped.any():
        deposit(previous[stopped], which[stopped], energy[stopped])
    keep = ~stopped
    which, previous, direction, energy = (
        which[keep],
        previous[keep],
        direction[keep],
        energy[keep],
    )
    sigma = elastic_cross_sections(material, energy)
    step = -elastic_mean_free_path(material, sigma) * np.log(rng.random(len(energy)))

    passes = 0
    while len(which):
        passes += 1
        place = placement.take(which)
        reached = previous + step[:, None] * direction
        within = inside(mesh, reached, place)
        power = stopping_power(material, energy)

        # Still inside: deposit, scatter, slow down, and draw the next step.
        a = np.flatnonzero(within)
        lost = step[a] * power[a]
        deposit(reached[a], which[a], lost)
        previous[a] = reached[a]
        cdf = elastic_element_cdf(material, sigma[:, a])
        element = _first_above(cdf, rng.random(len(a)))
        theta = deflection_angles(z_of[element], energy[a], rng.random(len(a)))
        phi = 2 * math.pi * rng.random(len(a))
        direction[a] = new_directions(direction[a], theta, phi)
        energy[a] = energy[a] - lost

        # Left: the distance to the nearest face plane, at the stopping power.
        b = np.flatnonzero(~within)
        to_edge = exit_distance(mesh, previous[b], direction[b])
        lost = to_edge * power[b]
        energy[b] = energy[b] - lost
        deposit(previous[b], which[b], lost)

        # Stopped: what is left, where it stopped.
        stopped = energy < STOPPED_KEV
        if stopped.any():
            deposit(previous[stopped], which[stopped], energy[stopped])

        keep = within & ~stopped
        which, previous, direction, energy = (
            which[keep],
            previous[keep],
            direction[keep],
            energy[keep],
        )
        sigma = elastic_cross_sections(material, energy)
        step = -elastic_mean_free_path(material, sigma) * np.log(
            rng.random(len(energy))
        )
    return passes


def _first_above(cdf: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """``getElasticElement``: the first element whose share passes the draw.

    The last, should rounding leave none; the Java would fail.
    """
    above = cdf > uniform[None, :]
    found = above.any(axis=0)
    return np.where(found, above.argmax(axis=0), cdf.shape[0] - 1)
