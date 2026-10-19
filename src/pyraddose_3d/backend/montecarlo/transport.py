"""Electrons, followed step by step: ``trackPhotoelectron`` in simple mode.

RADDOSE-3D follows one electron to its end before starting the next. Here
every electron advances one step per pass, and the passes go on until the
last has stopped or left. Each step, as the Java takes it:

- the electron moves ``s`` along its direction, ``s`` drawn from the elastic
  mean free path of the medium it is in, at its energy;
- if it is in the crystal, it deposits ``s`` times the stopping power at the
  point it reached, scatters elastically off an element drawn by its share
  of the elastic cross-section, through ELSEPA's angle for that element at
  the nearest tabulated energy, and loses that energy;
- if it has just left the crystal, it deposits the stopping power times the
  distance to the crystal's surface, at the point it left from. With no
  surrounding it is followed no further; with one, it carries on there;
- in the surrounding it scatters and slows the same way but deposits
  nothing, until it leaves the surrounding, or comes back into the crystal
  -- moved to the face it enters by, and slowed by the way;
- below 0.05 keV it deposits what is left, if it is in the crystal --
  negative, if the last step overshot, so that the energy deposited adds up
  to the energy it had.

An electron that comes back into the crystal with under 0.05 keV leaves it
where it came in; one that leaves into the surrounding with under 0.05 keV
leaves it where it left from, as with no surrounding. Both were lost, until
RADDOSE-3D #49. The random number
the Java draws each step to choose an inelastic event, which in simple mode
never happens, is not drawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    from collections.abc import Callable, Sequence

    from pyraddose_3d.backend.geometry import Mesh
    from pyraddose_3d.backend.microed import ElectronMaterial
    from pyraddose_3d.backend.montecarlo.geometry import Placement

__all__ = ["STOPPED_KEV", "Media", "new_directions", "track"]

#: Below this energy, keV, an electron has stopped.
STOPPED_KEV = 0.05


@dataclass(frozen=True)
class Media:
    """What the electrons move through.

    Attributes
    ----------
    crystal : ElectronMaterial
        The crystal.
    mesh : Mesh
        Its shape, unrotated, um.
    surrounding : ElectronMaterial | None
        The medium around it, if any (``CalcSurrounding``).
    boxes : Sequence[Mesh]
        The surrounding's extent: the crystal's mesh scaled about its centre
        by its thickness.
    """

    crystal: ElectronMaterial
    mesh: Mesh
    surrounding: ElectronMaterial | None = None
    boxes: Sequence[Mesh] = ()


def new_directions(
    direction: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """``getNewDirectionVector``: each direction turned through theta.

    At phi around the old direction, as RADDOSE-3D has it since #47.
    """
    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    polar = np.abs(z) > 0.99999
    s = np.sqrt(np.where(polar, 1.0, 1 - z * z))
    turned = np.stack(
        [
            np.where(
                polar,
                sin_t * cos_p,
                sin_t * (x * z * cos_p - y * sin_p) / s + x * cos_t,
            ),
            np.where(
                polar,
                sin_t * sin_p,
                sin_t * (y * z * cos_p + x * sin_p) / s + y * cos_t,
            ),
            np.where(polar, np.sign(z) * cos_t, -sin_t * cos_p * s + z * cos_t),
        ],
        axis=1,
    )
    magnitude = np.sqrt(np.sum(np.power(turned, 2), axis=1, keepdims=True))
    return np.asarray(turned / magnitude)


class _Physics:
    """Stopping power, steps and scattering, in whichever medium each is."""

    def __init__(self, media: Media, rng: np.random.Generator) -> None:
        self.media = media
        self.rng = rng
        surrounding = media.surrounding
        self.z = {
            False: np.array([a.z for a in media.crystal.atoms]),
            True: np.array([a.z for a in surrounding.atoms])
            if surrounding is not None
            else np.zeros(0, dtype=np.int64),
        }

    def _material(self, surrounding: bool) -> ElectronMaterial:
        material = self.media.surrounding if surrounding else self.media.crystal
        assert material is not None
        return material

    def _each(
        self,
        outside: np.ndarray,
        compute: Callable[[bool, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        result = np.empty(len(outside))
        for flag in (False, True):
            chosen = np.flatnonzero(outside == flag)
            if len(chosen):
                result[chosen] = compute(flag, chosen)
        return result

    def stopping_power(self, energy: np.ndarray, outside: np.ndarray) -> np.ndarray:
        """``getStoppingPower`` in each electron's medium, keV/nm."""
        return self._each(
            outside, lambda flag, i: stopping_power(self._material(flag), energy[i])
        )

    def step(self, energy: np.ndarray, outside: np.ndarray) -> np.ndarray:
        """A distance to the next elastic event, nm."""

        def free_path(flag: bool, i: np.ndarray) -> np.ndarray:
            material = self._material(flag)
            sigma = elastic_cross_sections(material, energy[i])
            return np.asarray(elastic_mean_free_path(material, sigma))

        return -self._each(outside, free_path) * np.log(self.rng.random(len(energy)))

    def deflect(
        self, direction: np.ndarray, energy: np.ndarray, outside: np.ndarray
    ) -> np.ndarray:
        """Scatter elastically: an element, its ELSEPA angle, and phi."""
        theta = np.empty(len(energy))
        for flag in (False, True):
            chosen = np.flatnonzero(outside == flag)
            if not len(chosen):
                continue
            material = self._material(flag)
            sigma = elastic_cross_sections(material, energy[chosen])
            cdf = elastic_element_cdf(material, sigma)
            element = _first_above(cdf, self.rng.random(len(chosen)))
            theta[chosen] = deflection_angles(
                self.z[flag][element],
                energy[chosen],
                self.rng.random(len(chosen)),
                tuple(self.z[flag].tolist()),
            )
        phi = 2 * math.pi * self.rng.random(len(energy))
        return new_directions(direction, theta, phi)


def track(
    rng: np.random.Generator,
    media: Media,
    position: np.ndarray,
    direction: np.ndarray,
    energy: np.ndarray,
    in_surrounding: np.ndarray,
    placement: Placement,
    box: np.ndarray,
    deposit: Callable[[np.ndarray, np.ndarray, np.ndarray, bool], None],
) -> int:
    """Follow every electron until it stops or leaves.

    Parameters
    ----------
    rng : numpy.random.Generator
        The run's random numbers.
    media : Media
        The crystal, and the surrounding if there is one.
    position : numpy.ndarray
        ``(N, 3)``: where each electron starts, nm, lab frame.
    direction : numpy.ndarray
        ``(N, 3)``: unit directions.
    energy : numpy.ndarray
        Starting energies, keV.
    in_surrounding : numpy.ndarray
        Which start in the surrounding rather than the crystal.
    placement : Placement
        How the crystal stood for each electron's photon.
    box : numpy.ndarray
        Each electron's surrounding: an index into ``media.boxes``.
    deposit : Callable
        Called as ``deposit(points_nm, which, energy_kev, starting)`` for
        energy left in the crystal by the electrons ``which`` -- indices
        into the arrays passed in. ``starting`` marks an electron freed with
        under 0.05 keV, which RADDOSE-3D also adds to its XFEL tallies.

    Returns
    -------
    int
        How many passes it took: the most steps any electron made.
    """
    physics = _Physics(media, rng)
    which = np.arange(len(energy))
    previous = np.array(position, dtype=float)
    direction = np.array(direction, dtype=float)
    energy = np.array(energy, dtype=float)
    outside = np.array(in_surrounding, dtype=bool)
    box = np.asarray(box)

    def crystal_has(points: np.ndarray, electrons: np.ndarray) -> np.ndarray:
        return inside(media.mesh, points, placement.take(electrons))

    # ``startingEnergy < 0.05``: the whole energy, where it was freed, if that
    # is in the crystal.
    low = np.flatnonzero(energy < STOPPED_KEV)
    if len(low):
        there = low[crystal_has(previous[low], which[low])]
        deposit(previous[there], which[there], energy[there], True)
    keep = energy >= STOPPED_KEV
    which, previous, direction, energy, outside, box = (
        which[keep], previous[keep], direction[keep], energy[keep],
        outside[keep], box[keep],
    )  # fmt: skip
    step = physics.step(energy, outside)

    passes = 0
    while len(which):
        passes += 1
        reached = previous + step[:, None] * direction
        within = crystal_has(reached, which)
        power = physics.stopping_power(energy, outside)
        # ``finished`` ends an electron at once; ``exits`` after the check
        # for having stopped, as the Java's loop ends.
        finished = np.zeros(len(which), dtype=bool)
        exits = np.zeros(len(which), dtype=bool)

        # Back into the crystal from the surrounding: to the face it enters by,
        # slowed by the way, then a fresh step in the crystal.
        entering = np.flatnonzero(within & outside)
        if len(entering):
            to_face = exit_distance(
                media.mesh,
                previous[entering],
                direction[entering],
                placement.take(which[entering]),
            )
            previous[entering] += to_face[:, None] * direction[entering]
            energy[entering] -= to_face * power[entering]
            died = energy[entering] < STOPPED_KEV
            finished[entering[died]] = True
            # Under 0.05 keV, it leaves what it has where it came in, if it
            # got there.
            arrived = entering[died & (energy[entering] > 0)]
            deposit(previous[arrived], which[arrived], energy[arrived], False)
            entering = entering[~died]
            outside[entering] = False
            power[entering] = physics.stopping_power(
                energy[entering], outside[entering]
            )
            step[entering] = physics.step(energy[entering], outside[entering])
            reached[entering] = (
                previous[entering] + step[entering, None] * direction[entering]
            )
            within[entering] = crystal_has(reached[entering], which[entering])

        # In the crystal: deposit, scatter, slow down, and draw the next step.
        a = np.flatnonzero(within & ~outside & ~finished)
        lost = step[a] * power[a]
        deposit(reached[a], which[a], lost, False)
        previous[a] = reached[a]
        direction[a] = physics.deflect(direction[a], energy[a], outside[a])
        energy[a] -= lost
        # Only those that go on need a step; the rest stop below.
        a = a[energy[a] >= STOPPED_KEV]
        step[a] = physics.step(energy[a], outside[a])

        # Leaving the crystal: the distance to its surface, deposited where it
        # left from -- if that is in the crystal, which a point moved onto a
        # face on the way in need not be.
        b = np.flatnonzero(~within & ~outside & ~finished)
        to_edge = exit_distance(
            media.mesh, previous[b], direction[b], placement.take(which[b])
        )
        lost = to_edge * power[b]
        energy[b] -= lost
        there = crystal_has(previous[b], which[b])
        deposit(previous[b[there]], which[b[there]], lost[there], False)
        wandering = ~within & outside & ~finished
        if media.surrounding is None:
            exits[b] = True
        else:
            # On into the surrounding, just past the face. With under 0.05 keV
            # left, it leaves that where it left from, as with no surrounding
            # -- negative, if the way out took more than it had.
            spent = energy[b] < STOPPED_KEV
            kept = spent & there
            deposit(previous[b[kept]], which[b[kept]], energy[b[kept]], False)
            energy[b[spent]] = 0.0
            exits[b[spent]] = True
            previous[b] += (to_edge + (to_edge / 1000))[:, None] * direction[b]
            outside[b] = True
            on = b[~spent]
            step[on] = physics.step(energy[on], outside[on])

        # In the surrounding: scatter and slow, depositing nothing, until it
        # leaves the surrounding.
        c = np.flatnonzero(wandering)
        if len(c):
            still_in = np.zeros(len(c), dtype=bool)
            for index in np.unique(box[c]):
                group = np.flatnonzero(box[c] == index)
                still_in[group] = inside(
                    media.boxes[index],
                    reached[c[group]],
                    placement.take(which[c[group]]),
                )
            finished[c[~still_in]] = True
            c = c[still_in]
            lost = step[c] * power[c]
            previous[c] = reached[c]
            direction[c] = physics.deflect(direction[c], energy[c], outside[c])
            energy[c] -= lost
            c = c[energy[c] >= STOPPED_KEV]
            step[c] = physics.step(energy[c], outside[c])

        # Stopped: what is left, where it stopped, if that is in the crystal.
        stopped = np.flatnonzero((energy < STOPPED_KEV) & ~finished)
        if len(stopped):
            there = stopped[crystal_has(previous[stopped], which[stopped])]
            deposit(previous[there], which[there], energy[there], False)
            finished[stopped] = True

        keep = ~finished & ~exits
        which, previous, direction, energy, outside, box, step = (
            which[keep], previous[keep], direction[keep], energy[keep],
            outside[keep], box[keep], step[keep],
        )  # fmt: skip
    return passes


def _first_above(cdf: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """``getElasticElement``: the first element whose share passes the draw.

    The last, should rounding leave none; the Java would fail.
    """
    above = cdf > uniform[None, :]
    found = above.any(axis=0)
    return np.where(found, above.argmax(axis=0), cdf.shape[0] - 1)
