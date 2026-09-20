"""The exposure loop: how much dose each voxel receives.

Ports the deterministic path through ``Crystal.expose`` and
``Crystal.exposeAngle`` -- the MX calculation, with photoelectron and
fluorescence escape off. Those are Monte Carlo and are not ported.

What happens, per rotation angle
--------------------------------
The crystal turns; the beam does not. For each voxel still inside the beam:

1. Translate and rotate it into the beam's frame.
2. Read the unattenuated beam intensity there.
3. Find how much crystal the beam already crossed to reach it
   (:func:`~pyraddose_3d.backend.geometry.find_depth`) and attenuate by
   Beer-Lambert over that path.
4. Convert the surviving fluence into absorbed dose, elastic yield and the
   Compton contribution, and add them to the running totals.

Units
-----
Dose in MGy, fluence in joules, elastic yield in photons, coefficients in
inverse micrometres, density in g/cm^3, angles in radians.

Differentiability
-----------------
This is the function the port exists for. Dose is differentiable in beam
flux, photon energy, beam widths, collimation, exposure time, crystal
dimensions and rotation angle -- everything a data-collection strategy can
choose. The gradient is piecewise, inheriting the steps at collimation edges
and crystal silhouettes from the beam and geometry modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyraddose_3d.backend._array import array_namespace, promote
from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.geometry import Mesh, find_depth

if TYPE_CHECKING:
    from pyraddose_3d.backend.coefficients import MaterialCoefficients
    from pyraddose_3d.backend.voxels import VoxelGrid

__all__ = ["ExposureResult", "expose", "exposure_angles"]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""

#: Electron rest mass energy, as ``Crystal.exposeAngle`` spells it. Note the
#: speed of light is taken as 3e8 m/s rather than 299792458, which is a
#: 0.14% approximation in the Compton electron energy. Reproduced.
_ELECTRON_MASS_KG = 9.10938356e-31
_SPEED_OF_LIGHT_SQUARED = 3e8 * 3e8
_ELECTRON_REST_ENERGY_J = _ELECTRON_MASS_KG * _SPEED_OF_LIGHT_SQUARED

#: Exposures short enough to count as still are sampled this many times, so
#: that a still and a rotation of the same duration deposit the same energy.
STATIC_EXPOSURE_STEPS = 100


@dataclass(frozen=True)
class ExposureResult:
    """Per-voxel totals accumulated over a whole wedge.

    Attributes
    ----------
    dose : Array
        ``(N,)`` absorbed dose in MGy.
    fluence : Array
        ``(N,)`` energy incident on the voxel, in joules.
    elastic : Array
        ``(N,)`` elastically scattered photons.
    occupied : Array
        ``(N,)`` boolean, which voxels are inside the crystal. Voxels
        outside it are zero throughout, and must be excluded from averages
        rather than averaged in.
    """

    dose: Array
    fluence: Array
    elastic: Array
    occupied: Array

    @property
    def average_dose(self) -> Array:
        """Mean dose over the voxels inside the crystal, in MGy."""
        xp = array_namespace(self.dose)
        inside = xp.sum(xp.where(self.occupied, self.dose, 0.0))
        return inside / xp.sum(self.occupied)

    @property
    def max_dose(self) -> Array:
        """Largest dose in any voxel inside the crystal, in MGy."""
        xp = array_namespace(self.dose)
        return xp.max(xp.where(self.occupied, self.dose, 0.0))


def exposure_angles(
    start: float, end: float, resolution: float
) -> list[float]:
    """The rotation angles a wedge is sampled at.

    Parameters
    ----------
    start, end : float
        First and last angle of the wedge, in radians.
    resolution : float
        Angular step, in radians.

    Returns
    -------
    list[float]
        The angles, in order. A wedge narrower than one step is treated as a
        still and returns the same angle ``STATIC_EXPOSURE_STEPS`` times --
        the total exposure is divided by the number of steps, so repeating
        one angle deposits the same energy as rotating through many.

    Notes
    -----
    Both ends are first reduced by whole turns, using the number of turns in
    ``start``. A wedge from 10 to 370 degrees therefore becomes 10 to 370,
    not 10 to 10: only complete revolutions already in ``start`` are removed.
    """
    turns = int(start / (2 * math.pi))
    begin = start - 2 * math.pi * turns
    finish = end - 2 * math.pi * turns

    if abs(begin - finish) < resolution:
        return [begin] * STATIC_EXPOSURE_STEPS

    sign = 1 if finish >= begin else -1
    count = sign * int((finish - begin) / resolution + 1)
    return [begin + sign * step * resolution for step in range(count)]


def _fraction_absorbed(coefficient: Array, pixels_per_micron: float) -> Array:
    """Fraction of a beam absorbed crossing one voxel."""
    xp = array_namespace(coefficient)
    return -xp.expm1(-coefficient / pixels_per_micron)


def expose(
    mesh: Mesh,
    grid: VoxelGrid,
    occupied: Array,
    beam: Any,
    coefficients: MaterialCoefficients,
    density: Array,
    angles: Array,
    exposure_seconds: Array,
    *,
    translations: Array | None = None,
    off_axis: Array = 0.0,
) -> ExposureResult:
    """Accumulate dose over a wedge.

    Parameters
    ----------
    mesh : Mesh
        The crystal, with ``AngleP`` and ``AngleL`` already applied and in
        the same frame as ``grid``.
    grid : VoxelGrid
        The voxel grid sampling it.
    occupied : Array
        ``(N,)`` boolean, which voxels are inside the crystal.
    beam : Any
        Anything with an ``intensity(x, y, off_axis)`` method --
        :class:`~pyraddose_3d.backend.beam.GaussianBeam` or
        :class:`~pyraddose_3d.backend.beam.TophatBeam`.
    coefficients : MaterialCoefficients
        Absorption, attenuation, elastic and inelastic, in inverse
        micrometres.
    density : Array
        Crystal density in g/cm^3.
    angles : Array
        Rotation angles in radians, from :func:`exposure_angles`.
    exposure_seconds : Array
        Total exposure time for the whole wedge. It is divided between the
        angles, so a longer wedge at the same total time does not deposit
        more energy.
    translations : Array, optional
        ``(A, 3)`` displacement per angle: the wedge start position plus
        whatever its per-degree translation has accumulated. Zero if
        omitted.
    off_axis : Array, optional
        Rotation axis offset from the beam centre, in micrometres.

    Returns
    -------
    ExposureResult
        Per-voxel dose, fluence and elastic yield.
    """
    xp = array_namespace(grid.origin, beam.energy, beam.flux, density)
    coordinates = grid.coordinates()
    angle_count = len(angles)

    pixels_per_micron = grid.pixels_per_micron
    photon_energy = beam.energy

    # Per-voxel conversion factors. All constant across the wedge, so they
    # are formed once rather than per angle as the Java does.
    voxel_mass_term = 1e-15 * (pixels_per_micron**-3 * density)
    fluence_to_dose = (
        _fraction_absorbed(coefficients.absorption, pixels_per_micron)
        / voxel_mass_term
        * 1e-6
    )
    fluence_to_compton_dose = (
        _fraction_absorbed(coefficients.inelastic, pixels_per_micron)
        / voxel_mass_term
        * 1e-6
    )
    fluence_to_elastic = _fraction_absorbed(
        coefficients.elastic, pixels_per_micron
    ) / (photon_energy * KEV_TO_JOULES)

    # Voxel cross-section times the time each angle is exposed for.
    beam_to_fluence = (
        pixels_per_micron**-2 * exposure_seconds / angle_count
    )

    # Compton electrons carry away this fraction of each photon's energy.
    beam_energy_joules = photon_energy * KEV_TO_JOULES
    # Promoted against the values that decide the backend: the ratio is a
    # plain float whenever the photon energy is, even in a torch exposure.
    _, (ratio, *_rest) = promote(
        _ELECTRON_REST_ENERGY_J
        / (2 * beam_energy_joules + _ELECTRON_REST_ENERGY_J),
        grid.origin,
        beam.flux,
        density,
    )
    compton_fraction = 1 - xp.sqrt(ratio)

    dose = xp.zeros_like(coordinates[:, 0])
    fluence = xp.zeros_like(dose)
    elastic = xp.zeros_like(dose)

    for index, angle in enumerate(angles):
        shift = (
            xp.zeros(3, dtype=coordinates.dtype)
            if translations is None
            else translations[index]
        )
        placed = _place(coordinates, shift, angle)

        intensity = beam.intensity(placed[:, 0], placed[:, 1], off_axis)
        depth = find_depth(mesh.rotated(angle, shift), placed)

        # Beer-Lambert over the path already travelled through the crystal.
        surviving = intensity * xp.exp(-coefficients.attenuation * depth)
        voxel_fluence = surviving * beam_to_fluence

        # Voxels outside the crystal receive nothing, whatever the beam is
        # doing there: the Java never visits them.
        voxel_fluence = xp.where(occupied, voxel_fluence, 0.0)

        fluence = fluence + voxel_fluence
        elastic = elastic + fluence_to_elastic * voxel_fluence
        dose = dose + fluence_to_dose * voxel_fluence
        dose = dose + fluence_to_compton_dose * voxel_fluence * compton_fraction

    return ExposureResult(
        dose=dose, fluence=fluence, elastic=elastic, occupied=occupied
    )


def _place(coordinates: Array, shift: Array, angle: Array) -> Array:
    """Translate voxels, then rotate them about the y axis.

    The same transform ``setupDepthFinding`` applies to the mesh, so the two
    stay in step. Translation first: the goniometer moves the crystal in its
    own frame and the whole assembly then turns.
    """
    xp, (coordinates, shift, angle) = promote(coordinates, shift, angle)
    moved = coordinates + shift
    cos = xp.cos(angle)
    sin = xp.sin(angle)
    x, y, z = moved[:, 0], moved[:, 1], moved[:, 2]
    return xp.stack([x * cos + z * sin, y, -x * sin + z * cos], axis=-1)
