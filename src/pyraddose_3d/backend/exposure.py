"""The exposure loop: how much dose each voxel receives.

Ports ``Crystal.expose`` and ``Crystal.exposeAngle`` -- the MX calculation.
Photoelectron and fluorescence escape, when on, replace step 4's local
deposition of the photoelectric dose; they are Monte Carlo, and live in
:mod:`~pyraddose_3d.backend.escape`. A pink beam repeats step 4 for each of
its energies, sharing the angle's exposure between them.

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
and crystal silhouettes from the beam and geometry modules. Escape modelling
has none: it chooses each electron's direction at random.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend._array import array_namespace, promote, to_namespace
from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.geometry import Mesh, find_depth

if TYPE_CHECKING:
    from pyraddose_3d.backend.coefficients import MaterialCoefficients
    from pyraddose_3d.backend.voxels import VoxelGrid

__all__ = ["ExposureResult", "ImageRecord", "expose", "exposure_angles", "no_decay"]

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
class ImageRecord:
    """Per-voxel fields for every image of a wedge, kept only on request.

    RADDOSE-3D's ``VoxelDose``, ``VoxelFluences`` and ``RDE`` outputs are per
    image, so they cannot be reconstructed from the totals an exposure
    otherwise returns. Holding a field per image costs ``A`` times the memory
    of the totals, which is nothing for a test case and real for a large
    crystal, so :func:`expose` records them only when asked.

    Attributes
    ----------
    dose : Array
        ``(A, N)`` cumulative dose in MGy after each image -- the crystal's
        whole history up to that point, not the increment.
    fluence : Array
        ``(A, N)`` fluence each voxel received in that image alone.
    decay : Array
        ``(A, N)`` relative diffraction efficiency each voxel had while it
        diffracted in that image, from the decay model.
    weighted_decay, fluence_total : Array
        ``(A,)`` sums over the image's observations of fluence times decay,
        and of fluence: the fluence-weighted RDE is their ratio.
    minimum_decay : Array
        ``(A,)`` lowest decay any lit voxel had in the image; infinite if
        none was lit.
    observations : Array
        ``(A,)`` lit voxels counted once per observation -- once per sampled
        energy for a pink beam, as RADDOSE-3D counts the exposed volume.

    For a pink beam ``dose``, ``fluence`` and ``decay`` are the image's last
    observation, as in the Java -- so ``fluence`` is the last energy's alone.
    """

    dose: Array
    fluence: Array
    decay: Array
    weighted_decay: Array
    fluence_total: Array
    minimum_decay: Array
    observations: Array


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
    absorbed_energy : Array
        ``(N,)`` energy actually absorbed, in joules. Distinct from
        ``fluence``, which is the energy that arrived: only the absorbed
        part becomes dose.
    image_dwd : Array
        ``(A,)`` diffraction-weighted dose after each image, in MGy. The
        fluence- and decay-weighted mean dose over the voxels that
        contributed to that image.
    occupied : Array
        ``(N,)`` boolean, which voxels are inside the crystal. Voxels
        outside it are zero throughout, and must be excluded from averages
        rather than averaged in.
    images : ImageRecord | None
        Per-image fields, if :func:`expose` was asked to record them.
    last_observed : Array | None
        ``(N,)`` dose each voxel had when it was last lit, to carry into the
        next wedge with ``dose``.
    """

    dose: Array
    fluence: Array
    elastic: Array
    occupied: Array
    absorbed_energy: Array
    image_dwd: Array
    images: ImageRecord | None = None
    last_observed: Array | None = None

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


def exposure_angles(start: float, end: float, resolution: float) -> list[float]:
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


def no_decay(dose: Array) -> float:
    """RADDOSE-3D's ``DDM Simple``: diffraction does not decay with dose.

    Parameters
    ----------
    dose : Array
        Accumulated dose, ignored.

    Returns
    -------
    float
        Always one.
    """
    del dose
    return 1.0


def _fraction_absorbed(coefficient: Array, pixels_per_micron: float) -> Array:
    """Fraction of a beam absorbed crossing one voxel."""
    xp = array_namespace(coefficient)
    return -xp.expm1(-coefficient / pixels_per_micron)


def _beam_settings(beam: Any) -> tuple[Any, ...]:
    """Every value the beam holds, for choosing an array namespace.

    A beam's aperture and its width are as likely to be the thing being
    optimised as its flux, and missing one leaves the namespace on NumPy
    while the intensities coming back are tensors. Read off the dataclass
    rather than listed by hand, so a new field cannot be forgotten; anything
    that is not a dataclass falls back to the two fields every beam has.

    Parameters
    ----------
    beam : Any
        A backend beam.

    Returns
    -------
    tuple[Any, ...]
        Its field values.
    """
    if dataclasses.is_dataclass(beam):
        return tuple(getattr(beam, f.name) for f in dataclasses.fields(beam))
    return (beam.energy, beam.flux)


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
    decay_model: Any = None,
    initial_dose: Array | None = None,
    last_observed: Array | None = None,
    record_images: bool = False,
    energies: Array | None = None,
    escape: Any = None,
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
        :class:`~pyraddose_3d.backend.beam.GaussianBeam`,
        :class:`~pyraddose_3d.backend.beam.TophatBeam` or
        :class:`~pyraddose_3d.backend.beam.ExperimentalBeam`.
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
    decay_model : Any, optional
        Callable mapping accumulated dose to relative diffraction
        efficiency, for the diffraction-weighted dose. Defaults to
        RADDOSE-3D's ``DDM Simple``, which is no decay at all.
    initial_dose : Array, optional
        ``(N,)`` dose the crystal already carries, from earlier wedges.
        Zero if omitted.
    last_observed : Array, optional
        ``(N,)`` dose each voxel had when it was last lit, from earlier
        wedges. Defaults to ``initial_dose``, which it equals unless escape
        moved dose into voxels while they were dark.
    record_images : bool, optional
        Keep the dose, fluence and decay of every voxel after every image,
        as :attr:`ExposureResult.images`. Off by default; see
        :class:`ImageRecord` for why.
    energies : Array, optional
        For a pink beam, the energies it is sampled at, in keV; the fields
        of ``coefficients`` are then arrays with one value per energy. Each
        angle's exposure is shared equally between them. A monochromatic
        beam, the default, is sampled at ``beam.energy`` alone.
    escape : Any, optional
        An :class:`~pyraddose_3d.backend.escape.Escape`, to model
        photoelectron and fluorescence escape. NumPy only.

    Returns
    -------
    ExposureResult
        Per-voxel dose, fluence and elastic yield.

    Notes
    -----
    Dose is the only quantity that carries between wedges. RADDOSE-3D holds
    it on the crystal, so a second wedge starts where the first left off and
    its summary describes the crystal's whole history; the elastic yield and
    absorbed energy live on the per-wedge observer and start again at zero.
    Reproduced by returning a cumulative ``dose`` and per-wedge everything
    else, so a caller need only feed one back in.
    """
    if decay_model is None:
        decay_model = no_decay

    # Every operand that could be a tensor has to be offered here, not just
    # the obvious ones: miss one and the namespace comes out NumPy while the
    # values coming back from the beam are tensors, and the first `where`
    # that mixes them fails.
    xp = array_namespace(
        grid.origin,
        *_beam_settings(beam),
        density,
        exposure_seconds,
        initial_dose,
        off_axis,
        translations,
        *angles,
    )
    # The crystal is usually fixed NumPy geometry while the beam and wedge
    # settings being optimised are tensors, so the grid has to be brought
    # into whichever namespace won before anything indexes into it.
    coordinates = to_namespace(xp, grid.coordinates())
    occupied = to_namespace(xp, occupied)
    angle_count = len(angles)

    pixels_per_micron = grid.pixels_per_micron
    energies = [beam.energy] if energies is None else list(energies)
    samples = len(energies)

    def at(value: Array, k: int) -> Array:
        """A coefficient at the k-th energy: indexed for a pink beam."""
        return value if samples == 1 else value[k]

    # Voxel cross-section times the time each angle is exposed for, shared
    # between the energies a pink beam is sampled at.
    beam_to_fluence = pixels_per_micron**-2 * exposure_seconds / (angle_count * samples)
    voxel_mass_term = 1e-15 * (pixels_per_micron**-3 * density)

    steps = []
    for k, photon_energy in enumerate(energies):
        absorption = at(coefficients.absorption, k)
        fluence_to_dose = (
            _fraction_absorbed(absorption, pixels_per_micron) / voxel_mass_term * 1e-6
        )
        fluence_to_compton_dose = (
            _fraction_absorbed(at(coefficients.inelastic, k), pixels_per_micron)
            / voxel_mass_term
            * 1e-6
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
        # Dispatched on the ratio itself rather than on `xp`: it stays a plain
        # float whenever the photon energy is one, even in a torch exposure,
        # and forcing it into a tensor here would take torch's default float32
        # and silently halve the precision of everything downstream.
        steps.append(
            _EnergyStep(
                energy_joules=beam_energy_joules,
                attenuation=at(coefficients.attenuation, k),
                fluence_to_dose=fluence_to_dose,
                fluence_to_compton_dose=fluence_to_compton_dose,
                fluence_to_elastic=_fraction_absorbed(
                    at(coefficients.elastic, k), pixels_per_micron
                )
                / beam_energy_joules,
                compton_fraction=1 - array_namespace(ratio).sqrt(ratio),
                # Energy absorbed per unit fluence crossing one voxel: the
                # numerator of fluence_to_dose, before the mass conversion.
                energy_per_fluence=_fraction_absorbed(absorption, pixels_per_micron),
            )
        )

    dose = xp.zeros_like(coordinates[:, 0]) if initial_dose is None else initial_dose
    # The dose each voxel had when the diffraction-weighted dose last saw it
    # -- ``voxImageDoseLast``. Without escape it is always the current dose,
    # since a voxel only gains dose while lit; with escape a voxel can gain
    # dose from its neighbours while dark, and the Java's midpoint then
    # spans everything since it was last lit.
    last_seen = dose if last_observed is None else last_observed
    fluence = xp.zeros_like(coordinates[:, 0])
    elastic = xp.zeros_like(fluence)
    absorbed_energy = xp.zeros_like(fluence)
    image_dwd = []
    recorded = _Recorder() if record_images else None

    if escape is not None and xp is not np:
        raise NotImplementedError(
            "escape modelling chooses its photoelectron directions at random, "
            "so it has no gradient; run it without tensor overrides"
        )

    for index, angle in enumerate(angles):
        shift = (
            xp.zeros(3, dtype=coordinates.dtype)
            if translations is None
            else translations[index]
        )
        placed = _place(coordinates, shift, angle)
        rotated = mesh.rotated(angle, shift)
        intensity = beam.intensity(placed[:, 0], placed[:, 1], off_axis)
        depth = find_depth(rotated, placed)

        numerator: Array = 0.0
        denominator: Array = 0.0
        if recorded is not None:
            recorded.start_image()
        for sample, step in enumerate(steps):
            # Beer-Lambert over the path already travelled through the
            # crystal. Voxels outside it receive nothing, whatever the beam
            # is doing there: the Java never visits them.
            voxel_fluence = (
                intensity * xp.exp(-step.attenuation * depth) * beam_to_fluence
            )
            voxel_fluence = xp.where(occupied, voxel_fluence, 0.0)
            photoelectric_dose = step.fluence_to_dose * voxel_fluence
            compton_dose = (
                step.fluence_to_compton_dose * voxel_fluence * step.compton_fraction
            )

            if escape is None:
                dose = dose + photoelectric_dose
            else:
                dose = escape.expose_step(
                    dose=dose,
                    lit=voxel_fluence > 0,
                    photoelectric_dose=photoelectric_dose,
                    photons=voxel_fluence / step.energy_joules,
                    fluence_to_dose=step.fluence_to_dose,
                    angle=float(angle),
                    rotated=rotated,
                    shift=shift,
                    beam=beam,
                    off_axis=off_axis,
                    attenuation=step.attenuation,
                    exposure_seconds=float(exposure_seconds),
                    angle_count=angle_count,
                    sample=sample,
                )
            dose = dose + compton_dose

            # The diffraction-weighted dose for this image: the dose each
            # voxel was at while it diffracted, weighted by how much it
            # diffracted. The Java takes the midpoint of the dose across the
            # image for the weighting, but the *end* of it for the decay --
            # reproduced.
            decay = decay_model(dose + photoelectric_dose / 2)
            contributing = voxel_fluence > 0
            weight = xp.where(contributing, voxel_fluence * decay, 0.0)
            midpoint = last_seen + (dose - last_seen) / 2
            numerator = numerator + xp.sum(
                xp.where(contributing, midpoint * weight, 0.0)
            )
            denominator = denominator + xp.sum(weight)
            last_seen = xp.where(contributing, dose, last_seen)

            if recorded is not None:
                recorded.observe(dose, voxel_fluence, decay * xp.ones_like(dose), xp)

            fluence = fluence + voxel_fluence
            elastic = elastic + step.fluence_to_elastic * voxel_fluence
            # Both channels deposit energy, and both count towards the
            # absorbed total used for dose inefficiency.
            absorbed_energy = (
                absorbed_energy
                + step.energy_per_fluence * voxel_fluence * (1 + step.compton_fraction)
            )

        # An image in which nothing diffracted contributes nothing, and the
        # division is avoided rather than guarded after the fact: reading
        # the denominator back as a Python float would break the gradient.
        empty = denominator == 0
        image_dwd.append(
            xp.where(empty, 0.0, numerator / xp.where(empty, 1.0, denominator))
        )

    return ExposureResult(
        dose=dose,
        fluence=fluence,
        elastic=elastic,
        occupied=occupied,
        absorbed_energy=absorbed_energy,
        image_dwd=xp.stack(image_dwd) if image_dwd else xp.zeros_like(fluence[:0]),
        images=(
            recorded.result(xp) if recorded is not None and len(angles) > 0 else None
        ),
        last_observed=last_seen,
    )


@dataclass(frozen=True)
class _EnergyStep:
    """The conversion factors for one energy a beam is sampled at."""

    energy_joules: Array
    attenuation: Array
    fluence_to_dose: Array
    fluence_to_compton_dose: Array
    fluence_to_elastic: Array
    compton_fraction: Array
    energy_per_fluence: Array


class _Recorder:
    """Builds an :class:`ImageRecord`, as ``ExposureSummary`` observes.

    A pink beam observes each voxel once per sampled energy within an
    image, and RADDOSE-3D's per-voxel images keep only the last of those --
    so ``fluence`` is the last energy's, a 200th of the image's. Reproduced;
    the per-image aggregates, which the Java sums over every observation,
    are what the RDE uses.
    """

    def __init__(self) -> None:
        self.dose: list[Array] = []
        self.fluence: list[Array] = []
        self.decay: list[Array] = []
        self.weighted: list[Array] = []
        self.fluence_total: list[Array] = []
        self.minimum: list[Array] = []
        self.observations: list[Array] = []

    def start_image(self) -> None:
        for column in (
            self.dose, self.fluence, self.decay, self.weighted,
            self.fluence_total, self.minimum, self.observations,
        ):  # fmt: skip
            column.append(None)

    def observe(self, dose: Array, fluence: Array, decay: Array, xp: Any) -> None:
        lit = fluence > 0
        self.dose[-1] = dose
        self.fluence[-1] = fluence
        self.decay[-1] = decay
        weighted = xp.sum(xp.where(lit, fluence * decay, 0.0))
        total = xp.sum(fluence)
        smallest = xp.min(xp.where(lit, decay, math.inf))
        count = xp.sum(xp.where(lit, 1, 0))
        if self.weighted[-1] is None:
            self.weighted[-1], self.fluence_total[-1] = weighted, total
            self.minimum[-1], self.observations[-1] = smallest, count
        else:
            self.weighted[-1] = self.weighted[-1] + weighted
            self.fluence_total[-1] = self.fluence_total[-1] + total
            self.minimum[-1] = xp.minimum(self.minimum[-1], smallest)
            self.observations[-1] = self.observations[-1] + count

    def result(self, xp: Any) -> ImageRecord:
        return ImageRecord(
            dose=xp.stack(self.dose),
            fluence=xp.stack(self.fluence),
            decay=xp.stack(self.decay),
            weighted_decay=xp.stack(self.weighted),
            fluence_total=xp.stack(self.fluence_total),
            minimum_decay=xp.stack(self.minimum),
            observations=xp.stack(self.observations),
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
