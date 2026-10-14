"""RADDOSE-3D's simple Monte Carlo, ``Subprogram MONTECARLO``.

Photons cross the crystal; those that interact free a photoelectron or a
Compton electron, or scatter elastically and are counted. Each electron is
followed by its elastic scattering and its stopping power until it stops
or leaves (:mod:`.transport`). The atom a photoelectron leaves behind
relaxes where it is: its binding energy is deposited there, less the energy
of a fluorescence photon if it emits one.

Every run is simulated at once, in batches of photons, and tallied by run:
the runs share one generator, whose draws are independent whichever run
they fall to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.geometry import triangle_planes
from pyraddose_3d.backend.montecarlo import tables
from pyraddose_3d.backend.montecarlo.geometry import (
    Placement,
    entry_depth,
    inside,
    voxel_indices,
)
from pyraddose_3d.backend.montecarlo.photons import (
    BeamShape,
    beam_positions,
    compton_electrons,
    first_above,
    photoelectron_directions,
    photon_coefficients,
    shells,
)
from pyraddose_3d.backend.montecarlo.transport import track

if TYPE_CHECKING:
    from pyraddose_3d.backend.composition import Composition
    from pyraddose_3d.backend.geometry import Mesh
    from pyraddose_3d.backend.microed import ElectronMaterial

__all__ = ["SimpleRun", "SimpleSetup", "simulate_simple"]

#: Photons per batch.
_BATCH = 250_000


@dataclass(frozen=True)
class SimpleSetup:
    """Everything a simple Monte Carlo run reads.

    Attributes
    ----------
    mesh : Mesh
        The crystal, um, with its P and L angles applied.
    pixels_per_micron : float
        Its voxel resolution.
    composition : Composition
        The unit cell's contents.
    density : float
        g/cm^3.
    material : ElectronMaterial
        The same, for the electron physics.
    beam : BeamShape
        The beam's profile and aperture.
    photon_energy : float
        keV.
    flux : float
        Photons per second.
    exposure : float
        The wedge's exposure time, s.
    start_angle, end_angle : float
        The wedge's angles, radians.
    angular_resolution : float
        Radians, as ``Wedge`` keeps it.
    start_offset : tuple[float, float, float]
        The wedge's start position, um.
    translation_per_radian : tuple[float, float, float]
        um per radian of rotation.
    photons : int
        ``SimPhotons``: photons per run.
    runs : int
        ``Runs``.
    rotated_polarisation : bool
        Whether the goniometer and the polarisation are at right angles.
    """

    mesh: Mesh
    pixels_per_micron: float
    composition: Composition
    density: float
    material: ElectronMaterial
    beam: BeamShape
    photon_energy: float
    flux: float
    exposure: float
    start_angle: float
    end_angle: float
    angular_resolution: float
    start_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_per_radian: tuple[float, float, float] = (0.0, 0.0, 0.0)
    photons: int = 1_000_000
    runs: int = 1
    rotated_polarisation: bool = False


@dataclass
class SimpleRun:
    """One run's tallies and what ``processDose`` makes of them.

    Attributes
    ----------
    rd3d_adwc, rd3d_ader : float
        RADDOSE-3D-style dose -- the energy of every absorbed photon and
        Compton electron, deposited where it was freed -- over the whole
        crystal and over its exposed region, MGy.
    mc_adwc, mc_ader : float
        The Monte Carlo's dose, the same two ways, MGy.
    elastic : float
        Elastically scattered photons, scaled to the real photon count.
    ions_per_atom : float
        Ionisations per atom in the exposed region.
    """

    rd3d_adwc: float = 0.0
    rd3d_ader: float = 0.0
    mc_adwc: float = 0.0
    mc_ader: float = 0.0
    elastic: float = 0.0
    ions_per_atom: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass
class _Tallies:
    """What the runs accumulate, in keV and counts."""

    dose: np.ndarray
    raddose: np.ndarray
    raddose_compton: np.ndarray
    elastic: np.ndarray
    ionisations: np.ndarray
    electrons: np.ndarray
    passes: int = 0


def _placements(setup: SimpleSetup, index: np.ndarray) -> Placement:
    """How the crystal stands for each photon: its place in its run."""
    start, end = setup.start_angle, setup.end_angle
    if abs(start - end) < setup.angular_resolution or end == 0:
        angle = np.zeros(len(index))
    else:
        sign = -1.0 if end < start else 1.0
        progress = index / setup.photons
        # Always forwards from the start, whichever way the wedge runs.
        angle = start + (progress * (sign * (end - start)))
    # "convert to 0-2pi", truncating towards zero as Java's cast does.
    times_over = np.trunc(angle / (2 * math.pi))
    angle = angle - (times_over * 2 * math.pi)
    rate = np.array(setup.translation_per_radian)
    offset = (
        np.array(setup.start_offset)[None, :] + rate[None, :] * (angle - start)[:, None]
    )
    return Placement(angle, offset)


def _relaxation(
    z: np.ndarray,
    shell: np.ndarray,
    binding: np.ndarray,
    yields: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """``produceAugerElectron`` in simple mode: the energy left at the atom.

    The binding energy, less a fluorescence photon's energy if the vacancy
    is in an element and shell RADDOSE-3D relaxes and the draw says it
    fluoresces. The transition is drawn from the Auger table and its energy
    read from the fluorescence table at the same row; reproduced.
    """
    left = binding.copy()
    eligible = (
        np.isin(
            z, (6, 7, 8, 11, 12, 14, 16, 17, 19, 20, 25, 26, 27, 28, 29, 30, 33, 34)
        )
    ) & (
        (shell == 0)
        | ((shell < 2) & (z == 11))
        | ((shell < 3) & (z == 12))
        | ((shell < 4) & (z <= 20) & (z > 12))
        | ((shell < 7) & (z >= 25) & (z < 31))
        | ((shell < 9) & (z > 32) & (z < 35))
    )
    chosen = np.flatnonzero(eligible)
    if not len(chosen):
        return left
    fluoresces = ~(rng.random(len(chosen)) > yields[chosen])
    auger = tables.auger_transitions()
    fluorescence = tables.fluorescence_transitions()
    for i in chosen[fluoresces]:
        key = (int(z[i]), int(shell[i]))
        cumulative = auger[key].cumulative
        u = rng.random()
        hits = np.flatnonzero(u < cumulative)
        index = int(hits[0]) if len(hits) else 0
        left[i] = binding[i] - fluorescence[key].energy_kev[index]
    return left


def _grid(setup: SimpleSetup) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The voxel grid's corner and size: ``getMaxPixelCoordinates``."""
    vertices = np.asarray(setup.mesh.vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    nx, ny, nz = (
        math.floor((high[i] - low[i]) * setup.pixels_per_micron + 0.5) + 1
        for i in range(3)
    )
    return low, (nx, ny, nz)


@dataclass
class _DoseGrid:
    """``doseSimple``: energy deposited per run and voxel, keV."""

    minimum: np.ndarray
    size: tuple[int, int, int]
    pixels_per_micron: float
    dose: np.ndarray

    def add(
        self,
        points: np.ndarray,
        placement: Placement,
        run: np.ndarray,
        energy: np.ndarray,
    ) -> None:
        """Deposit ``energy`` in the voxels the points are in, by run."""
        ijk = voxel_indices(points, placement, self.minimum, self.pixels_per_micron)
        ijk = np.minimum(ijk, np.array(self.size) - 1)
        nx, ny, nz = self.size
        flat = ((run * nx + ijk[:, 0]) * ny + ijk[:, 1]) * nz + ijk[:, 2]
        np.add.at(self.dose, flat, energy)


@dataclass(frozen=True)
class _Owned:
    """Deposits by electrons, each knowing its photon's placement and run."""

    grid: _DoseGrid
    placement: Placement
    run: np.ndarray

    def __call__(
        self, points: np.ndarray, which: np.ndarray, energy: np.ndarray
    ) -> None:
        """Deposit for the electrons ``which``."""
        self.grid.add(points, self.placement.take(which), self.run[which], energy)


def simulate_simple(setup: SimpleSetup, rng: np.random.Generator) -> list[SimpleRun]:
    """Run ``Runs`` simple Monte Carlo runs.

    Parameters
    ----------
    setup : SimpleSetup
        The experiment.
    rng : numpy.random.Generator
        The random numbers.

    Returns
    -------
    list[SimpleRun]
        One per run.
    """
    runs, photons = setup.runs, setup.photons
    minimum, size = _grid(setup)
    voxels = size[0] * size[1] * size[2]
    tallies = _Tallies(
        dose=np.zeros(runs * voxels),
        raddose=np.zeros(runs),
        raddose_compton=np.zeros(runs),
        elastic=np.zeros(runs),
        ionisations=np.zeros(runs),
        electrons=np.zeros(runs),
    )
    coefficients = photon_coefficients(setup.composition, setup.photon_energy)
    z_of = np.array([e.atomic_number for e in setup.composition.elements])
    binding_of = np.stack([shells(int(z)).binding for z in z_of])
    yield_of = np.stack([shells(int(z)).fluorescence_yield for z in z_of])
    vertices = np.asarray(setup.mesh.vertices)
    reach = float(np.max(np.linalg.norm(vertices[:, None] - vertices[None], axis=-1)))

    grid = _DoseGrid(minimum, size, setup.pixels_per_micron, tallies.dose)

    total = runs * photons
    for first in range(0, total, _BATCH):
        index = np.arange(first, min(first + _BATCH, total))
        run = index // photons
        placement = _placements(setup, index % photons)
        n = len(index)

        x, y = beam_positions(rng, setup.beam, n)
        depth = entry_depth(setup.mesh, x, y, placement)
        travel = -coefficients.total_mean_free_path_nm * np.log(rng.random(n))
        # Beyond the crystal's widest extent a point cannot be inside it.
        candidate = np.flatnonzero((depth != 0.0) & (travel <= reach * 1000))
        points = np.stack(
            [x[candidate], y[candidate], depth[candidate] + travel[candidate]], axis=1
        )
        hit = inside(setup.mesh, points, placement.take(candidate))
        candidate, points = candidate[hit], points[hit]

        elastic = rng.random(len(candidate)) < coefficients.elastic_chance
        np.add.at(tallies.elastic, run[candidate[elastic]], 1)
        candidate, points = candidate[~elastic], points[~elastic]
        compton = rng.random(len(candidate)) < coefficients.compton_chance

        # Photoelectric absorption.
        pe, pe_points = candidate[~compton], points[~compton]
        element = first_above(
            coefficients.photoelectric_cdf, rng.random(len(pe)), len(z_of) - 1
        )
        u = rng.random(len(pe))
        rows = coefficients.shell_cdf[element]
        above = rows > u[:, None]
        shell = np.where(above.any(axis=1), above.argmax(axis=1), 0)
        binding = binding_of[element, shell]
        pe_place = placement.take(pe)
        pe_direction = photoelectron_directions(
            rng, shell == 0, pe_place.angle, setup.rotated_polarisation
        )
        np.add.at(tallies.raddose, run[pe], setup.photon_energy)
        np.add.at(tallies.ionisations, run[pe], 1)
        left = _relaxation(z_of[element], shell, binding, yield_of[element, shell], rng)
        grid.add(pe_points, pe_place, run[pe], left)

        # Compton scattering.
        cs, cs_points = candidate[compton], points[compton]
        cs_energy, cs_direction = compton_electrons(rng, setup.photon_energy, len(cs))
        np.add.at(tallies.raddose_compton, run[cs], cs_energy)
        np.add.at(tallies.ionisations, run[cs], 1)
        rng.random(len(cs))  # getIonisedElement: only the per-element tallies read it

        start = np.concatenate([pe_points, cs_points])
        direction = np.concatenate([pe_direction, cs_direction])
        energy = np.concatenate([setup.photon_energy - binding, cs_energy])
        owner = np.concatenate([pe, cs])
        electron_place = placement.take(owner)
        electron_run = run[owner]
        np.add.at(tallies.electrons, electron_run, 1)

        passes = track(
            rng,
            setup.material,
            setup.mesh,
            start,
            direction,
            energy,
            electron_place,
            _Owned(grid, electron_place, electron_run),
        )
        tallies.passes = max(tallies.passes, passes)

    return _process(setup, tallies, minimum, size)


def _sample_volume(mesh: Mesh) -> float:
    """``getSampleVolume``: the mesh's volume from its triangles' planes, um^3.

    Each triangle's area, by Heron's formula, times its plane's distance from
    the origin, over three: exact for a convex crystal around the origin.
    """
    _, offsets = triangle_planes(mesh)
    corners = np.asarray(mesh.corners())
    a = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
    b = np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1)
    c = np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1)
    p = (a + b + c) / 2
    area = np.power(p * (p - a) * (p - b) * (p - c), 0.5)
    return float(np.sum(np.abs((np.asarray(offsets) * area) / 3)))


def _process(
    setup: SimpleSetup,
    tallies: _Tallies,
    minimum: np.ndarray,
    size: tuple[int, int, int],
) -> list[SimpleRun]:
    """``processDose`` for the simple Monte Carlo's quantities."""
    mean_energy_joules = setup.photon_energy * KEV_TO_JOULES
    pulse_energy = (setup.flux * setup.exposure) * (setup.photon_energy * KEV_TO_JOULES)
    real_photons = pulse_energy / mean_energy_joules
    scale = real_photons / setup.photons

    volume_um3 = _sample_volume(setup.mesh)
    volume = volume_um3 * 1e-12
    mass = (setup.density * volume) / 1000
    counts = np.asarray(setup.composition.counts, dtype=float)
    total_atoms = (
        float(np.sum(counts)) * (volume * 1e24) / float(setup.composition.cell_volume)
    )
    vertices = np.asarray(setup.mesh.vertices)
    z_dimension = 1000 * float(vertices[:, 2].max() - vertices[:, 2].min())
    exposed_volume = min(setup.beam.exposed_area_nm2() * z_dimension * 1e-21, volume)
    exposed_mass = (setup.density * exposed_volume) / 1000
    fraction = exposed_volume / volume

    a = np.arange(size[0])
    b = np.arange(size[1])
    x = ((a / setup.pixels_per_micron) + minimum[0]) * 1000
    y = ((b / setup.pixels_per_micron) + minimum[1]) * 1000
    exposed_xy = setup.beam.exposes(np.repeat(x, size[1]), np.tile(y, size[0])).reshape(
        size[0], size[1]
    )

    dose = tallies.dose.reshape(setup.runs, *size)
    results = []
    for r in range(setup.runs):
        joules = dose[r] * scale * KEV_TO_JOULES
        rd3d = ((tallies.raddose[r] * scale * KEV_TO_JOULES) / mass) / 1e6 + (
            (tallies.raddose_compton[r] * scale * KEV_TO_JOULES) / mass
        ) / 1e6
        ions = math.floor(tallies.ionisations[r] * scale + 0.5)
        results.append(
            SimpleRun(
                rd3d_adwc=rd3d,
                rd3d_ader=(rd3d * mass) / exposed_mass,
                mc_adwc=(float(joules.sum()) / mass) / 1e6,
                mc_ader=(float(joules[exposed_xy].sum()) / exposed_mass) / 1e6,
                elastic=float(tallies.elastic[r]) * scale,
                ions_per_atom=(ions / total_atoms) / fraction,
                diagnostics={
                    "electrons": float(tallies.electrons[r]),
                    "passes": float(tallies.passes),
                },
            )
        )
    return results
