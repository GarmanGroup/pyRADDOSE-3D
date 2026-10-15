"""RADDOSE-3D's XFEL subprogram, ``Subprogram XFEL``.

The GOS model of ``Subprogram GOS`` (:mod:`.gos_transport`), followed in time
through one pulse. The pulse's photons arrive evenly over its length, in the
order they are simulated, and cross the crystal at the speed of light; each
electron's steps take their flight time, and each relaxation its lifetime.
Nothing is followed after ``lastTime``, when the last photon has left the
crystal, and the dose and ionisations the run reports are those each voxel
holds a time bin before the last photon has crossed it.

As ``XFEL.java`` has it, which differs from ``MC.java`` (``UPSTREAM_FIXES.md``):

- photons start at the front of the crystal's bounding box, not of the
  crystal, and the crystal does not rotate;
- every K-shell photoelectron leaves along the polarisation's dipole
  distribution, not three in four;
- a fluorescence photon carries its energy away;
- the crystal's volume and mass are its bounding box's;
- the elastic count behind the diffraction efficiency is not the simulated
  one, but the share of the real photons that the elastic coefficient, at
  the last photon's energy, scatters in the crystal's depth.

Only the XFEL subprogram's own photons and electrons are ported: without a
surrounding, which has its own model there, and without a rotation or
offset, which it applies to the voxels but not to the crystal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.montecarlo.geometry import Placement, inside, voxel_indices
from pyraddose_3d.backend.montecarlo.gos_transport import (
    Clock,
    GOSMedium,
    gos_medium,
    track_gos,
)
from pyraddose_3d.backend.montecarlo.photons import (
    OUTER_SHELL,
    BeamShape,
    beam_positions,
    choose,
    compton_electrons,
    photon_coefficients,
)
from pyraddose_3d.backend.montecarlo.simple import (
    SimpleRun,
    SimpleSetup,
    _Grid,
    _grid,
    _photoabsorb,
    _pulse,
)

__all__ = ["simulate_xfel"]

#: Photons per batch.
_BATCH = 250_000
#: Electrons followed together, at most.
_TRACKED = 200_000

_SPEED_OF_LIGHT = 299792458.0


def _light(distance_nm: np.ndarray | float) -> np.ndarray:
    """How long light takes to travel ``distance_nm``, fs."""
    return np.asarray((1 / _SPEED_OF_LIGHT) * (np.asarray(distance_nm) / 1e9) * 1e15)


@dataclass(frozen=True)
class _Pulse:
    """The pulse's timing, fs.

    Attributes
    ----------
    length, bin : float
        ``PULSE_LENGTH`` and ``PULSE_BIN_LENGTH``.
    last : float
        ``lastTime``: when the last photon leaves the crystal.
    cutoff : numpy.ndarray
        By voxel layer in z: ``lastTimeVox`` less a bin. A deposit counts in
        the run's dose if its bin starts before this, an ionisation if it
        happens before it.
    """

    length: float
    bin: float
    last: float
    cutoff: np.ndarray


def _timing(setup: SimpleSetup, layers: int, depth_nm: float) -> _Pulse:
    length, bin_ = _pulse(setup.exposure)
    last = float(_light(depth_nm)) + length
    layer_time = np.array(
        [
            ((1 / _SPEED_OF_LIGHT) * ((depth_nm / 1e9) / layers) * (k + 1) * 1e15)
            + length
            for k in range(layers)
        ]
    )
    return _Pulse(length, bin_, last, layer_time - (1 * bin_))


class _Tallies:
    """What the runs accumulate: keV and counts, by run and voxel."""

    def __init__(self, runs: int, voxels: int, elements: int) -> None:
        self.resolved = np.zeros(runs * voxels)  # every bin of voxelEnergyvResolved
        self.resolved_cut = np.zeros(runs * voxels)  # the bins processDose keeps
        self.voxel_elastic = np.zeros(runs * voxels)
        self.raddose = np.zeros(runs)
        self.raddose_compton = np.zeros(runs)
        self.elastic = np.zeros(runs)
        self.ionisations = np.zeros(runs)  # totalIonisationEventsvResolved
        self.element_ions = np.zeros((runs, elements))
        self.element_ions_exposed = np.zeros((runs, elements))
        self.electrons = np.zeros(runs)
        self.passes = 0


def _unrotated(n: int) -> Placement:
    return Placement(np.zeros(n), np.zeros((n, 3)))


@dataclass
class _Sink:
    """Where the electrons of a batch leave energy and ions, and when."""

    grid: _Grid
    tallies: _Tallies
    run: np.ndarray  # by owner
    beam: BeamShape
    pulse: _Pulse

    def _voxels(
        self, points: np.ndarray, owner: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each point's run-by-voxel index, and its voxel's layer in z."""
        ijk = voxel_indices(
            points,
            _unrotated(len(points)),
            self.grid.minimum,
            self.grid.pixels_per_micron,
        )
        ijk = np.clip(ijk, 0, np.array(self.grid.size) - 1)
        nx, ny, nz = self.grid.size
        flat = ((self.run[owner] * nx + ijk[:, 0]) * ny + ijk[:, 1]) * nz + ijk[:, 2]
        return np.asarray(flat), ijk[:, 2]

    def _layer(self, z_nm: np.ndarray) -> np.ndarray:
        """Each point's voxel layer in z, from z alone.

        An ionisation by an electron that has stopped is judged exposed or
        not where its next step would have ended, which can be undefined,
        but its time against its own voxel's.
        """
        index = np.floor(
            (z_nm / 1000 - self.grid.minimum[2]) * self.grid.pixels_per_micron + 0.5
        )
        return np.clip(index, 0, self.grid.size[2] - 1).astype(np.int64)

    def dose(self, points: np.ndarray, owner: np.ndarray, energy: np.ndarray) -> None:
        """``doseSimple``, which the XFEL subprogram does not report."""

    def resolved(
        self,
        points: np.ndarray,
        owner: np.ndarray,
        energy: np.ndarray,
        time: np.ndarray,
    ) -> None:
        """``voxelEnergyvResolved``, in its time bin.

        Kept if the bin starts before the voxel's cutoff.
        """
        if not len(owner):
            return
        flat, layer = self._voxels(points, owner)
        np.add.at(self.tallies.resolved, flat, energy)
        bins = np.maximum(np.trunc(time / self.pulse.bin), 0)
        kept = bins * self.pulse.bin < self.pulse.cutoff[layer]
        np.add.at(self.tallies.resolved_cut, flat[kept], energy[kept])

    def ionise(
        self,
        points: np.ndarray,
        owner: np.ndarray,
        element: np.ndarray,
        count: np.ndarray,
        time: np.ndarray,
    ) -> None:
        """``addIonisation``: counted only before the voxel's cutoff."""
        if not len(owner):
            return
        kept = time < self.pulse.cutoff[self._layer(points[:, 2])]
        run = self.run[owner]
        np.add.at(self.tallies.ionisations, run[kept], count[kept])
        known = kept & (element >= 0)
        np.add.at(self.tallies.element_ions, (run[known], element[known]), count[known])
        exposed = known & self.beam.exposes(points[:, 0], points[:, 1])
        np.add.at(
            self.tallies.element_ions_exposed,
            (run[exposed], element[exposed]),
            count[exposed],
        )


@dataclass
class _Started:
    """Each photon that freed an electron, batch by batch.

    Its run and time, its electron, and its vacancy if it left one.
    """

    run: list[np.ndarray] = field(default_factory=list)
    time: list[np.ndarray] = field(default_factory=list)
    points: list[np.ndarray] = field(default_factory=list)
    directions: list[np.ndarray] = field(default_factory=list)
    energies: list[np.ndarray] = field(default_factory=list)
    vacancy_owner: list[np.ndarray] = field(default_factory=list)
    vacancy_element: list[np.ndarray] = field(default_factory=list)
    vacancy_shell: list[np.ndarray] = field(default_factory=list)


def _track(
    rng: np.random.Generator,
    setup: SimpleSetup,
    crystal: GOSMedium,
    clock: Clock,
    grid: _Grid,
    tallies: _Tallies,
    pulse: _Pulse,
    started: _Started,
) -> None:
    """Follow every electron, in groups of photons.

    A pass costs about as much for a few electrons as for many, and an XFEL
    run frees few, so they are followed together rather than batch by batch.
    """
    if not started.run:
        return
    run = np.concatenate(started.run)
    points = np.concatenate(started.points)
    directions = np.concatenate(started.directions)
    energies = np.concatenate(started.energies)
    vacancy_owner = np.concatenate(started.vacancy_owner)
    vacancy_element = np.concatenate(started.vacancy_element)
    vacancy_shell = np.concatenate(started.vacancy_shell)
    start_time = np.concatenate(started.time)
    np.add.at(tallies.electrons, run, 1)
    sink = _Sink(grid, tallies, run, setup.beam, pulse)
    placement = _unrotated(len(run))
    for first in range(0, len(run), _TRACKED):
        owner = np.arange(first, min(first + _TRACKED, len(run)))
        held = np.flatnonzero(
            (vacancy_owner >= owner[0]) & (vacancy_owner <= owner[-1])
        )
        passes = track_gos(
            rng,
            crystal,
            None,
            setup.mesh,
            [],
            placement,
            sink,
            points[owner],
            directions[owner],
            energies[owner],
            owner,
            np.ones(len(owner), dtype=bool),
            np.zeros(len(owner), dtype=bool),
            np.zeros(len(owner), dtype=np.int64),
            (
                points[vacancy_owner[held]],
                vacancy_owner[held],
                vacancy_element[held],
                vacancy_shell[held],
            ),
            start_time,
            clock,
        )
        tallies.passes = max(tallies.passes, passes)


def simulate_xfel(setup: SimpleSetup, rng: np.random.Generator) -> list[SimpleRun]:
    """Run ``Runs`` XFEL pulses.

    Parameters
    ----------
    setup : SimpleSetup
        The experiment, with ``xfel`` set.
    rng : numpy.random.Generator
        The random numbers.

    Returns
    -------
    list[SimpleRun]
        One per run.
    """
    runs, photons = setup.runs, setup.photons
    grid = _grid(setup)
    vertices = np.asarray(setup.mesh.vertices)
    half = 1000 * (vertices.max(axis=0) - vertices.min(axis=0)) / 2  # nm
    pulse = _timing(setup, grid.size[2], 2 * float(half[2]))
    with np.errstate(divide="ignore"):
        # Photons per time bin; infinite for a pulse that rounds to 0 fs.
        divisions = photons / np.float64(pulse.length / pulse.bin)
    elements = setup.composition.elements
    tallies = _Tallies(runs, grid.voxels, len(elements))
    highest = (
        setup.photon_energy if setup.energies is None else float(np.max(setup.energies))
    )
    crystal = gos_medium(setup.material, highest)
    started = _Started()

    total = runs * photons
    for first in range(0, total, _BATCH):
        index = np.arange(first, min(first + _BATCH, total))
        run = index // photons
        in_run = index % photons
        n = len(index)
        energy = (
            np.full(n, setup.photon_energy)
            if setup.energies is None
            else np.asarray(setup.energies)[in_run]
        )
        coefficients = photon_coefficients(setup.composition, energy)
        x, y = beam_positions(rng, setup.beam, n)

        # From the bounding box's front face, along +z, at the moment its
        # place in the pulse gives it.
        travel = -coefficients.total_mean_free_path_nm * np.log(rng.random(n))
        z = -half[2] + travel
        start_time = np.trunc(in_run / divisions) * pulse.bin + _light(travel)
        boxed = np.flatnonzero(
            (np.abs(x) <= half[0]) & (np.abs(y) <= half[1]) & (z <= half[2])
        )
        points = np.stack([x[boxed], y[boxed], z[boxed]], axis=1)
        hit = inside(setup.mesh, points, _unrotated(len(boxed)))
        owner, points = boxed[hit], points[hit]
        sink = _Sink(grid, tallies, run, setup.beam, pulse)

        here = coefficients.take(owner)
        elastic = rng.random(len(owner)) < here.elastic_chance
        hit_owner = owner[elastic]
        flat, _ = sink._voxels(points[elastic], hit_owner)
        np.add.at(tallies.voxel_elastic, flat, 1)
        np.add.at(tallies.elastic, run[hit_owner], 1)
        owner, points = owner[~elastic], points[~elastic]
        here = here.take(np.flatnonzero(~elastic))
        compton = rng.random(len(owner)) < here.compton_chance

        pe = np.flatnonzero(~compton)
        absorbed = _photoabsorb(
            rng, setup, setup.composition, here.photoelectric_cdf[pe],
            energy[owner[pe]], owner[pe], points[pe],
        )  # fmt: skip
        np.add.at(
            tallies.raddose, run[absorbed.owner], absorbed.energy + absorbed.binding
        )
        sink.ionise(
            absorbed.points,
            absorbed.owner,
            absorbed.element,
            np.ones(len(pe), dtype=np.int64),
            start_time[absorbed.owner],
        )
        cs_owner, cs_points = owner[compton], points[compton]
        cs_energy, cs_direction = compton_electrons(rng, energy[cs_owner])
        np.add.at(tallies.raddose_compton, run[cs_owner], cs_energy)
        cs_element = choose(
            here.take(np.flatnonzero(compton)).compton_cdf, rng.random(len(cs_owner))
        )
        sink.ionise(
            cs_points,
            cs_owner,
            cs_element,
            np.ones(len(cs_owner), dtype=np.int64),
            start_time[cs_owner],
        )

        # Each photon that freed an electron becomes a record, which owns
        # the electron and the vacancy; all are followed together below.
        base = sum(len(r) for r in started.run)
        freed = np.concatenate([absorbed.owner, cs_owner])
        started.run.append(run[freed])
        started.time.append(start_time[freed])
        started.points.append(np.concatenate([absorbed.points, cs_points]))
        started.directions.append(np.concatenate([absorbed.direction, cs_direction]))
        started.energies.append(np.concatenate([absorbed.energy, cs_energy]))
        # An outer shell's vacancy has nothing to relax.
        held = np.flatnonzero(absorbed.shell != OUTER_SHELL)
        started.vacancy_owner.append(base + held)
        started.vacancy_element.append(absorbed.element[held])
        started.vacancy_shell.append(absorbed.shell[held])

    _track(rng, setup, crystal, Clock(pulse.last), grid, tallies, pulse, started)
    return _process(setup, tallies, grid, half)


def _process(
    setup: SimpleSetup, tallies: _Tallies, grid: _Grid, half: np.ndarray
) -> list[SimpleRun]:
    """``XFEL.processDose``: each run's row of ``outputXFEL.CSV``."""
    nx, ny, nz = grid.size
    real_photons = (setup.pulse_energy / 1e3) / (setup.photon_energy * KEV_TO_JOULES)
    scale = real_photons / setup.photons

    x_dimension, y_dimension, z_dimension = (2 * half).tolist()
    voxel_length = math.pow(setup.pixels_per_micron, -1) * 1000
    length_ratio = z_dimension / voxel_length
    voxel_mass = (setup.density * math.pow(setup.pixels_per_micron, -3) * 1e-12) / 1000
    # The bounding box's.
    volume = x_dimension * y_dimension * z_dimension * 1e-21
    mass = (setup.density * volume) / 1000
    counts = np.asarray(setup.composition.counts, dtype=float)
    cell_volume = float(setup.composition.cell_volume)
    atoms_of = counts * (volume * 1e24) / cell_volume
    total_atoms = float(np.sum(counts)) * (volume * 1e24) / cell_volume
    exposed_volume = min(setup.beam.exposed_area_nm2() * z_dimension * 1e-21, volume)
    exposed_mass = (setup.density * exposed_volume) / 1000
    fraction = exposed_volume / volume
    z_of = [e.atomic_number for e in setup.composition.elements]

    # The coefficients are left at the last photon's energy.
    last_energy = (
        setup.photon_energy if setup.energies is None else float(setup.energies[-1])
    )
    elastic_coefficient = float(
        photon_coefficients(setup.composition, np.array([last_energy])).elastic[0]
    )
    number_elastic = real_photons * (
        1 - math.exp(-elastic_coefficient * (z_dimension / 1000))
    )

    x = ((np.arange(nx) / setup.pixels_per_micron) + grid.minimum[0]) * 1000
    y = ((np.arange(ny) / setup.pixels_per_micron) + grid.minimum[1]) * 1000
    exposed_xy = setup.beam.exposes(np.repeat(x, ny), np.tile(y, nx)).reshape(nx, ny)

    resolved_all = tallies.resolved.reshape(setup.runs, nx, ny, nz)
    cut_all = tallies.resolved_cut.reshape(setup.runs, nx, ny, nz)
    elastic_all = tallies.voxel_elastic.reshape(setup.runs, nx, ny, nz)
    results = []
    for r in range(setup.runs):
        to_joules = scale * KEV_TO_JOULES
        rd3d = ((tallies.raddose[r] * to_joules) / mass) / 1e6 + (
            (tallies.raddose_compton[r] * to_joules) / mass
        ) / 1e6
        resolved = resolved_all[r] * to_joules
        cut = cut_all[r] * to_joules
        voxel_elastic = elastic_all[r] * scale
        total_elastic = float(tallies.elastic[r] * scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = voxel_elastic / total_elastic
            dwd = float(np.sum(np.where(voxel_elastic > 0, cut * share, 0.0)))
            column_energy = cut.sum(axis=2)
            column_elastic = voxel_elastic.sum(axis=2)
            coarse = float(
                np.sum(
                    np.where(
                        (column_elastic > 0) & (column_energy > 0),
                        column_energy * (column_elastic / total_elastic),
                        0.0,
                    )
                )
            )
        xfel_ader = (float(cut[exposed_xy].sum()) / exposed_mass) / 1e6

        ions = math.floor(tallies.ionisations[r] * scale + 0.5)
        per_element: dict[int, float] = {}
        non_h_ions = non_h_atoms = 0.0
        for e, z in enumerate(z_of):
            if tallies.element_ions[r, e] == 0:
                continue
            exposed = math.floor(tallies.element_ions_exposed[r, e] * scale + 0.5)
            per_element[z] = exposed / (atoms_of[e] * fraction)
            if z != 1:
                non_h_ions += exposed
                non_h_atoms += atoms_of[e] * fraction
        with np.errstate(invalid="ignore", divide="ignore"):
            diffraction = np.float64(number_elastic) / np.float64(xfel_ader)
            per_non_h = np.float64(non_h_ions) / np.float64(non_h_atoms)
        results.append(
            SimpleRun(
                rd3d_adwc=rd3d,
                rd3d_ader=(rd3d * mass) / exposed_mass,
                elastic=total_elastic,
                xfel_adwc=(float(cut.sum()) / mass) / 1e6,
                xfel_ader=xfel_ader,
                diffraction_efficiency=float(diffraction),
                ions_per_atom=(ions / total_atoms) / fraction,
                ions_per_non_h_atom=float(per_non_h),
                resolved_no_cut=(float(resolved.sum()) / mass) / 1e6,
                dwd_coarse=(coarse / (voxel_mass * length_ratio)) / 1e6,
                dwd=(dwd / voxel_mass) / 1e6,
                gos=True,
                xfel=True,
                printed_ions_per_atom=(tallies.ionisations[r] * scale)
                / (total_atoms * fraction),
                ions_per_element=per_element,
                diagnostics={
                    "electrons": float(tallies.electrons[r]),
                    "passes": float(tallies.passes),
                },
            )
        )
    return results
