"""RADDOSE-3D's XFEL subprogram, ``Subprogram XFEL``: the Monte Carlo's time mode.

The Monte Carlo's GOS model (:mod:`.simple`, :mod:`.gos_transport`), followed
in time through one pulse, as RADDOSE-3D has it since #56. The pulse's
photons arrive evenly over its length, in the order they are simulated; each
reaches its interaction when the pulse's front, a plane that meets the
crystal's front at the photon's place in the pulse, does. Each electron's
steps take their flight time, and each relaxation its lifetime. Nothing is
followed after ``lastTime``, when the last photon has left the crystal, and
the dose and ionisations the run reports are those each voxel holds a time
bin before the last photon has crossed it. The pulse lasts femtoseconds, so
the crystal stands still through it, at the wedge's start.

This module holds what the time mode adds: the pulse's timing, a sink that
sorts each deposit and ionisation by its time as it is made, and
``processDose``'s XFEL output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.montecarlo.geometry import Placement, voxel_indices
from pyraddose_3d.backend.montecarlo.photons import photon_coefficients

if TYPE_CHECKING:
    from pyraddose_3d.backend.montecarlo.photons import BeamShape
    from pyraddose_3d.backend.montecarlo.simple import (
        SimpleRun,
        SimpleSetup,
        _Grid,
        _Tallies,
    )

__all__ = ["Pulse", "TimedSink", "light", "pulse_timing", "xfel_runs"]

_SPEED_OF_LIGHT = 299792458.0


def light(distance_nm: np.ndarray | float) -> np.ndarray:
    """How long light takes to travel ``distance_nm``, fs."""
    return np.asarray((1 / _SPEED_OF_LIGHT) * (np.asarray(distance_nm) / 1e9) * 1e15)


@dataclass(frozen=True)
class Pulse:
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
    depth : float
        The crystal's depth, nm: its bounding box's along z.
    """

    length: float
    bin: float
    last: float
    cutoff: np.ndarray
    depth: float

    def slots(self, in_run: np.ndarray, photons: int) -> np.ndarray:
        """Each photon's place in the pulse, fs: evenly, in the order simulated."""
        with np.errstate(divide="ignore"):
            # Photons per time bin; infinite for a pulse of no length.
            divisions = photons / np.float64(self.length / self.bin)
        return np.asarray(np.trunc(in_run / divisions) * self.bin)

    def arrival(self, slot: np.ndarray, z_nm: np.ndarray) -> np.ndarray:
        """When the pulse's front, at the crystal's front at ``slot``, reaches z."""
        return np.asarray(slot + light(z_nm + self.depth / 2))


def pulse_timing(setup: SimpleSetup, layers: int) -> Pulse:
    """``PULSE_LENGTH``, its bins, ``lastTime`` and ``lastTimeVox``."""
    from pyraddose_3d.backend.montecarlo.simple import _pulse

    vertices = np.asarray(setup.mesh.vertices)
    depth = 1000 * float(vertices[:, 2].max() - vertices[:, 2].min())
    length, bin_ = _pulse(setup.exposure)
    last = float(light(depth)) + length
    layer_time = np.array(
        [
            ((1 / _SPEED_OF_LIGHT) * ((depth / 1e9) / layers) * (k + 1) * 1e15) + length
            for k in range(layers)
        ]
    )
    return Pulse(length, bin_, last, layer_time - (1 * bin_), depth)


@dataclass
class TimedSink:
    """Where a pulse's electrons leave energy and ions, and when.

    Each owner is a photon's record, with its run and how the crystal stood.
    """

    grid: _Grid
    tallies: _Tallies
    placement: Placement
    run: np.ndarray
    beam: BeamShape
    pulse: Pulse

    def _voxels(
        self, points: np.ndarray, owner: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each point's run-by-voxel index, and its voxel's layer in z."""
        ijk = voxel_indices(
            points,
            self.placement.take(owner),
            self.grid.minimum,
            self.grid.pixels_per_micron,
        )
        ijk = np.clip(ijk, 0, np.array(self.grid.size) - 1)
        nx, ny, nz = self.grid.size
        flat = ((self.run[owner] * nx + ijk[:, 0]) * ny + ijk[:, 1]) * nz + ijk[:, 2]
        return np.asarray(flat), ijk[:, 2]

    def _layer(self, points: np.ndarray, owner: np.ndarray) -> np.ndarray:
        """Each point's voxel layer in z.

        An ionisation by an electron that has stopped is judged exposed or
        not where its next step would have ended, which can be undefined; its
        voxel is where it stopped, whose z alone the layer then takes.
        """
        defined = np.isfinite(points).all(axis=1)
        layer = np.empty(len(points), dtype=np.int64)
        if defined.any():
            _, layer[defined] = self._voxels(points[defined], owner[defined])
        if (~defined).any():
            z = points[~defined, 2]
            index = np.floor(
                (z / 1000 - self.grid.minimum[2]) * self.grid.pixels_per_micron + 0.5
            )
            layer[~defined] = np.clip(index, 0, self.grid.size[2] - 1)
        return layer

    def dose(self, points: np.ndarray, owner: np.ndarray, energy: np.ndarray) -> None:
        """``doseSimple``, which the XFEL subprogram does not report."""

    def resolved(
        self,
        points: np.ndarray,
        owner: np.ndarray,
        energy: np.ndarray,
        time: np.ndarray,
    ) -> None:
        """``addResolved``: kept if its time bin starts before the voxel's cutoff."""
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
        kept = time < self.pulse.cutoff[self._layer(points, owner)]
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


def xfel_runs(
    setup: SimpleSetup,
    tallies: _Tallies,
    grid: _Grid,
    volume_cm3: float,
    depth_nm: float,
) -> list[SimpleRun]:
    """``processDose`` for the XFEL subprogram: each run's ``outputXFEL.CSV`` row.

    As the Monte Carlo's, with the doses, diffraction-weighted doses and
    ionisations those made before each voxel's cutoff, and XFEL's printout.
    """
    from pyraddose_3d.backend.montecarlo.simple import SimpleRun

    nx, ny, nz = grid.size
    real_photons = (setup.pulse_energy / 1e3) / (setup.photon_energy * KEV_TO_JOULES)
    scale = real_photons / setup.photons

    voxel_length = math.pow(setup.pixels_per_micron, -1) * 1000
    length_ratio = depth_nm / voxel_length
    voxel_mass = (setup.density * math.pow(setup.pixels_per_micron, -3) * 1e-12) / 1000
    mass = (setup.density * volume_cm3) / 1000
    counts = np.asarray(setup.composition.counts, dtype=float)
    cell_volume = float(setup.composition.cell_volume)
    atoms_of = counts * (volume_cm3 * 1e24) / cell_volume
    total_atoms = float(np.sum(counts)) * (volume_cm3 * 1e24) / cell_volume
    exposed_volume = min(setup.beam.exposed_area_nm2() * depth_nm * 1e-21, volume_cm3)
    exposed_mass = (setup.density * exposed_volume) / 1000
    fraction = exposed_volume / volume_cm3
    z_of = [e.atomic_number for e in setup.composition.elements]

    # The elastic photons: the share of the pulse's that the elastic
    # coefficient, left at the last photon's energy, scatters in the crystal's
    # depth.
    last_energy = (
        setup.photon_energy if setup.energies is None else float(setup.energies[-1])
    )
    elastic_coefficient = float(
        photon_coefficients(setup.composition, np.array([last_energy])).elastic[0]
    )
    number_elastic = real_photons * (
        1 - math.exp(-elastic_coefficient * (depth_nm / 1000))
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
