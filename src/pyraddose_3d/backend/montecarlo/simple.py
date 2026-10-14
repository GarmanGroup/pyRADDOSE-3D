"""RADDOSE-3D's simple Monte Carlo, ``Subprogram MONTECARLO``.

Photons cross the crystal, and the surrounding if there is one; those that
interact free a photoelectron or a Compton electron, or scatter elastically
and are counted. Each electron is followed by its elastic scattering and its
stopping power until it stops or leaves (:mod:`.transport`). The atom a
photoelectron leaves behind in the crystal relaxes where it is: its binding
energy is deposited there, less the energy of a fluorescence photon if it
emits one.

Every run is simulated at once, in batches of photons, and tallied by run:
the runs share one generator, whose draws are independent whichever run
they fall to. ``processDose`` then turns each run's tallies into the rows of
``outputMC.CSV`` and ``outputXFEL.CSV``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.geometry import Mesh, triangle_planes
from pyraddose_3d.backend.montecarlo import tables
from pyraddose_3d.backend.montecarlo.geometry import (
    Placement,
    entry_depth,
    inside,
    voxel_indices,
)
from pyraddose_3d.backend.montecarlo.material import stopping_power
from pyraddose_3d.backend.montecarlo.photons import (
    OUTER_SHELL,
    BeamShape,
    PhotonCoefficients,
    beam_positions,
    choose,
    compton_electrons,
    ionised_shells,
    photoelectron_directions,
    photon_coefficients,
    shell_cdf,
    shells,
)
from pyraddose_3d.backend.montecarlo.transport import Media, track

if TYPE_CHECKING:
    from pyraddose_3d.backend.composition import Composition
    from pyraddose_3d.backend.microed import ElectronMaterial

__all__ = ["SimpleRun", "SimpleSetup", "Surrounding", "simulate_simple"]

#: Photons per batch.
_BATCH = 250_000

_SPEED_OF_LIGHT = 299792458.0

#: The elements RADDOSE-3D relaxes: those with relaxation tables. Phosphorus
#: joined them in #48.
_RELAXING = (6, 7, 8, 11, 12, 14, 15, 16, 17, 19, 20, 25, 26, 27, 28, 29, 30, 33, 34)


@dataclass(frozen=True)
class Surrounding:
    """The medium around the crystal: ``CalcSurrounding``.

    Attributes
    ----------
    composition : Composition
        Its contents, in a cell the size of the crystal's unit cell.
    density : float
        g/cm^3.
    material : ElectronMaterial
        The same, for the electron physics.
    thickness_um : tuple[float, float, float]
        ``SurroundingThickness``: how far it extends in x, y and z. Each is
        capped, photon by photon, at the photon's energy over the
        surrounding's stopping power at it; 0 takes that cap.
    """

    composition: Composition
    density: float
    material: ElectronMaterial
    thickness_um: tuple[float, float, float] = (0.0, 0.0, 0.0)


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
        keV: the beam's, or a pink beam's mean.
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
    energies : numpy.ndarray | None
        A pink beam's energy for each photon of a run, keV, in the order the
        photons are simulated; ``None`` for one energy.
    surrounding : Surrounding | None
        The medium around the crystal, if any.
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
    energies: np.ndarray | None = None
    surrounding: Surrounding | None = None


@dataclass
class SimpleRun:
    """One run: a row of ``outputMC.CSV`` and one of ``outputXFEL.CSV``.

    Attributes
    ----------
    rd3d_adwc, rd3d_ader : float
        RADDOSE-3D-style dose -- the energy of every absorbed photon and
        Compton electron, left where it was freed -- over the whole crystal
        and over its exposed region, MGy.
    mc_adwc, mc_ader : float
        The Monte Carlo's dose, the same two ways, MGy.
    elastic : float
        Elastically scattered photons, scaled to the real photon count:
        ``numEl``.
    xfel_adwc, xfel_ader : float
        ``outputXFEL.CSV``'s XFEL doses, which in simple mode hold only the
        binding energies of atoms that do not relax and the energy of
        electrons freed with under 0.05 keV, MGy.
    diffraction_efficiency : float
        ``numEl`` over ``xfel_ader``.
    ions_per_atom, ions_per_non_h_atom : float
        Ionisations per atom, and per atom other than hydrogen, in the
        exposed region.
    resolved_no_cut, dwd_coarse, dwd : float
        ``vResolvedNoCut``, ``DWDcoarse`` and ``DWD``.
    ions_per_element : dict[int, float]
        ``outputIonsPerAtom.CSV``: ionisations per atom in the exposed
        region, for each element ionised, by atomic number.
    diagnostics : dict[str, float]
        Electrons followed and the passes it took.
    """

    rd3d_adwc: float = 0.0
    rd3d_ader: float = 0.0
    mc_adwc: float = 0.0
    mc_ader: float = 0.0
    elastic: float = 0.0
    xfel_adwc: float = 0.0
    xfel_ader: float = 0.0
    diffraction_efficiency: float = 0.0
    ions_per_atom: float = 0.0
    ions_per_non_h_atom: float = 0.0
    resolved_no_cut: float = 0.0
    dwd_coarse: float = 0.0
    dwd: float = 0.0
    ions_per_element: dict[int, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


class _Tallies:
    """What the runs accumulate: keV and counts, by run and voxel."""

    def __init__(self, runs: int, voxels: int, elements: int) -> None:
        self.dose = np.zeros(runs * voxels)  # doseSimple
        self.resolved = np.zeros(runs * voxels)  # voxelEnergyvResolved
        self.voxel_elastic = np.zeros(runs * voxels)  # voxelElastic
        self.raddose = np.zeros(runs)
        self.raddose_compton = np.zeros(runs)
        self.elastic = np.zeros(runs)
        self.ionisations = np.zeros(runs)
        self.element_ions = np.zeros((runs, elements))
        self.element_ions_exposed = np.zeros((runs, elements))
        self.electrons = np.zeros(runs)
        self.passes = 0


@dataclass
class _Grid:
    """The voxel grid, and where points fall in it."""

    minimum: np.ndarray
    size: tuple[int, int, int]
    pixels_per_micron: float

    @property
    def voxels(self) -> int:
        """How many."""
        return self.size[0] * self.size[1] * self.size[2]

    def flat(
        self, points: np.ndarray, placement: Placement, run: np.ndarray
    ) -> np.ndarray:
        """``convertToPixelCoordinates``, as an index into run-by-voxel tallies."""
        ijk = voxel_indices(points, placement, self.minimum, self.pixels_per_micron)
        ijk = np.minimum(ijk, np.array(self.size) - 1)
        nx, ny, nz = self.size
        return np.asarray(((run * nx + ijk[:, 0]) * ny + ijk[:, 1]) * nz + ijk[:, 2])


def _grid(setup: SimpleSetup) -> _Grid:
    """``getMaxPixelCoordinates``."""
    vertices = np.asarray(setup.mesh.vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    nx, ny, nz = (
        math.floor((high[i] - low[i]) * setup.pixels_per_micron + 0.5) + 1
        for i in range(3)
    )
    return _Grid(low, (nx, ny, nz), setup.pixels_per_micron)


@dataclass(frozen=True)
class _Owned:
    """Electron deposits, each electron knowing its photon's placement and run."""

    grid: _Grid
    tallies: _Tallies
    placement: Placement
    run: np.ndarray

    def __call__(
        self, points: np.ndarray, which: np.ndarray, energy: np.ndarray, starting: bool
    ) -> None:
        """Deposit for the electrons ``which``."""
        flat = self.grid.flat(points, self.placement.take(which), self.run[which])
        np.add.at(self.tallies.dose, flat, energy)
        if starting:
            np.add.at(self.tallies.resolved, flat, energy)


def _placements(setup: SimpleSetup, index: np.ndarray) -> Placement:
    """How the crystal stands for each photon: its place in its run.

    ``MC.photonAngle``: a still at its start angle, a rotation from its start
    to its end, as RADDOSE-3D has it since #48.
    """
    start, end = setup.start_angle, setup.end_angle
    if abs(start - end) < setup.angular_resolution:
        angle = np.full(len(index), start)
    else:
        angle = start + ((index / setup.photons) * (end - start))
    # "convert to 0-2pi", truncating towards zero as Java's cast does.
    times_over = np.trunc(angle / (2 * math.pi))
    angle = angle - (times_over * 2 * math.pi)
    rate = np.array(setup.translation_per_radian)
    offset = (
        np.array(setup.start_offset)[None, :] + rate[None, :] * (angle - start)[:, None]
    )
    return Placement(angle, offset)


def _box(mesh: Mesh, thickness_nm: tuple[float, float, float]) -> Mesh:
    """``setupSurroundingVolume``: the crystal's mesh, pushed out.

    Each vertex moves away from the origin along each axis by that axis's
    thickness.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    push = np.array(thickness_nm) / 1000
    return Mesh(
        np.where(vertices < 0, vertices - push, vertices + push), mesh.triangles
    )


def _relaxation(
    z: np.ndarray,
    shell: np.ndarray,
    binding: np.ndarray,
    yields: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``produceAugerElectron`` in simple mode: the energy left at the atom.

    Returns what goes to the Monte Carlo's dose, and what also to the XFEL
    tallies: the binding energy, where the element or shell does not relax.
    Where it does, the binding energy, less a fluorescence photon's energy
    if the draw says it fluoresces, by a transition drawn from the
    fluorescence table, as RADDOSE-3D draws it since #48. An outer shell
    leaves nothing.
    """
    known = shell != OUTER_SHELL
    left = np.where(known, binding, 0.0)
    eligible = (
        np.isin(z, _RELAXING)
        & known
        & (
            (shell == 0)
            | ((shell < 2) & (z == 11))
            | ((shell < 3) & (z == 12))
            | ((shell < 4) & (z <= 20) & (z > 12))
            | ((shell < 7) & (z >= 25) & (z < 31))
            | ((shell < 9) & (z > 32) & (z < 35))
        )
    )
    resolved = np.where(~eligible & known, binding, 0.0)
    chosen = np.flatnonzero(eligible)
    if len(chosen):
        fluoresces = ~(rng.random(len(chosen)) > yields[chosen])
        fluorescence = tables.fluorescence_transitions()
        for i in chosen[fluoresces]:
            key = (int(z[i]), int(shell[i]))
            hits = np.flatnonzero(rng.random() < fluorescence[key].cumulative)
            index = int(hits[0]) if len(hits) else 0
            left[i] = binding[i] - fluorescence[key].energy_kev[index]
    return left, resolved


@dataclass
class _Absorbed:
    """Photoelectric absorptions in one medium."""

    owner: np.ndarray
    points: np.ndarray
    direction: np.ndarray
    energy: np.ndarray
    element: np.ndarray
    shell: np.ndarray
    binding: np.ndarray


def _photoabsorb(
    rng: np.random.Generator,
    setup: SimpleSetup,
    composition: Composition,
    cdf: np.ndarray,
    energy: np.ndarray,
    owner: np.ndarray,
    points: np.ndarray,
) -> _Absorbed:
    """``producePhotoElectron``: the element, the shell, the electron."""
    z_of = np.array([e.atomic_number for e in composition.elements])
    element = choose(cdf, rng.random(len(owner)))
    u = rng.random(len(owner))
    shell = np.empty(len(owner), dtype=np.int64)
    binding = np.zeros(len(owner))
    for e in np.unique(element):
        chosen = np.flatnonzero(element == e)
        z = int(z_of[e])
        shell[chosen] = ionised_shells(shell_cdf(z, energy[chosen]), u[chosen])
        known = chosen[shell[chosen] != OUTER_SHELL]
        binding[known] = shells(z).binding[shell[known]]
    direction = photoelectron_directions(rng, shell == 0, setup.rotated_polarisation)
    return _Absorbed(
        owner, points, direction, energy - binding, element, shell, binding
    )


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
    grid = _grid(setup)
    elements = setup.composition.elements
    z_of = np.array([e.atomic_number for e in elements])
    yield_of = np.stack([shells(int(z)).fluorescence_yield for z in z_of])
    tallies = _Tallies(runs, grid.voxels, len(elements))
    vertices = np.asarray(setup.mesh.vertices)
    reach = float(np.max(np.linalg.norm(vertices[:, None] - vertices[None], axis=-1)))
    surrounding = setup.surrounding
    boxes: dict[tuple[float, float, float], int] = {}
    box_meshes: list[Mesh] = []
    carry = np.full((runs, 3), np.inf)
    if surrounding is not None:
        carry[:] = [t * 1000 if t > 0 else np.inf for t in surrounding.thickness_um]
    media = Media(
        setup.material,
        setup.mesh,
        surrounding.material if surrounding is not None else None,
        box_meshes,
    )

    total = runs * photons
    for first in range(0, total, _BATCH):
        index = np.arange(first, min(first + _BATCH, total))
        run = index // photons
        in_run = index % photons
        placement = _placements(setup, in_run)
        n = len(index)
        energy = (
            np.full(n, setup.photon_energy)
            if setup.energies is None
            else np.asarray(setup.energies)[in_run]
        )
        crystal = photon_coefficients(setup.composition, energy)
        x, y = beam_positions(rng, setup.beam, n)
        front = entry_depth(setup.mesh, x, y, placement)

        # Each interaction: (photons, points, in the surrounding?).
        found: list[tuple[np.ndarray, np.ndarray, bool]] = []
        box = np.zeros(n, dtype=np.int64)
        if surrounding is None:
            travel = -crystal.total_mean_free_path_nm * np.log(rng.random(n))
            candidate = np.flatnonzero(~np.isnan(front) & (travel <= reach * 1000))
            points = np.stack(
                [x[candidate], y[candidate], front[candidate] + travel[candidate]],
                axis=1,
            )
            hit = inside(setup.mesh, points, placement.take(candidate))
            found.append((candidate[hit], points[hit], False))
        else:
            box = _boxes(setup, surrounding, energy, run, carry, boxes, box_meshes)
            paths = _through_surrounding(
                rng, setup, surrounding, crystal, energy, x, y, front, box,
                box_meshes, placement,
            )  # fmt: skip
            found.extend(paths)

        starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
        for owner, points, outside in found:
            if not len(owner):
                continue
            if outside:
                assert surrounding is not None
                composition = surrounding.composition
                medium = photon_coefficients(composition, energy[owner])
                compton = rng.random(len(owner)) < medium.compton_chance
                photo_cdf = medium.photoelectric_cdf
            else:
                composition = setup.composition
                here = crystal.take(owner)
                elastic = rng.random(len(owner)) < here.elastic_chance
                hit = owner[elastic]
                flat = grid.flat(points[elastic], placement.take(hit), run[hit])
                np.add.at(tallies.voxel_elastic, flat, 1)
                np.add.at(tallies.elastic, run[hit], 1)
                owner, points = owner[~elastic], points[~elastic]
                here = here.take(np.flatnonzero(~elastic))
                compton = rng.random(len(owner)) < here.compton_chance
                photo_cdf = here.photoelectric_cdf

            pe = np.flatnonzero(~compton)
            absorbed = _photoabsorb(
                rng, setup, composition, photo_cdf[pe], energy[owner[pe]],
                owner[pe], points[pe],
            )  # fmt: skip
            cs_owner, cs_points = owner[compton], points[compton]
            cs_energy, cs_direction = compton_electrons(rng, energy[cs_owner])
            if not outside:
                _tally_crystal(
                    rng, setup, grid, tallies, z_of, yield_of, placement, run, x, y,
                    absorbed, cs_owner, cs_energy, here.take(np.flatnonzero(compton)),
                )  # fmt: skip
            starts.append(
                (
                    np.concatenate([absorbed.owner, cs_owner]),
                    np.concatenate([absorbed.points, cs_points]),
                    np.concatenate([absorbed.direction, cs_direction]),
                    np.concatenate([absorbed.energy, cs_energy]),
                    outside,
                )
            )

        if not starts:
            continue
        owner = np.concatenate([s[0] for s in starts])
        electron_place = placement.take(owner)
        electron_run = run[owner]
        np.add.at(tallies.electrons, electron_run, 1)
        passes = track(
            rng,
            media,
            np.concatenate([s[1] for s in starts]),
            np.concatenate([s[2] for s in starts]),
            np.concatenate([s[3] for s in starts]),
            np.concatenate([np.full(len(s[0]), s[4]) for s in starts]),
            electron_place,
            box[owner],
            _Owned(grid, tallies, electron_place, electron_run),
        )
        tallies.passes = max(tallies.passes, passes)

    return _process(setup, tallies, grid)


def _tally_crystal(
    rng: np.random.Generator,
    setup: SimpleSetup,
    grid: _Grid,
    tallies: _Tallies,
    z_of: np.ndarray,
    yield_of: np.ndarray,
    placement: Placement,
    run: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    absorbed: _Absorbed,
    cs_owner: np.ndarray,
    cs_energy: np.ndarray,
    cs_coefficients: PhotonCoefficients,
) -> None:
    """What an absorption in the crystal adds besides its electron.

    RADDOSE-3D-style energy, an ionisation of the element (and of the
    exposed region, if the photon was within the aperture), and the
    relaxation of the atom a photoelectron left.
    """
    pe = absorbed.owner
    np.add.at(tallies.raddose, run[pe], absorbed.energy + absorbed.binding)
    _ionise(tallies, run[pe], absorbed.element, setup.beam.exposes(x[pe], y[pe]))
    shell = absorbed.shell
    shell_yield = np.where(
        shell >= 0, yield_of[absorbed.element, np.maximum(shell, 0)], 0.0
    )
    left, resolved = _relaxation(
        z_of[absorbed.element], shell, absorbed.binding, shell_yield, rng
    )
    flat = grid.flat(absorbed.points, placement.take(pe), run[pe])
    np.add.at(tallies.dose, flat, left)
    np.add.at(tallies.resolved, flat, resolved)

    np.add.at(tallies.raddose_compton, run[cs_owner], cs_energy)
    cs_element = choose(cs_coefficients.compton_cdf, rng.random(len(cs_owner)))
    exposed = setup.beam.exposes(x[cs_owner], y[cs_owner])
    _ionise(tallies, run[cs_owner], cs_element, exposed)


def _ionise(
    tallies: _Tallies, run: np.ndarray, element: np.ndarray, exposed: np.ndarray
) -> None:
    """``addIonisation``."""
    np.add.at(tallies.ionisations, run, 1)
    np.add.at(tallies.element_ions, (run, element), 1)
    np.add.at(tallies.element_ions_exposed, (run[exposed], element[exposed]), 1)


def _boxes(
    setup: SimpleSetup,
    surrounding: Surrounding,
    energy: np.ndarray,
    run: np.ndarray,
    carry: np.ndarray,
    boxes: dict[tuple[float, float, float], int],
    meshes: list[Mesh],
) -> np.ndarray:
    """Each photon's surrounding, as an index into ``meshes``.

    Its thickness is capped, in order, by the distance a photoelectron of the
    photon's energy could travel in it.

    ``surroundingThickness`` is a field of the run, so each cap is a running
    minimum over the run's photons so far.
    """
    reach = energy / stopping_power(surrounding.material, energy)
    thickness = np.empty((len(energy), 3))
    for r in np.unique(run):
        chosen = np.flatnonzero(run == r)
        running = np.minimum.accumulate(reach[chosen])
        for axis in range(3):
            thickness[chosen, axis] = np.minimum(carry[r, axis], running)
        carry[r] = thickness[chosen[-1]]
    index = np.empty(len(energy), dtype=np.int64)
    distinct, inverse = np.unique(thickness, axis=0, return_inverse=True)
    for k, row in enumerate(distinct):
        key = (float(row[0]), float(row[1]), float(row[2]))
        if key not in boxes:
            boxes[key] = len(meshes)
            meshes.append(_box(setup.mesh, key))
        index[inverse.reshape(-1) == k] = boxes[key]
    return index


def _in_boxes(
    meshes: list[Mesh],
    box: np.ndarray,
    points: np.ndarray,
    photons: np.ndarray,
    placement: Placement,
) -> np.ndarray:
    """``calculateCrystalOccupancySurrounding`` for each photon's surrounding."""
    result = np.zeros(len(photons), dtype=bool)
    for index in np.unique(box[photons]):
        group = np.flatnonzero(box[photons] == index)
        result[group] = inside(
            meshes[index], points[group], placement.take(photons[group])
        )
    return result


def _through_surrounding(
    rng: np.random.Generator,
    setup: SimpleSetup,
    surrounding: Surrounding,
    crystal: PhotonCoefficients,
    energy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    front: np.ndarray,
    box: np.ndarray,
    meshes: list[Mesh],
    placement: Placement,
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """The photons' paths with a surrounding: where each interacts, if it does.

    From the surrounding's front face, with the surrounding's photoelectric
    and Compton coefficients up to the crystal's front face. Past that, the
    crystal's every coefficient, still measured from the surrounding's face;
    a point that falls in the surrounding is taken to have crossed the
    crystal, and a new distance is drawn from its back face. A photon beside
    the crystal crosses the surrounding alone.
    """
    n = len(energy)
    outer = np.zeros(n)
    for index in np.unique(box):
        chosen = np.flatnonzero(box == index)
        outer[chosen] = entry_depth(
            meshes[index], x[chosen], y[chosen], placement.take(chosen)
        )
    medium = photon_coefficients(surrounding.composition, energy)
    tracked = ~np.isnan(outer)
    through_surrounding = front != outer
    hitting = ~np.isnan(front)
    found: list[tuple[np.ndarray, np.ndarray, bool]] = []

    # In front of the crystal.
    first = -medium.photon_mean_free_path_nm * np.log(rng.random(n))
    first = np.where(tracked & through_surrounding, first, 0.0)
    in_front = tracked & through_surrounding & hitting & (first < np.abs(front - outer))
    owner = np.flatnonzero(in_front)
    found.append(
        (
            owner,
            np.stack([x[owner], y[owner], outer[owner] + first[owner]], axis=1),
            True,
        )
    )

    travel = np.where(
        hitting, -crystal.total_mean_free_path_nm * np.log(rng.random(n)), first
    )
    rest = np.flatnonzero(tracked & ~in_front)
    points = np.stack([x[rest], y[rest], outer[rest] + travel[rest]], axis=1)
    in_crystal = inside(setup.mesh, points, placement.take(rest))
    found.append((rest[in_crystal], points[in_crystal], False))

    rest, points = rest[~in_crystal], points[~in_crystal]
    in_box = _in_boxes(meshes, box, points, rest, placement)
    rest, points = rest[in_box], points[in_box]

    beside = ~hitting[rest]
    found.append((rest[beside], points[beside], True))

    behind = rest[hitting[rest]]
    if len(behind):
        back = entry_depth(
            setup.mesh, x[behind], y[behind], placement.take(behind), crossing=1
        )
        again = -medium.photon_mean_free_path_nm[behind] * np.log(
            rng.random(len(behind))
        )
        points = np.stack([x[behind], y[behind], back + again], axis=1)
        in_box = _in_boxes(meshes, box, points, behind, placement)
        found.append((behind[in_box], points[in_box], True))
    return found


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


def _pulse(exposure: float) -> tuple[float, float]:
    """``PULSE_LENGTH`` (the exposure, rounded) and ``PULSE_BIN_LENGTH``."""
    length = float(math.floor(exposure + 0.5))
    if length <= 1:
        return length, 0.01
    if length <= 50:
        return length, 0.1
    return length, 1.0


def _process(setup: SimpleSetup, tallies: _Tallies, grid: _Grid) -> list[SimpleRun]:
    """``processDose``: each run's rows of ``outputMC.CSV`` and ``outputXFEL.CSV``."""
    nx, ny, nz = grid.size
    pulse_energy = (setup.flux * setup.exposure) * (setup.photon_energy * KEV_TO_JOULES)
    real_photons = pulse_energy / (setup.photon_energy * KEV_TO_JOULES)
    scale = real_photons / setup.photons

    vertices = np.asarray(setup.mesh.vertices)
    z_dimension = 1000 * float(vertices[:, 2].max() - vertices[:, 2].min())
    voxel_length = math.pow(setup.pixels_per_micron, -1) * 1000
    length_ratio = z_dimension / voxel_length
    voxel_mass = (setup.density * math.pow(setup.pixels_per_micron, -3) * 1e-12) / 1000
    volume = _sample_volume(setup.mesh) * 1e-12
    mass = (setup.density * volume) / 1000
    counts = np.asarray(setup.composition.counts, dtype=float)
    cell_volume = float(setup.composition.cell_volume)
    atoms_of = counts * (volume * 1e24) / cell_volume
    total_atoms = float(np.sum(counts)) * (volume * 1e24) / cell_volume
    exposed_volume = min(setup.beam.exposed_area_nm2() * z_dimension * 1e-21, volume)
    exposed_mass = (setup.density * exposed_volume) / 1000
    fraction = exposed_volume / volume
    z_of = [e.atomic_number for e in setup.composition.elements]

    # The voxels the XFEL time cut keeps: all, unless the pulse is very short.
    length, bin_ = _pulse(setup.exposure)
    last = np.array(
        [
            ((1 / _SPEED_OF_LIGHT) * ((z_dimension / 1e9) / nz) * (c + 1) * 1e15)
            + length
            for c in range(nz)
        ]
    )
    kept = (0 * bin_) < last - (1 * bin_)
    x = ((np.arange(nx) / setup.pixels_per_micron) + grid.minimum[0]) * 1000
    y = ((np.arange(ny) / setup.pixels_per_micron) + grid.minimum[1]) * 1000
    exposed_xy = setup.beam.exposes(np.repeat(x, ny), np.tile(y, nx)).reshape(nx, ny)

    dose = tallies.dose.reshape(setup.runs, nx, ny, nz)
    resolved_all = tallies.resolved.reshape(setup.runs, nx, ny, nz)
    elastic_all = tallies.voxel_elastic.reshape(setup.runs, nx, ny, nz)
    results = []
    for r in range(setup.runs):
        to_joules = scale * KEV_TO_JOULES
        joules = dose[r] * to_joules
        rd3d = ((tallies.raddose[r] * to_joules) / mass) / 1e6 + (
            (tallies.raddose_compton[r] * to_joules) / mass
        ) / 1e6
        resolved = resolved_all[r] * to_joules
        cut = resolved * kept[None, None, :]
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
            diffraction = np.float64(total_elastic) / np.float64(xfel_ader)
            per_non_h = np.float64(non_h_ions) / np.float64(non_h_atoms)
        results.append(
            SimpleRun(
                rd3d_adwc=rd3d,
                rd3d_ader=(rd3d * mass) / exposed_mass,
                mc_adwc=(float(joules.sum()) / mass) / 1e6,
                mc_ader=(float(joules[exposed_xy].sum()) / exposed_mass) / 1e6,
                elastic=total_elastic,
                xfel_adwc=(float(cut.sum()) / mass) / 1e6,
                xfel_ader=xfel_ader,
                diffraction_efficiency=float(diffraction),
                ions_per_atom=(ions / total_atoms) / fraction,
                ions_per_non_h_atom=float(per_non_h),
                resolved_no_cut=(float(resolved.sum()) / mass) / 1e6,
                dwd_coarse=(coarse / (voxel_mass * length_ratio)) / 1e6,
                dwd=(dwd / voxel_mass) / 1e6,
                ions_per_element=per_element,
                diagnostics={
                    "electrons": float(tallies.electrons[r]),
                    "passes": float(tallies.passes),
                },
            )
        )
    return results
