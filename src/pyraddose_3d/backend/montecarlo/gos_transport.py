"""Electrons in the GOS model: ``trackPhotoelectron`` and ``produceAugerElectron``.

``Subprogram GOS`` follows each electron through elastic and inelastic
collisions. Each step, in the crystal:

- the electron moves ``s``, drawn from its total mean free path: elastic,
  and inelastic -- the GOS model's outer subshells and Bethe's inner shells;
- if it is a primary, ``s`` times the stopping power goes to the Monte
  Carlo's own dose, which GOS mode does not report;
- an inelastic collision ionises an inner shell or an outer one, distantly
  (longitudinally or transversely) or closely, losing W. The freed electron,
  W less the binding energy, is followed in turn if it has over 0.05 keV;
  otherwise its energy is left there, with the ionisations it would make.
  The vacancy relaxes (:class:`Cascade`);
- the electron's energy falls only by these W: unlike the simple model, the
  stopping power does not slow it.

GOS mode's dose -- ``vResolved`` -- is what the secondaries too slow to
follow, the vacancies' relaxation and the electrons' last 0.05 keV leave.
Out of the crystal, into a surrounding, an electron scatters elastically and
slows by its stopping power, depositing nothing, as in the simple model.

Here every electron takes one step per pass, and the secondaries and Auger
electrons each step frees join the next pass. Each carries the Java's
per-electron state, its quirks included (``UPSTREAM_FIXES.md``):

- the inner shells' mean free path in its total is the one at the energy the
  electron started with, ``startingInnerShellLambda``, never updated;
- with none at the start, the total took ``1 / gosOuterLambda``, an inverse
  length, for a length -- for the first step, and after leaving or coming
  back into the crystal; later updates below 0.5 keV take the length;
- the chance an inelastic collision is with an inner shell is not updated
  while no inner shell is open, and comes back into the crystal computed at
  the starting energy, as do the inner shells' relative chances.

The XFEL subprogram follows the same electrons with a :class:`Clock`: each
carries the time since the pulse began, which its steps advance by their
flight time and a relaxation by the transition's lifetime, and nothing is
followed past ``lastTime``, when the pulse's last photon has left the
crystal. Every deposit and ionisation is passed on with its time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.microed import _wk
from pyraddose_3d.backend.montecarlo import tables
from pyraddose_3d.backend.montecarlo.angles import deflection_angles
from pyraddose_3d.backend.montecarlo.geometry import exit_distance, inside
from pyraddose_3d.backend.montecarlo.gos import (
    ShellTable,
    bethe_shells,
    energy_loss_distant,
    primary_angle_close,
    primary_angle_distant,
    qak,
    recoil_distant,
    sample_k,
    secondary_angle_close,
    secondary_angle_distant,
    shell_table,
    wak,
)
from pyraddose_3d.backend.montecarlo.material import (
    elastic_cross_sections,
    elastic_element_cdf,
    elastic_mean_free_path,
    stopping_power,
)
from pyraddose_3d.backend.montecarlo.photons import isotropic_directions, shells
from pyraddose_3d.backend.montecarlo.transport import STOPPED_KEV, new_directions
from pyraddose_3d.backend.montecarlo.transport import _first_above as _elastic_element

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyraddose_3d.backend.geometry import Mesh
    from pyraddose_3d.backend.microed import ElectronMaterial
    from pyraddose_3d.backend.montecarlo.geometry import Placement

__all__ = ["Cascade", "Clock", "GOSMedium", "Sink", "gos_medium", "track_gos"]

#: Above this energy, keV, an electron's cross-sections are brought up to date
#: only after it loses energy.
_UPDATE_ABOVE = 0.5
#: Elements whose vacancies relax by the tables: ``augerElements``.
_RELAXING = frozenset(
    (6, 7, 8, 11, 12, 14, 15, 16, 17, 19, 20, 25, 26, 27, 28, 29, 30, 33, 34)
)
#: Columns the binding-energy table is padded to: vacancies past M5 bind
#: nothing, as ``getShellBindingEnergy``'s switch gives.
_SHELL_COLUMNS = 20
#: Generations of vacancies after which a cascade is cut short; real ones end
#: within a few.
_MAX_GENERATIONS = 64

#: ``XFEL.c``, ``m`` and ``h``: m/s, kg and J s.
_SPEED_OF_LIGHT = 299792458.0
_ELECTRON_MASS = 9.10938356e-31
_PLANCK = 6.62607004e-34


@dataclass(frozen=True)
class Clock:
    """The XFEL subprogram's time, in fs.

    Attributes
    ----------
    last : float
        ``lastTime``: the pulse's length, rounded to whole fs, and the time
        light takes to cross the crystal. An electron whose step ends later
        is dropped, with the energy it had; a vacancy that would relax later
        is left as it is.
    """

    last: float

    @staticmethod
    def flight(energy: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """``getTimeToDistance``: fs for ``distance`` nm at ``energy`` keV."""
        rest = _ELECTRON_MASS * _SPEED_OF_LIGHT**2
        beta_squared = 1 - ((rest / (energy * KEV_TO_JOULES + rest)) ** 2)
        speed = np.sqrt(beta_squared * _SPEED_OF_LIGHT**2) * 1e9 / 1e15  # nm/fs
        return np.asarray((1 / speed) * distance)

    @staticmethod
    def lifetime(linewidth_ev: np.ndarray) -> np.ndarray:
        """A transition's lifetime from its natural linewidth: hbar over it, fs."""
        return np.asarray(
            1e15 * ((_PLANCK / (2 * math.pi)) / ((linewidth_ev / 1000) * KEV_TO_JOULES))
        )


class Sink(Protocol):
    """Where electrons leave their energy and ionisations.

    Each deposit and ionisation comes with its time, fs: 0 but in the XFEL
    subprogram.
    """

    def dose(self, points: np.ndarray, owner: np.ndarray, energy: np.ndarray) -> None:
        """``doseSimple``: the Monte Carlo's own dose, keV."""

    def resolved(
        self,
        points: np.ndarray,
        owner: np.ndarray,
        energy: np.ndarray,
        time: np.ndarray,
    ) -> None:
        """``voxelEnergyvResolved``: GOS mode's dose, keV."""

    def ionise(
        self,
        points: np.ndarray,
        owner: np.ndarray,
        element: np.ndarray,
        count: np.ndarray,
        time: np.ndarray,
    ) -> None:
        """``addIonisation``: ``count`` ions of ``element`` at each point.

        ``element`` is -1 for none; an ion is exposed if the point's x and y
        are in the beam.
        """


@dataclass(frozen=True)
class GOSMedium:
    """One medium as the GOS transport reads it.

    Attributes
    ----------
    material : ElectronMaterial
        The medium.
    table : ShellTable
        Its subshells' inverse mean free paths over energy.
    inner : numpy.ndarray
        Each element's inner subshells, which is also its valence subshell's
        row: ``getNumValenceElectronsSubshells``.
    wk : numpy.ndarray
        ``(elements, 10)``: ``getWkMolecule`` for each subshell row, eV.
    binding : numpy.ndarray
        ``(elements, _SHELL_COLUMNS)``: ``getShellBindingEnergy``, keV: K, L1
        to L3, M1 to M5, then zeros. An inner-shell collision's binding
        energy, and a photoelectric vacancy's.
    valence_binding : numpy.ndarray
        Each element's valence subshell's ``getShellBindingSubshell``, keV: a
        valence collision's binding energy, from the table the model's
        cross-sections are built with (``MC.collisionBindingEnergy``).
    plasma : float
        ``getPlasmaEnergy``, keV.
    z : numpy.ndarray
        Each element's atomic number.
    """

    material: ElectronMaterial
    table: ShellTable
    inner: np.ndarray
    wk: np.ndarray
    binding: np.ndarray
    valence_binding: np.ndarray
    plasma: float
    z: np.ndarray

    def outer(self, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``getGOSOuterLambda`` and ``getGOSOuterShellProbs`` at each energy.

        Returns the mean free path, nm, and the running chances over the
        elements, ``(len(energy), elements)``.
        """
        rows = self.table.at(energy)
        values = rows[:, np.arange(len(self.inner)), self.inner, 3]
        total = values.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1 / total, np.cumsum(values, axis=1) / total[:, None]

    def inner_shells(self, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``betheIonisationxSection`` and ``getAllShellProbs`` at each energy.

        Returns the mean free path, nm (0 where no shell is open), and the
        running chances over every element's nine shells in turn,
        ``(len(energy), elements * 9)``: NaN where no shell is open.
        """
        sigma = bethe_shells(self.material, energy).reshape(len(energy), -1)
        total = sigma.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            free_path = np.where(total > 0, 1 / total, 0.0)
            return free_path, np.cumsum(sigma, axis=1) / total[:, None]

    def elastic_free_path(self, energy: np.ndarray) -> np.ndarray:
        """``getElectronElasticMFPL``, nm."""
        sigma = elastic_cross_sections(self.material, energy)
        return np.asarray(elastic_mean_free_path(self.material, sigma))

    def stopping_power(self, energy: np.ndarray) -> np.ndarray:
        """``getStoppingPower``, keV/nm."""
        return np.asarray(stopping_power(self.material, energy))


def _resonance(material: ElectronMaterial, atom: object, index: int) -> float:
    """``getWkMolecule``, or NaN where the tables give no binding energy.

    No collision reaches such a subshell.
    """
    try:
        return _wk(material, material.sternheimer_a, atom, index)  # type: ignore[arg-type]
    except ValueError:
        return math.nan


def gos_medium(material: ElectronMaterial, highest: float) -> GOSMedium:
    """Prepare a medium for electrons of up to ``highest`` keV."""
    atoms = material.atoms
    inner = np.array([atom.valence[1] for atom in atoms], dtype=np.int64)
    wk = np.array(
        [[_resonance(material, atom, i) for i in range(10)] for atom in atoms]
    )
    binding = np.zeros((len(atoms), _SHELL_COLUMNS))
    for row, atom in enumerate(atoms):
        binding[row, :9] = np.nan_to_num(shells(atom.z).binding)
    return GOSMedium(
        material=material,
        table=shell_table(material, STOPPED_KEV, highest * 1.001),
        inner=inner,
        wk=wk,
        binding=binding,
        valence_binding=np.array(
            [
                atom.subshell_binding(int(v))
                for atom, v in zip(atoms, inner, strict=True)
            ]
        ),
        plasma=material.plasma_energy / 1000,
        z=np.array([atom.z for atom in atoms]),
    )


def _first_above(cumulative: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """The first column whose running chance exceeds the draw, -1 if none."""
    above = cumulative > uniform[:, None]
    return np.where(above.any(axis=1), above.argmax(axis=1), -1)


def _relaxes(z: np.ndarray, shell: np.ndarray) -> np.ndarray:
    """``produceAugerElectron``'s test: which vacancies the tables relax."""
    return np.isin(z, list(_RELAXING)) & (
        (shell == 0)
        | ((shell < 2) & (z == 11))
        | ((shell < 3) & (z == 12))
        | ((shell < 4) & (z <= 20) & (z > 12))
        | ((shell < 7) & (z >= 25) & (z < 31))
        | ((shell < 9) & (z > 32) & (z < 35))
    )


@dataclass
class _Born:
    """Electrons freed during a pass, to join the next."""

    points: list[np.ndarray] = field(default_factory=list)
    directions: list[np.ndarray] = field(default_factory=list)
    energies: list[np.ndarray] = field(default_factory=list)
    owners: list[np.ndarray] = field(default_factory=list)
    times: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        energies: np.ndarray,
        owners: np.ndarray,
        times: np.ndarray,
    ) -> None:
        if len(energies):
            self.points.append(points)
            self.directions.append(directions)
            self.energies.append(energies)
            self.owners.append(owners)
            self.times.append(times)


class Cascade:
    """``produceAugerElectron`` in GOS mode: a vacancy's relaxation, in full.

    A vacancy the tables cover fluoresces or emits an Auger electron, drawn
    by the shell's fluorescence yield. An Auger transition frees the
    electron, to be followed, ionises the atom again, and leaves two
    vacancies, which relax in turn; a fluorescence transition emits a
    photon, which escapes, and leaves one vacancy. Each leaves at the atom
    only what the electron or photon and the vacancies it leaves do not
    carry: a few eV where the tables disagree. Any other vacancy leaves its
    binding energy, to GOS mode's dose and to the Monte Carlo's. As
    RADDOSE-3D has it since #53: the vacancies' energy was counted twice,
    and in GOS mode a fluorescence photon's left at the atom.

    With a :class:`Clock`, as the XFEL subprogram relaxes a vacancy
    (``XFEL.produceAugerElectron``, ``sendOutAuger`` and ``sendOutFl``): only
    before ``lastTime``; and the electron, the ionisation and the vacancies a
    transition leaves come the transition's lifetime later, and only if that
    is still before ``lastTime``.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        crystal: GOSMedium,
        sink: Sink,
        clock: Clock | None = None,
    ) -> None:
        self.rng = rng
        self.crystal = crystal
        self.sink = sink
        self.clock = clock
        self.yields = np.zeros((len(crystal.z), _SHELL_COLUMNS))
        for row, z in enumerate(crystal.z):
            self.yields[row, :9] = shells(int(z)).fluorescence_yield
        self.auger = tables.auger_transitions()
        self.fluorescence = tables.fluorescence_transitions()

    def _transition(self, table: dict, z: np.ndarray, shell: np.ndarray) -> np.ndarray:
        """``getTransitionIndex`` for each vacancy: slot 0 if no draw falls."""
        index = np.zeros(len(z), dtype=np.int64)
        for key in set(zip(z.tolist(), shell.tolist(), strict=True)):
            chosen = np.flatnonzero((z == key[0]) & (shell == key[1]))
            hit = _first_above(
                np.broadcast_to(table[key].cumulative, (len(chosen), 65)),
                self.rng.random(len(chosen)),
            )
            index[chosen] = np.maximum(hit, 0)
        return index

    def _edge(self, element: np.ndarray, shell: np.ndarray) -> np.ndarray:
        """``getShellBindingEnergy``: a vacancy's binding energy, keV; 0 past M5."""
        return self.crystal.binding[element, np.minimum(shell, _SHELL_COLUMNS - 1)]

    def _lifetimes(
        self, table: dict, z: np.ndarray, shell: np.ndarray, index: np.ndarray
    ) -> np.ndarray:
        """Each transition's lifetime, fs; 0 without a clock."""
        if self.clock is None:
            return np.zeros(len(z))
        linewidth = np.array(
            [
                table[(int(zz), int(ss))].linewidth_ev[i]
                for zz, ss, i in zip(z, shell, index, strict=True)
            ]
        ).reshape(-1)
        return self.clock.lifetime(linewidth)

    def relax(
        self,
        points: np.ndarray,
        owner: np.ndarray,
        element: np.ndarray,
        shell: np.ndarray,
        born: _Born,
        time: np.ndarray | None = None,
        held: np.ndarray | None = None,
    ) -> None:
        """Relax each vacancy, and every one it leaves, to the end.

        ``time`` is when each vacancy was made, fs. ``held`` is the binding
        energy each holds, keV, where a collision gave it one other than its
        shell's edge; the vacancies it leaves hold their edges.
        """
        crystal = self.crystal
        if time is None:
            time = np.zeros(len(element))
        if held is None:
            held = crystal.binding[element, np.minimum(shell, _SHELL_COLUMNS - 1)]
        for _ in range(_MAX_GENERATIONS):
            if self.clock is not None:
                on_time = np.flatnonzero(time < self.clock.last)
                points, owner, element, shell, time, held = (
                    points[on_time],
                    owner[on_time],
                    element[on_time],
                    shell[on_time],
                    time[on_time],
                    held[on_time],
                )
            if not len(element):
                return
            shell = np.minimum(shell, _SHELL_COLUMNS - 1)
            z = crystal.z[element]
            binding = held
            relaxing = _relaxes(z, shell)
            if (~relaxing).any():
                rest = np.flatnonzero(~relaxing)
                self.sink.resolved(points[rest], owner[rest], binding[rest], time[rest])
                self.sink.dose(points[rest], owner[rest], binding[rest])
            chosen = np.flatnonzero(relaxing)
            fluoresces = ~(
                self.rng.random(len(chosen))
                > self.yields[element[chosen], shell[chosen]]
            )

            auger = chosen[~fluoresces]
            index = self._transition(self.auger, z[auger], shell[auger])
            energy = np.array(
                [
                    self.auger[(int(zz), int(ss))].energy_kev[i]
                    for zz, ss, i in zip(z[auger], shell[auger], index, strict=True)
                ]
            ).reshape(-1)
            exits = np.array(
                [
                    self.auger[(int(zz), int(ss))].exit_shell[i]  # type: ignore[index]
                    for zz, ss, i in zip(z[auger], shell[auger], index, strict=True)
                ],
                dtype=np.int64,
            ).reshape(-1)
            drops_a = np.array(
                [
                    self.auger[(int(zz), int(ss))].drop_shell[i]
                    for zz, ss, i in zip(z[auger], shell[auger], index, strict=True)
                ],
                dtype=np.int64,
            ).reshape(-1)
            # What the Auger electron and the two vacancies left do not carry.
            left = (
                binding[auger]
                - energy
                - self._edge(element[auger], exits)
                - self._edge(element[auger], drops_a)
            )
            self.sink.resolved(points[auger], owner[auger], left, time[auger])
            later_a = time[auger] + self._lifetimes(
                self.auger, z[auger], shell[auger], index
            )
            directions = isotropic_directions(self.rng, len(auger))
            emitted = (
                np.flatnonzero(later_a < self.clock.last)
                if self.clock is not None
                else np.arange(len(auger))
            )
            self.sink.ionise(
                points[auger[emitted]],
                owner[auger[emitted]],
                element[auger[emitted]],
                np.ones(len(emitted), dtype=np.int64),
                later_a[emitted],
            )
            born.add(
                points[auger[emitted]],
                directions[emitted],
                energy[emitted],
                owner[auger[emitted]],
                later_a[emitted],
            )

            light = chosen[fluoresces]
            index = self._transition(self.fluorescence, z[light], shell[light])
            drops_f = np.array(
                [
                    self.fluorescence[(int(zz), int(ss))].drop_shell[i]
                    for zz, ss, i in zip(z[light], shell[light], index, strict=True)
                ],
                dtype=np.int64,
            ).reshape(-1)
            # What the photon, which escapes, and the vacancy left do not carry.
            photon = np.array(
                [
                    self.fluorescence[(int(zz), int(ss))].energy_kev[i]
                    for zz, ss, i in zip(z[light], shell[light], index, strict=True)
                ]
            ).reshape(-1)
            left = binding[light] - photon - self._edge(element[light], drops_f)
            self.sink.resolved(points[light], owner[light], left, time[light])
            later_f = time[light] + self._lifetimes(
                self.fluorescence, z[light], shell[light], index
            )

            # The vacancies left: the Auger electron's and the one filled
            # from, then the fluorescence's, each relaxed in turn.
            points = np.concatenate([points[auger], points[auger], points[light]])
            owner = np.concatenate([owner[auger], owner[auger], owner[light]])
            element = np.concatenate([element[auger], element[auger], element[light]])
            shell = np.concatenate([exits, drops_a, drops_f])
            time = np.concatenate([later_a, later_a, later_f])
            held = crystal.binding[element, np.minimum(shell, _SHELL_COLUMNS - 1)]


@dataclass
class _Electrons:
    """Every electron being followed, and the state the Java keeps for it."""

    position: np.ndarray  # previousX/Y/Z, nm
    target: np.ndarray  # xn/yn/zn: where the next step ends
    direction: np.ndarray
    energy: np.ndarray
    start: np.ndarray  # startingEnergy
    primary: np.ndarray
    outside: np.ndarray  # surrounding
    owner: np.ndarray
    box: np.ndarray
    fresh: np.ndarray  # not yet set up
    step: np.ndarray  # s
    inelastic_free_path: np.ndarray  # gosInelasticLambda
    inner_start: np.ndarray  # startingInnerShellLambda, crystal
    p_inner: np.ndarray  # Pinner
    p_inelastic: np.ndarray  # Pinel
    inner_energy: np.ndarray  # where ionisationProbs was last computed
    inner_current: np.ndarray  # innerShellLamda: the last inner-shell update
    time: np.ndarray  # timeStamp, fs

    @classmethod
    def blank(cls, n: int) -> _Electrons:
        """``n`` electrons, every field zero, false or fresh."""
        arrays: dict[str, np.ndarray] = {}
        for name in cls.__dataclass_fields__:
            if name in _VECTORS:
                arrays[name] = np.zeros((n, 3))
            elif name in _FLAGS:
                arrays[name] = np.full(n, name == "fresh")
            elif name in _INDICES:
                arrays[name] = np.zeros(n, dtype=np.int64)
            else:
                arrays[name] = np.zeros(n)
        return cls(**arrays)

    def __len__(self) -> int:
        return len(self.energy)

    def take(self, keep: np.ndarray) -> _Electrons:
        return _Electrons(
            **{name: getattr(self, name)[keep] for name in self.__dataclass_fields__}
        )

    def extend(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        energies: np.ndarray,
        owners: np.ndarray,
        primary: bool,
        outside: np.ndarray,
        box: np.ndarray,
        times: np.ndarray,
    ) -> _Electrons:
        """These electrons, and new ones, fresh, to be set up."""
        new = _Electrons.blank(len(energies))
        new.position = np.asarray(points, dtype=float)
        new.target = new.position.copy()
        new.direction = np.asarray(directions, dtype=float)
        new.energy = np.asarray(energies, dtype=float)
        new.start = new.energy.copy()
        new.primary[:] = primary
        new.outside = np.asarray(outside, dtype=bool)
        new.owner = np.asarray(owners, dtype=np.int64)
        new.box = np.asarray(box, dtype=np.int64)
        new.time = np.asarray(times, dtype=float)
        return _Electrons(
            **{
                name: np.concatenate([getattr(self, name), getattr(new, name)])
                for name in self.__dataclass_fields__
            }
        )


#: ``_Electrons``' fields of three components, of flags, and of indices.
_VECTORS = frozenset(("position", "target", "direction"))
_FLAGS = frozenset(("primary", "outside", "fresh"))
_INDICES = frozenset(("owner", "box"))


@dataclass
class _Context:
    rng: np.random.Generator
    crystal: GOSMedium
    surrounding: GOSMedium | None
    mesh: Mesh
    boxes: Sequence[Mesh]
    placement: Placement  # by owner
    sink: Sink
    cascade: Cascade
    clock: Clock | None

    def crystal_has(self, points: np.ndarray, owner: np.ndarray) -> np.ndarray:
        return inside(self.mesh, points, self.placement.take(owner))

    def medium(self, outside: bool) -> GOSMedium:
        if outside:
            assert self.surrounding is not None
            return self.surrounding
        return self.crystal


def _total_free_path(elastic: np.ndarray, inelastic: np.ndarray) -> np.ndarray:
    """``lambdaT``: elastic and inelastic together, or elastic alone."""
    with np.errstate(divide="ignore"):
        return np.where(inelastic > 0, 1 / (1 / elastic + 1 / inelastic), elastic)


def _combined(
    outer: np.ndarray, inner_start: np.ndarray, inverted: np.ndarray
) -> np.ndarray:
    """``gosInelasticLambda``: outer subshells with the starting inner shells.

    Without inner shells at the start, ``inverted`` says whether the Java took
    ``1 / gosOuterLambda`` there, as it does but for an update below 0.5 keV.
    """
    with np.errstate(divide="ignore"):
        return np.where(
            inner_start > 0,
            1 / (1 / outer + 1 / inner_start),
            np.where(inverted, 1 / outer, outer),
        )


def _draw_step(
    ctx: _Context, e: _Electrons, chosen: np.ndarray, elastic: np.ndarray
) -> None:
    """Draw ``s``, the chance of an inelastic collision, and the step's end.

    From the electron's current mean free paths. An electron at rest, whose
    mean free paths are infinite, or one whose direction is undefined, gets
    an undefined step, as in the Java; it stops or leaves before taking it.
    """
    total = _total_free_path(elastic, e.inelastic_free_path[chosen])
    with np.errstate(invalid="ignore"):
        e.step[chosen] = -total * np.log(ctx.rng.random(len(chosen)))
        e.p_inelastic[chosen] = 1 - total / elastic
        e.target[chosen] = (
            e.position[chosen] + e.step[chosen, None] * e.direction[chosen]
        )


def _crystal_state(
    ctx: _Context, e: _Electrons, chosen: np.ndarray, inner_energy: np.ndarray
) -> None:
    """Set up an electron in the crystal, at its start or coming back in."""
    energy = e.energy[chosen]
    inner_start, _ = ctx.crystal.inner_shells(e.start[chosen])
    e.inner_start[chosen] = inner_start
    e.inner_energy[chosen] = inner_energy
    outer, _ = ctx.crystal.outer(energy)
    e.inelastic_free_path[chosen] = _combined(
        outer, inner_start, np.ones(len(chosen), dtype=bool)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        e.p_inner[chosen] = np.where(
            inner_start > 0, e.inelastic_free_path[chosen] / inner_start, 0.0
        )
    _draw_step(ctx, e, chosen, ctx.crystal.elastic_free_path(energy))


def _low_energy_ions(
    ctx: _Context,
    e: _Electrons,
    chosen: np.ndarray,
    energy: np.ndarray,
    points: np.ndarray,
) -> None:
    """The ionisations an electron of under 0.05 keV is taken to make.

    ``int(E / plasmon energy)`` of them, all of one element, drawn as an
    inelastic collision's is: ``chooseLowEnElement``.
    """
    crystal = ctx.crystal
    if not crystal.plasma > 0:
        return
    count = (energy / crystal.plasma).astype(np.int64)
    some = count > 0
    chosen, energy, points, count = (
        chosen[some],
        energy[some],
        points[some],
        count[some],
    )
    if not len(chosen):
        return
    element = _choose_element(ctx, e, chosen)[0]
    ctx.sink.ionise(points, e.owner[chosen], element, count, e.time[chosen])


def _choose_element(
    ctx: _Context, e: _Electrons, chosen: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The element and shell an inelastic collision ionises.

    An inner shell, with the chance ``Pinner``, by Bethe's cross-sections at
    the energy they were last computed at; otherwise an element's valence
    subshell, by the GOS model's at the electron's energy. Returns the
    element (-1 for none: an inner draw where no shell is open), the shell,
    and which were inner.
    """
    crystal = ctx.crystal
    r_inner = ctx.rng.random(len(chosen))
    r_element = ctx.rng.random(len(chosen))
    inner = r_inner < e.p_inner[chosen]
    element = np.full(len(chosen), -1, dtype=np.int64)
    shell = np.full(len(chosen), -1, dtype=np.int64)
    a = np.flatnonzero(inner)
    if len(a):
        _, cumulative = crystal.inner_shells(e.inner_energy[chosen[a]])
        flat = _first_above(cumulative, r_element[a])
        found = flat >= 0
        element[a[found]] = flat[found] // 9
        shell[a[found]] = flat[found] % 9
    b = np.flatnonzero(~inner)
    if len(b):
        _, cumulative = crystal.outer(e.energy[chosen[b]])
        atom = _first_above(cumulative, r_element[b])
        # A draw past the last element, only by rounding: the last element.
        atom = np.where(atom < 0, len(crystal.inner) - 1, atom)
        element[b] = atom
        shell[b] = crystal.inner[atom]
    return element, shell, inner


def _inelastic(
    ctx: _Context, e: _Electrons, chosen: np.ndarray, born: _Born
) -> np.ndarray:
    """An inelastic collision for each electron: returns the W each loses.

    Each has just moved to ``position``; ``direction`` is still the one it
    came in on.
    """
    crystal = ctx.crystal
    rng = ctx.rng
    n = len(chosen)
    energy = e.energy[chosen]
    points = e.position[chosen]
    owner = e.owner[chosen]
    element, shell, inner = _choose_element(ctx, e, chosen)
    plasmon = element < 0

    # The kind of collision: distant longitudinal (0), transverse (1), close (2).
    kind = np.zeros(n, dtype=np.int64)
    r_kind = rng.random(n)
    real = np.flatnonzero(~plasmon)
    if len(real):
        rows = crystal.table.at(energy[real])[
            np.arange(len(real)), element[real], shell[real]
        ]
        with np.errstate(divide="ignore", invalid="ignore"):
            running = np.cumsum(rows[:, :3] / rows[:, 3:4], axis=1)
        hit = _first_above(running, r_kind[real])
        kind[real] = np.maximum(hit, 0)

    safe_element = np.maximum(element, 0)
    safe_shell = np.clip(shell, 0, 9)
    # A valence collision's binding energy is its subshell's, as the model's
    # cross-sections have it; an inner shell's, its edge. As RADDOSE-3D has
    # it since #52: the edges have none for the valence subshells of Al, Si,
    # P, As and Se, whose collisions then recoiled by nothing and lost their
    # electron to an undefined deflection.
    uk = np.where(
        plasmon,
        0.0,
        np.where(
            inner,
            crystal.binding[safe_element, safe_shell],
            crystal.valence_binding[safe_element],
        )
        * 1000,
    )
    wk = np.where(plasmon, 0.0, crystal.wk[safe_element, safe_shell])
    q_ak = np.where(plasmon, wk, qak(energy, wk, uk))
    w_ak = wak(energy, wk, uk)
    w_dis = 3 * w_ak - 2 * uk

    distant = kind < 2
    loss = np.zeros(n)
    q = np.zeros(n)
    theta = np.zeros(n)
    d = np.flatnonzero(distant)
    if len(d):
        loss[d] = np.where(
            plasmon[d], wk[d] / 1000, energy_loss_distant(rng, w_dis[d], uk[d]) / 1000
        )
        q[d] = recoil_distant(rng, energy[d], w_ak[d], q_ak[d])
        longitudinal = kind[d] == 0
        theta[d] = np.where(
            longitudinal, primary_angle_distant(energy[d], q[d], w_ak[d]), 0.0
        )
    c = np.flatnonzero(~distant)
    if len(c):
        k = sample_k(rng, energy[c], uk[c])
        loss[c] = np.where(plasmon[c], wk[c] / 1000, k * (energy[c] + uk[c] / 1000))
        theta[c] = primary_angle_close(energy[c], loss[c])

    secondary = loss - uk / 1000
    freed = np.flatnonzero(secondary > 0)
    time = e.time[chosen]
    if len(freed):
        ctx.sink.ionise(
            points[freed],
            owner[freed],
            element[freed],
            np.ones(len(freed), dtype=np.int64),
            time[freed],
        )
        followed = freed[secondary[freed] > STOPPED_KEV]
        if len(followed):
            angle = np.where(
                distant[followed],
                secondary_angle_distant(energy[followed], w_ak[followed], q[followed]),
                secondary_angle_close(energy[followed], loss[followed]),
            )
            phi = 2 * math.pi * rng.random(len(followed))
            born.add(
                points[followed],
                new_directions(e.direction[chosen[followed]], angle, phi),
                secondary[followed],
                owner[followed],
                time[followed],
            )
        slow = freed[secondary[freed] <= STOPPED_KEV]
        if len(slow):
            ctx.sink.resolved(points[slow], owner[slow], secondary[slow], time[slow])
            _low_energy_ions(ctx, e, chosen[slow], secondary[slow], points[slow])
        # The vacancy relaxes, wherever it is.
        # The vacancy holds the binding energy the collision gave it.
        ctx.cascade.relax(
            points[freed],
            owner[freed],
            element[freed],
            shell[freed],
            born,
            time[freed],
            uk[freed] / 1000,
        )

    phi = 2 * math.pi * rng.random(n)
    e.direction[chosen] = new_directions(e.direction[chosen], theta, phi)
    return np.where(secondary > 0, loss, 0.0)


def _elastic(ctx: _Context, e: _Electrons, chosen: np.ndarray, outside: bool) -> None:
    """Scatter elastically, as the simple model does."""
    if not len(chosen):
        return
    medium = ctx.medium(outside)
    energy = e.energy[chosen]
    sigma = elastic_cross_sections(medium.material, energy)
    cdf = elastic_element_cdf(medium.material, sigma)
    element = _elastic_element(cdf, ctx.rng.random(len(chosen)))
    theta = deflection_angles(medium.z[element], energy, ctx.rng.random(len(chosen)))
    phi = 2 * math.pi * ctx.rng.random(len(chosen))
    e.direction[chosen] = new_directions(e.direction[chosen], theta, phi)


def _setup(ctx: _Context, e: _Electrons, fresh: np.ndarray) -> np.ndarray:
    """The start of ``trackPhotoelectron``: returns which have ended at once."""
    e.fresh[fresh] = False
    ended = np.zeros(len(e), dtype=bool)
    inside_crystal = fresh[~e.outside[fresh]]
    if len(inside_crystal):
        _crystal_state(ctx, e, inside_crystal, e.start[inside_crystal])
    in_surrounding = fresh[e.outside[fresh]]
    if len(in_surrounding):
        # No GOS model out there: the electron scatters elastically only.
        e.inelastic_free_path[in_surrounding] = 0.0
        e.p_inner[in_surrounding] = 0.0
        e.inner_start[in_surrounding] = 0.0
        e.inner_energy[in_surrounding] = e.start[in_surrounding]
        _draw_step(
            ctx,
            e,
            in_surrounding,
            ctx.medium(True).elastic_free_path(e.energy[in_surrounding]),
        )

    # Freed with under 0.05 keV: left where it was, if that is in the crystal
    # (and, in the XFEL subprogram, before lastTime).
    low = fresh[e.start[fresh] < STOPPED_KEV]
    if len(low):
        ended[low] = True
        there = low[ctx.crystal_has(e.position[low], e.owner[low])]
        if ctx.clock is not None:
            there = there[e.time[there] < ctx.clock.last]
        primary = there[e.primary[there]]
        ctx.sink.dose(e.position[primary], e.owner[primary], e.start[primary])
        # An Auger electron's too, as RADDOSE-3D has it since #53: it was lost.
        ctx.sink.resolved(
            e.position[there], e.owner[there], e.start[there], e.time[there]
        )
        crystal_side = there[~e.outside[there]]
        points = e.position[crystal_side].copy()
        points[:, :2] = e.target[crystal_side, :2]
        _low_energy_ions(ctx, e, crystal_side, e.start[crystal_side], points)
    return ended


def track_gos(
    rng: np.random.Generator,
    crystal: GOSMedium,
    surrounding: GOSMedium | None,
    mesh: Mesh,
    boxes: Sequence[Mesh],
    placement: Placement,
    sink: Sink,
    position: np.ndarray,
    direction: np.ndarray,
    energy: np.ndarray,
    owner: np.ndarray,
    primary: np.ndarray,
    in_surrounding: np.ndarray,
    box: np.ndarray,
    vacancies: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    start_time: np.ndarray | None = None,
    clock: Clock | None = None,
) -> int:
    """Follow every electron, and every one they free, to its end.

    Until all have stopped or left.

    Parameters
    ----------
    rng : numpy.random.Generator
        The draws.
    crystal, surrounding : GOSMedium
        The crystal, and its surrounding if it has one.
    mesh : Mesh
        The crystal's shape.
    boxes : Sequence[Mesh]
        The surrounding's extent.
    sink : Sink
        Where energy and ionisations go.
    placement : Placement
        How the crystal stands for each owner: the photon an electron's
        cascade began with.
    position, direction, energy, owner, primary, in_surrounding, box
        The first electrons.
    vacancies : tuple, optional
        Vacancies to relax first: points, owners, elements, shells -- the
        photoelectric absorptions' in the crystal.
    start_time : numpy.ndarray, optional
        By owner: when its photon was absorbed, fs, and so when its first
        electrons and vacancies begin. 0 if not given.
    clock : Clock, optional
        The XFEL subprogram's: follow the electrons in time, and stop at
        ``lastTime``. It has no surrounding.

    Returns
    -------
    int
        How many passes it took.
    """
    if clock is not None and surrounding is not None:
        raise ValueError("the XFEL subprogram's surrounding is not this one")
    cascade = Cascade(rng, crystal, sink, clock)
    ctx = _Context(
        rng, crystal, surrounding, mesh, boxes, placement, sink, cascade, clock
    )
    if start_time is None:
        start_time = np.zeros(int(np.max(owner, initial=-1)) + 1)
    e = _Electrons.blank(0).extend(
        position, direction, energy, owner, False, in_surrounding, box,
        start_time[owner],
    )  # fmt: skip
    e.primary[:] = np.asarray(primary, dtype=bool)
    born = _Born()
    if vacancies is not None:
        cascade.relax(*vacancies, born, start_time[vacancies[1]])
    e = _join(e, born, crystal)

    passes = 0
    while len(e):
        passes += 1
        ended = _setup(ctx, e, np.flatnonzero(e.fresh))
        born = _Born()
        live = ~ended

        # Coming back into the crystal: to the face it crosses, the crystal's
        # cross-sections, with the inner shells' at the starting energy, and a
        # fresh step.
        first_in = live & ctx.crystal_has(e.target, e.owner)
        back = np.flatnonzero(first_in & e.outside)
        if len(back):
            to_face = exit_distance(
                mesh, e.position[back], e.direction[back], placement.take(e.owner[back])
            )
            e.position[back] += to_face[:, None] * e.direction[back]
            e.outside[back] = False
            _crystal_state(ctx, e, back, e.start[back])
        in_crystal = live & ~e.outside
        in_crystal[in_crystal] = ctx.crystal_has(
            e.target[in_crystal], e.owner[in_crystal]
        )

        a = np.flatnonzero(in_crystal)
        if len(a):
            ended |= _in_crystal(ctx, e, a, born)

        leaving = np.flatnonzero(live & ~in_crystal & ~e.outside)
        if len(leaving):
            ended |= _leave(ctx, e, leaving)

        wandering = np.flatnonzero(live & ~in_crystal & e.outside & ~ended)
        wandering = wandering[~np.isin(wandering, leaving)]
        if len(wandering):
            ended |= _wander(ctx, e, wandering)

        stopped = np.flatnonzero(live & ~ended & (e.energy < STOPPED_KEV))
        if len(stopped):
            ended[stopped] = True
            there = stopped[ctx.crystal_has(e.position[stopped], e.owner[stopped])]
            primary_there = there[e.primary[there]]
            sink.dose(
                e.position[primary_there],
                e.owner[primary_there],
                e.energy[primary_there],
            )
            sink.resolved(
                e.position[there], e.owner[there], e.energy[there], e.time[there]
            )
            # Its ionisations' exposure is judged where its next step would
            # have ended, as the Java passes it.
            points = e.position[there].copy()
            points[:, :2] = e.target[there, :2]
            _low_energy_ions(ctx, e, there, e.energy[there], points)

        e = _join(e.take(~ended), born, crystal)
    return passes


def _join(e: _Electrons, born: _Born, crystal: GOSMedium) -> _Electrons:
    """The freed electrons, in the crystal, joining those still followed."""
    if not born.energies:
        return e
    energies = np.concatenate(born.energies)
    return e.extend(
        np.concatenate(born.points),
        np.concatenate(born.directions),
        energies,
        np.concatenate(born.owners),
        False,
        np.zeros(len(energies), dtype=bool),
        np.zeros(len(energies), dtype=np.int64),
        np.concatenate(born.times),
    )


def _in_crystal(ctx: _Context, e: _Electrons, a: np.ndarray, born: _Born) -> np.ndarray:
    """A step that ends in the crystal: deposit, collide, update, step on.

    With a clock, the step first takes its flight time, and one that ends
    after ``lastTime`` ends the electron there. Returns which have ended.
    """
    crystal = ctx.crystal
    ended = np.zeros(len(e), dtype=bool)
    if ctx.clock is not None:
        e.time[a] += ctx.clock.flight(e.energy[a], e.step[a])
        late = e.time[a] > ctx.clock.last
        ended[a[late]] = True
        a = a[~late]
    energy = e.energy[a]
    primary = a[e.primary[a]]
    ctx.sink.dose(
        e.target[primary],
        e.owner[primary],
        e.step[primary] * crystal.stopping_power(e.energy[primary]),
    )
    e.position[a] = e.target[a]

    loss = np.zeros(len(a))
    inelastic = ctx.rng.random(len(a)) < e.p_inelastic[a]
    hit = np.flatnonzero(inelastic)
    if len(hit):
        loss[hit] = _inelastic(ctx, e, a[hit], born)
    _elastic(ctx, e, a[~inelastic], False)

    # Brought up to date after losing energy, and always below 0.5 keV.
    above = energy > _UPDATE_ABOVE
    update = np.flatnonzero(~above | (loss > 0))
    if len(update):
        u = a[update]
        e.energy[u] -= loss[update]
        outer, _ = crystal.outer(e.energy[u])
        e.inelastic_free_path[u] = _combined(outer, e.inner_start[u], above[update])
        e.inner_current[u], _ = crystal.inner_shells(e.energy[u])
        e.inner_energy[u] = e.energy[u]
    # After every step, from the last inner-shell update -- just back in the
    # crystal, the surrounding's, from when it left.
    current = e.inner_current[a]
    with np.errstate(divide="ignore", invalid="ignore"):
        e.p_inner[a] = np.where(
            current > 0, e.inelastic_free_path[a] / current, e.p_inner[a]
        )
    going = a[e.energy[a] >= 0]
    _draw_step(
        ctx, e, going, crystal.elastic_free_path(np.maximum(e.energy[going], 1e-9))
    )
    return ended


def _leave(ctx: _Context, e: _Electrons, leaving: np.ndarray) -> np.ndarray:
    """Out of the crystal: lost, or on into the surrounding.

    Just past the face, with the surrounding's cross-sections for one step.
    Returns which have ended.
    """
    ended = np.zeros(len(e), dtype=bool)
    if ctx.surrounding is None:
        ended[leaving] = True
        return ended
    to_edge = exit_distance(
        ctx.mesh,
        e.position[leaving],
        e.direction[leaving],
        ctx.placement.take(e.owner[leaving]),
    )
    e.position[leaving] += (to_edge + to_edge / 1000)[:, None] * e.direction[leaving]
    e.outside[leaving] = True
    on = leaving[e.energy[leaving] > STOPPED_KEV]
    if len(on):
        surrounding = ctx.surrounding
        outer, _ = surrounding.outer(e.energy[on])
        e.inelastic_free_path[on] = _combined(
            outer, e.inner_start[on], np.ones(len(on), dtype=bool)
        )
        inner, _ = surrounding.inner_shells(e.energy[on])
        e.inner_current[on] = inner
        with np.errstate(divide="ignore", invalid="ignore"):
            e.p_inner[on] = np.where(
                inner > 0, e.inelastic_free_path[on] / inner, e.p_inner[on]
            )
        _draw_step(ctx, e, on, surrounding.elastic_free_path(e.energy[on]))
    return ended


def _wander(ctx: _Context, e: _Electrons, wandering: np.ndarray) -> np.ndarray:
    """In the surrounding: scatter and slow, depositing nothing.

    Until it leaves the surrounding. Returns which have ended.
    """
    ended = np.zeros(len(e), dtype=bool)
    still = np.zeros(len(wandering), dtype=bool)
    for index in np.unique(e.box[wandering]):
        group = np.flatnonzero(e.box[wandering] == index)
        still[group] = inside(
            ctx.boxes[index],
            e.target[wandering[group]],
            ctx.placement.take(e.owner[wandering[group]]),
        )
    ended[wandering[~still]] = True
    c = wandering[still]
    if len(c):
        surrounding = ctx.medium(True)
        loss = e.step[c] * surrounding.stopping_power(e.energy[c])
        e.position[c] = e.target[c]
        _elastic(ctx, e, c, True)
        e.energy[c] -= loss
        going = c[e.energy[c] >= STOPPED_KEV]
        if len(going):
            e.inelastic_free_path[going] = 0.0
            _draw_step(ctx, e, going, surrounding.elastic_free_path(e.energy[going]))
    return ended
