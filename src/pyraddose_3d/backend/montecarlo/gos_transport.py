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
per-electron state: its mean free paths and the chance an inelastic
collision is with an inner shell, at its energy, brought up to date as it
loses energy and when it comes back into the crystal. Until RADDOSE-3D #55
the inner shells' mean free path stayed at the starting energy's, an
inverse length was taken for a length where none was open at the start,
and the chance went stale.

The XFEL subprogram follows the same electrons with a :class:`Clock`: each
carries the time since the pulse began, which its steps advance by their
flight time and a relaxation by the transition's lifetime, and nothing is
followed past ``lastTime``, when the pulse's last photon has left the
crystal. Every deposit and ionisation is passed on with its time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

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
from pyraddose_3d.backend.montecarlo.ops import namespace, on, random_namespace
from pyraddose_3d.backend.montecarlo.photons import (
    isotropic_directions,
    stacked_shells,
)
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
        ``lastTime``: the pulse's length, its window for a Gaussian pulse,
        and the time light takes to cross the crystal. An electron whose step ends later
        is dropped, with the energy it had; a vacancy that would relax later
        is left as it is.
    """

    last: float

    @staticmethod
    def flight(energy: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """``getTimeToDistance``: fs for ``distance`` nm at ``energy`` keV."""
        xp = namespace(energy)
        rest = _ELECTRON_MASS * _SPEED_OF_LIGHT**2
        beta_squared = 1 - ((rest / (energy * KEV_TO_JOULES + rest)) ** 2)
        speed = xp.sqrt(beta_squared * _SPEED_OF_LIGHT**2) * 1e9 / 1e15  # nm/fs
        return (1 / speed) * distance

    @staticmethod
    def lifetime(linewidth_ev: np.ndarray) -> np.ndarray:
        """A transition's lifetime from its natural linewidth: hbar over it, fs."""
        return 1e15 * (
            (_PLANCK / (2 * math.pi)) / ((linewidth_ev / 1000) * KEV_TO_JOULES)
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
        xp = namespace(energy)
        values = self.table.column(energy, self.inner, 3)
        # Summed element after element, as the Java's loop does: a sum over
        # the axis would be pairwise for eight elements or more.
        running = xp.cumsum(values, axis=1)
        total = running[:, -1]
        with xp.errstate(divide="ignore", invalid="ignore"):
            return 1 / total, running / total[:, None]

    def inner_shells(self, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``betheIonisationxSection`` and ``getAllShellProbs`` at each energy.

        Returns the mean free path, nm (0 where no shell is open), and the
        running chances over every element's nine shells in turn,
        ``(len(energy), elements * 9)``: NaN where no shell is open.
        """
        xp = namespace(energy)
        sigma = bethe_shells(self.material, energy).reshape(len(energy), -1)
        total = sigma.sum(axis=1)
        with xp.errstate(divide="ignore", invalid="ignore"):
            free_path = xp.where(total > 0, 1 / total, 0.0)
            return free_path, xp.cumsum(sigma, axis=1) / total[:, None]

    def elastic_free_path(self, energy: np.ndarray) -> np.ndarray:
        """``getElectronElasticMFPL``, nm."""
        sigma = elastic_cross_sections(self.material, energy)
        return elastic_mean_free_path(self.material, sigma)

    def stopping_power(self, energy: np.ndarray) -> np.ndarray:
        """``getStoppingPower``, keV/nm."""
        return stopping_power(self.material, energy)


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
    binding[:, :9] = np.nan_to_num(
        stacked_shells(tuple(atom.z for atom in atoms)).binding
    )
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
    xp = namespace(uniform)
    above = cumulative > uniform[:, None]
    return xp.where(xp.any(above, axis=1), xp.argmax(above, axis=1), -1)


def _relaxes(z: np.ndarray, shell: np.ndarray) -> np.ndarray:
    """``produceAugerElectron``'s test: which vacancies the tables relax."""
    return namespace(z).isin(z, list(_RELAXING)) & (
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
    RADDOSE-3D has it since #54: the vacancies' energy was counted twice,
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
        self.xp = xp = random_namespace(rng)
        self.crystal = crystal
        self.sink = sink
        self.clock = clock
        self.yields = xp.zeros((len(crystal.z), _SHELL_COLUMNS))
        self.yields[:, :9] = on(
            xp, stacked_shells(tuple(crystal.z.tolist()))
        ).fluorescence_yield
        self.auger = on(xp, tables.packed_auger())
        self.fluorescence = on(xp, tables.packed_fluorescence())

    def _transition(
        self, table: tables.PackedTransitions, z: np.ndarray, shell: np.ndarray
    ) -> np.ndarray:
        """``getTransitionIndex`` for each vacancy: slot 0 if no draw falls.

        NumPy's draws are made a group of vacancies of one element and shell
        at a time, the groups in a set's order, as they always were; torch's
        (:class:`~.ops.TorchRandom`), for every vacancy at once. The tables
        are then read for every vacancy at once.
        """
        xp = self.xp
        if getattr(self.rng, "draws_in_groups", True):
            uniform = xp.empty(len(z))
            for key in set(zip(z.tolist(), shell.tolist(), strict=True)):
                chosen = xp.flatnonzero((z == key[0]) & (shell == key[1]))
                uniform[chosen] = self.rng.random(len(chosen))
        else:
            uniform = self.rng.random(len(z))
        return xp.maximum(_first_above(table.cumulative[z, shell], uniform), 0)

    def _edge(self, element: np.ndarray, shell: np.ndarray) -> np.ndarray:
        """``getShellBindingEnergy``: a vacancy's binding energy, keV; 0 past M5."""
        clamped = self.xp.minimum(shell, _SHELL_COLUMNS - 1)
        return self.crystal.binding[element, clamped]

    def _lifetimes(
        self,
        table: tables.PackedTransitions,
        z: np.ndarray,
        shell: np.ndarray,
        index: np.ndarray,
    ) -> np.ndarray:
        """How long each transition takes, fs; 0 without a clock.

        Drawn from the exponential its lifetime is the mean of, as since
        RADDOSE-3D #58: it took the lifetime itself, every time.
        """
        xp = self.xp
        if self.clock is None:
            return xp.zeros(len(z))
        linewidth = table.linewidth_ev[z, shell, index]
        draws = -xp.log(1 - self.rng.random(len(z)))
        return self.clock.lifetime(linewidth) * draws

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
        xp = self.xp
        if time is None:
            time = xp.zeros(len(element))
        if held is None:
            held = crystal.binding[element, xp.minimum(shell, _SHELL_COLUMNS - 1)]
        for _ in range(_MAX_GENERATIONS):
            if self.clock is not None:
                on_time = xp.flatnonzero(time < self.clock.last)
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
            shell = xp.minimum(shell, _SHELL_COLUMNS - 1)
            z = crystal.z[element]
            binding = held
            relaxing = _relaxes(z, shell)
            if (~relaxing).any():
                rest = xp.flatnonzero(~relaxing)
                self.sink.resolved(points[rest], owner[rest], binding[rest], time[rest])
                self.sink.dose(points[rest], owner[rest], binding[rest])
            chosen = xp.flatnonzero(relaxing)
            fluoresces = ~(
                self.rng.random(len(chosen))
                > self.yields[element[chosen], shell[chosen]]
            )

            auger = chosen[~fluoresces]
            index = self._transition(self.auger, z[auger], shell[auger])
            slot = (z[auger], shell[auger], index)
            energy = self.auger.energy_kev[slot]
            exits = self.auger.exit_shell[slot]
            drops_a = self.auger.drop_shell[slot]
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
                xp.flatnonzero(later_a < self.clock.last)
                if self.clock is not None
                else xp.arange(len(auger))
            )
            self.sink.ionise(
                points[auger[emitted]],
                owner[auger[emitted]],
                element[auger[emitted]],
                xp.ones(len(emitted), dtype=xp.int64),
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
            slot = (z[light], shell[light], index)
            drops_f = self.fluorescence.drop_shell[slot]
            # What the photon, which escapes, and the vacancy left do not carry.
            photon = self.fluorescence.energy_kev[slot]
            left = binding[light] - photon - self._edge(element[light], drops_f)
            self.sink.resolved(points[light], owner[light], left, time[light])
            later_f = time[light] + self._lifetimes(
                self.fluorescence, z[light], shell[light], index
            )

            # The vacancies left: the Auger electron's and the one filled
            # from, then the fluorescence's, each relaxed in turn.
            points = xp.concatenate([points[auger], points[auger], points[light]])
            owner = xp.concatenate([owner[auger], owner[auger], owner[light]])
            element = xp.concatenate([element[auger], element[auger], element[light]])
            shell = xp.concatenate([exits, drops_a, drops_f])
            time = xp.concatenate([later_a, later_a, later_f])
            held = crystal.binding[element, xp.minimum(shell, _SHELL_COLUMNS - 1)]


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
    p_inner: np.ndarray  # Pinner
    p_inelastic: np.ndarray  # Pinel
    inner_energy: np.ndarray  # where ionisationProbs was last computed
    inner_current: np.ndarray  # innerShellLamda, at inner_energy
    time: np.ndarray  # timeStamp, fs

    @classmethod
    def blank(cls, n: int, xp: Any = np) -> _Electrons:
        """``n`` electrons, every field zero, false or fresh."""
        arrays: dict[str, np.ndarray] = {}
        for name in cls.__dataclass_fields__:
            if name in _VECTORS:
                arrays[name] = xp.zeros((n, 3))
            elif name in _FLAGS:
                arrays[name] = xp.full(n, name == "fresh")
            elif name in _INDICES:
                arrays[name] = xp.zeros(n, dtype=xp.int64)
            else:
                arrays[name] = xp.zeros(n)
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
        xp = namespace(energies)
        new = _Electrons.blank(len(energies), xp)
        new.position = xp.asarray(points, dtype=float)
        new.target = xp.copy(new.position)
        new.direction = xp.asarray(directions, dtype=float)
        new.energy = xp.asarray(energies, dtype=float)
        new.start = xp.copy(new.energy)
        new.primary[:] = primary
        new.outside = xp.asarray(outside, dtype=bool)
        new.owner = xp.asarray(owners, dtype=xp.int64)
        new.box = xp.asarray(box, dtype=xp.int64)
        new.time = xp.asarray(times, dtype=float)
        return _Electrons(
            **{
                name: xp.concatenate([getattr(self, name), getattr(new, name)])
                for name in self.__dataclass_fields__
            }
        )


#: ``_Electrons``' fields of three components, of flags, and of indices.
_VECTORS = frozenset(("position", "target", "direction"))
_FLAGS = frozenset(("primary", "outside", "fresh"))
_INDICES = frozenset(("owner", "box"))


@dataclass
class _Context:
    xp: Any
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
    xp = namespace(elastic)
    with xp.errstate(divide="ignore"):
        return xp.where(inelastic > 0, 1 / (1 / elastic + 1 / inelastic), elastic)


def _combined(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """``gosInelasticLambda``: the outer subshells with the inner shells."""
    xp = namespace(outer)
    with xp.errstate(divide="ignore"):
        return xp.where(inner > 0, 1 / (1 / outer + 1 / inner), outer)


def _inner_chance(free_path: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """``Pinner``: the chance an inelastic collision is with an inner shell."""
    xp = namespace(free_path)
    with xp.errstate(divide="ignore", invalid="ignore"):
        return xp.where(inner > 0, free_path / inner, 0.0)


def _draw_step(
    ctx: _Context, e: _Electrons, chosen: np.ndarray, elastic: np.ndarray
) -> None:
    """Draw ``s``, the chance of an inelastic collision, and the step's end.

    From the electron's current mean free paths. An electron at rest, whose
    mean free paths are infinite, or one whose direction is undefined, gets
    an undefined step, as in the Java; it stops or leaves before taking it.
    """
    xp = ctx.xp
    total = _total_free_path(elastic, e.inelastic_free_path[chosen])
    with xp.errstate(invalid="ignore"):
        e.step[chosen] = -total * xp.log(ctx.rng.random(len(chosen)))
        e.p_inelastic[chosen] = 1 - total / elastic
        e.target[chosen] = (
            e.position[chosen] + e.step[chosen, None] * e.direction[chosen]
        )


def _crystal_state(ctx: _Context, e: _Electrons, chosen: np.ndarray) -> None:
    """Set up an electron in the crystal, at its start or coming back in."""
    energy = e.energy[chosen]
    inner, _ = ctx.crystal.inner_shells(energy)
    e.inner_current[chosen] = inner
    e.inner_energy[chosen] = energy
    outer, _ = ctx.crystal.outer(energy)
    e.inelastic_free_path[chosen] = _combined(outer, inner)
    e.p_inner[chosen] = _inner_chance(e.inelastic_free_path[chosen], inner)
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
    xp = ctx.xp
    crystal = ctx.crystal
    if not crystal.plasma > 0:
        return
    count = xp.astype(energy / crystal.plasma, xp.int64)
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
    xp = ctx.xp
    crystal = ctx.crystal
    r_inner = ctx.rng.random(len(chosen))
    r_element = ctx.rng.random(len(chosen))
    inner = r_inner < e.p_inner[chosen]
    element = xp.full(len(chosen), -1, dtype=xp.int64)
    shell = xp.full(len(chosen), -1, dtype=xp.int64)
    a = xp.flatnonzero(inner)
    if len(a):
        _, cumulative = crystal.inner_shells(e.inner_energy[chosen[a]])
        flat = _first_above(cumulative, r_element[a])
        found = flat >= 0
        element[a[found]] = flat[found] // 9
        shell[a[found]] = flat[found] % 9
    b = xp.flatnonzero(~inner)
    if len(b):
        _, cumulative = crystal.outer(e.energy[chosen[b]])
        atom = _first_above(cumulative, r_element[b])
        # A draw past the last element, only by rounding: the last element.
        atom = xp.where(atom < 0, len(crystal.inner) - 1, atom)
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
    xp = ctx.xp
    crystal = ctx.crystal
    rng = ctx.rng
    n = len(chosen)
    energy = e.energy[chosen]
    points = e.position[chosen]
    owner = e.owner[chosen]
    element, shell, inner = _choose_element(ctx, e, chosen)
    plasmon = element < 0

    # The kind of collision: distant longitudinal (0), transverse (1), close (2).
    kind = xp.zeros(n, dtype=xp.int64)
    r_kind = rng.random(n)
    real = xp.flatnonzero(~plasmon)
    if len(real):
        rows = crystal.table.rows(energy[real], element[real], shell[real])
        with xp.errstate(divide="ignore", invalid="ignore"):
            running = xp.cumsum(rows[:, :3] / rows[:, 3:4], axis=1)
        hit = _first_above(running, r_kind[real])
        kind[real] = xp.maximum(hit, 0)

    safe_element = xp.maximum(element, 0)
    safe_shell = xp.clip(shell, 0, 9)
    # A valence collision's binding energy is its subshell's, as the model's
    # cross-sections have it; an inner shell's, its edge. As RADDOSE-3D has
    # it since #52: the edges have none for the valence subshells of Al, Si,
    # P, As and Se, whose collisions then recoiled by nothing and lost their
    # electron to an undefined deflection.
    uk = xp.where(
        plasmon,
        0.0,
        xp.where(
            inner,
            crystal.binding[safe_element, safe_shell],
            crystal.valence_binding[safe_element],
        )
        * 1000,
    )
    wk = xp.where(plasmon, 0.0, crystal.wk[safe_element, safe_shell])
    q_ak = xp.where(plasmon, wk, qak(energy, wk, uk))
    w_ak = wak(energy, wk, uk)
    w_dis = 3 * w_ak - 2 * uk

    distant = kind < 2
    loss = xp.zeros(n)
    q = xp.zeros(n)
    theta = xp.zeros(n)
    d = xp.flatnonzero(distant)
    if len(d):
        loss[d] = xp.where(
            plasmon[d], wk[d] / 1000, energy_loss_distant(rng, w_dis[d], uk[d]) / 1000
        )
        q[d] = recoil_distant(rng, energy[d], w_ak[d], q_ak[d])
        longitudinal = kind[d] == 0
        theta[d] = xp.where(
            longitudinal, primary_angle_distant(energy[d], q[d], w_ak[d]), 0.0
        )
    c = xp.flatnonzero(~distant)
    if len(c):
        k = sample_k(rng, energy[c], uk[c])
        loss[c] = xp.where(plasmon[c], wk[c] / 1000, k * (energy[c] + uk[c] / 1000))
        theta[c] = primary_angle_close(energy[c], loss[c])

    secondary = loss - uk / 1000
    freed = xp.flatnonzero(secondary > 0)
    time = e.time[chosen]
    if len(freed):
        ctx.sink.ionise(
            points[freed],
            owner[freed],
            element[freed],
            xp.ones(len(freed), dtype=xp.int64),
            time[freed],
        )
        followed = freed[secondary[freed] > STOPPED_KEV]
        if len(followed):
            angle = xp.where(
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
    return xp.where(secondary > 0, loss, 0.0)


def _elastic(ctx: _Context, e: _Electrons, chosen: np.ndarray, outside: bool) -> None:
    """Scatter elastically, as the simple model does."""
    if not len(chosen):
        return
    medium = ctx.medium(outside)
    energy = e.energy[chosen]
    sigma = elastic_cross_sections(medium.material, energy)
    cdf = elastic_element_cdf(medium.material, sigma)
    element = _elastic_element(cdf, ctx.rng.random(len(chosen)))
    theta = deflection_angles(
        medium.z[element],
        energy,
        ctx.rng.random(len(chosen)),
        tuple(atom.z for atom in medium.material.atoms),
    )
    phi = 2 * math.pi * ctx.rng.random(len(chosen))
    e.direction[chosen] = new_directions(e.direction[chosen], theta, phi)


def _setup(ctx: _Context, e: _Electrons, fresh: np.ndarray) -> np.ndarray:
    """The start of ``trackPhotoelectron``: returns which have ended at once."""
    xp = ctx.xp
    e.fresh[fresh] = False
    ended = xp.zeros(len(e), dtype=bool)
    inside_crystal = fresh[~e.outside[fresh]]
    if len(inside_crystal):
        _crystal_state(ctx, e, inside_crystal)
    in_surrounding = fresh[e.outside[fresh]]
    if len(in_surrounding):
        # No GOS model out there: the electron scatters elastically only.
        e.inelastic_free_path[in_surrounding] = 0.0
        e.p_inner[in_surrounding] = 0.0
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
        # An Auger electron's too, as RADDOSE-3D has it since #54: it was lost.
        ctx.sink.resolved(
            e.position[there], e.owner[there], e.start[there], e.time[there]
        )
        crystal_side = there[~e.outside[there]]
        points = xp.copy(e.position[crystal_side])
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
        ``lastTime``.

    Returns
    -------
    int
        How many passes it took.
    """
    xp = random_namespace(rng)
    cascade = Cascade(rng, crystal, sink, clock)
    ctx = _Context(
        xp, rng, crystal, surrounding, mesh, boxes, placement, sink, cascade, clock
    )
    if start_time is None:
        start_time = xp.zeros(int(xp.max(owner, initial=-1)) + 1)
    e = _Electrons.blank(0, xp).extend(
        position, direction, energy, owner, False, in_surrounding, box,
        start_time[owner],
    )  # fmt: skip
    e.primary[:] = xp.asarray(primary, dtype=bool)
    born = _Born()
    if vacancies is not None:
        cascade.relax(*vacancies, born, start_time[vacancies[1]])
    e = _join(e, born, crystal)

    passes = 0
    while len(e):
        passes += 1
        ended = _setup(ctx, e, xp.flatnonzero(e.fresh))
        born = _Born()
        live = ~ended

        # Coming back into the crystal: to the face it crosses, the crystal's
        # cross-sections, with the inner shells' at the starting energy, and a
        # fresh step.
        first_in = live & ctx.crystal_has(e.target, e.owner)
        back = xp.flatnonzero(first_in & e.outside)
        if len(back):
            to_face = exit_distance(
                mesh, e.position[back], e.direction[back], placement.take(e.owner[back])
            )
            if clock is not None:
                e.time[back] += clock.flight(e.energy[back], to_face)
                late = e.time[back] > clock.last
                ended[back[late]] = True
                first_in[back[late]] = False
                back, to_face = back[~late], to_face[~late]
            if len(back):
                e.position[back] += to_face[:, None] * e.direction[back]
                e.outside[back] = False
                _crystal_state(ctx, e, back)
        in_crystal = live & ~e.outside
        checked = xp.flatnonzero(in_crystal)
        in_crystal[checked] = ctx.crystal_has(e.target[checked], e.owner[checked])

        a = xp.flatnonzero(in_crystal)
        if len(a):
            ended |= _in_crystal(ctx, e, a, born)

        leaving = xp.flatnonzero(live & ~in_crystal & ~e.outside)
        if len(leaving):
            ended |= _leave(ctx, e, leaving)

        wandering = xp.flatnonzero(live & ~in_crystal & e.outside & ~ended)
        wandering = wandering[~xp.isin(wandering, leaving)]
        if len(wandering):
            ended |= _wander(ctx, e, wandering)

        stopped = xp.flatnonzero(live & ~ended & (e.energy < STOPPED_KEV))
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
            points = xp.copy(e.position[there])
            points[:, :2] = e.target[there, :2]
            _low_energy_ions(ctx, e, there, e.energy[there], points)

        e = _join(e.take(~ended), born, crystal)
    return passes


def _join(e: _Electrons, born: _Born, crystal: GOSMedium) -> _Electrons:
    """The freed electrons, in the crystal, joining those still followed."""
    xp = namespace(e.energy)
    if not born.energies:
        return e
    energies = xp.concatenate(born.energies)
    return e.extend(
        xp.concatenate(born.points),
        xp.concatenate(born.directions),
        energies,
        xp.concatenate(born.owners),
        False,
        xp.zeros(len(energies), dtype=bool),
        xp.zeros(len(energies), dtype=xp.int64),
        xp.concatenate(born.times),
    )


def _in_crystal(ctx: _Context, e: _Electrons, a: np.ndarray, born: _Born) -> np.ndarray:
    """A step that ends in the crystal: deposit, collide, update, step on.

    With a clock, the step first takes its flight time, and one that ends
    after ``lastTime`` ends the electron there. Returns which have ended.
    """
    xp = ctx.xp
    crystal = ctx.crystal
    ended = xp.zeros(len(e), dtype=bool)
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

    loss = xp.zeros(len(a))
    inelastic = ctx.rng.random(len(a)) < e.p_inelastic[a]
    hit = xp.flatnonzero(inelastic)
    if len(hit):
        loss[hit] = _inelastic(ctx, e, a[hit], born)
    _elastic(ctx, e, a[~inelastic], False)

    # Brought up to date after losing energy, and always below 0.5 keV.
    above = energy > _UPDATE_ABOVE
    update = xp.flatnonzero(~above | (loss > 0))
    if len(update):
        u = a[update]
        e.energy[u] -= loss[update]
        outer, _ = crystal.outer(e.energy[u])
        e.inner_current[u], _ = crystal.inner_shells(e.energy[u])
        e.inelastic_free_path[u] = _combined(outer, e.inner_current[u])
        e.inner_energy[u] = e.energy[u]
    e.p_inner[a] = _inner_chance(e.inelastic_free_path[a], e.inner_current[a])
    going = a[e.energy[a] >= 0]
    _draw_step(
        ctx, e, going, crystal.elastic_free_path(xp.maximum(e.energy[going], 1e-9))
    )
    return ended


def _leave(ctx: _Context, e: _Electrons, leaving: np.ndarray) -> np.ndarray:
    """Out of the crystal: lost, or on into the surrounding.

    Just past the face, scattering elastically, as it does from then on out
    there: the GOS model is not used in the surrounding. Until RADDOSE-3D #55
    the first step there took its inelastic mean free path too. Returns
    which have ended.
    """
    xp = ctx.xp
    ended = xp.zeros(len(e), dtype=bool)
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
        e.inelastic_free_path[on] = 0.0
        e.p_inner[on] = 0.0
        _draw_step(ctx, e, on, ctx.surrounding.elastic_free_path(e.energy[on]))
    return ended


def _wander(ctx: _Context, e: _Electrons, wandering: np.ndarray) -> np.ndarray:
    """In the surrounding: scatter and slow, depositing nothing.

    Until it leaves the surrounding. Returns which have ended.
    """
    xp = ctx.xp
    ended = xp.zeros(len(e), dtype=bool)
    still = xp.zeros(len(wandering), dtype=bool)
    for index, mesh in enumerate(ctx.boxes):
        group = xp.flatnonzero(e.box[wandering] == index)
        if len(group):
            still[group] = inside(
                mesh,
                e.target[wandering[group]],
                ctx.placement.take(e.owner[wandering[group]]),
            )
    ended[wandering[~still]] = True
    c = wandering[still]
    if ctx.clock is not None and len(c):
        e.time[c] += ctx.clock.flight(e.energy[c], e.step[c])
        late = e.time[c] > ctx.clock.last
        ended[c[late]] = True
        c = c[~late]
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
