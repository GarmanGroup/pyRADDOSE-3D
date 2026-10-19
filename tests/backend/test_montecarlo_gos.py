"""The GOS model's parts, one at a time.

Its whole runs are compared with RADDOSE-3D statistically, in
``tests/conformance/test_montecarlo_files.py``. The values here were printed
by RADDOSE-3D at ``master`` ``85a6813`` for case 160's insulin crystal:
``CoefCalcCompute.betheIonisationxSection``, ``getAllShellProbs``,
``getGOSInel``, ``getGOSOuterLambda`` and ``getGOSVariable``.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from pyraddose_3d.backend.microed import (
    _SUBSHELL_ELECTRONS,
    ElectronMaterial,
    _Atom,
    _shell_table,
    inelastic_shells,
)
from pyraddose_3d.backend.montecarlo import gos
from pyraddose_3d.backend.montecarlo.gos_transport import Cascade, _Born, gos_medium
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import monte_carlo_setup

#: Case 140's crystal is case 160's.
CASE = "tests/conformance/cases/140-montecarlo-baseline/config.yaml"


@pytest.fixture(scope="module")
def material():  # type: ignore[no-untyped-def]
    return monte_carlo_setup(Experiment.from_yaml(CASE)).material


def _row(material, z: int) -> int:  # type: ignore[no-untyped-def]
    return next(i for i, atom in enumerate(material.atoms) if atom.z == z)


@pytest.mark.parametrize(
    ("energy", "bethe", "total", "outer", "carbon_k", "zinc"),
    [
        (
            1.0,
            1057.4484266166540,
            3.2738005699852210,
            3.2939024671248753,
            (5.2366469441040480e-4, 2.9571442488783428e-9, 2.5943561344970496e-4),
            (0.9937220053052456, 0.9957369404882336, 0.9999999999999999),
        ),
        (
            12.1,
            3178.8279164368190,
            26.054036102571473,
            26.339961302836130,
            (1.3500690979615718e-4, 3.5651876911060294e-8, 4.5059096405196470e-5),
            (0.9971145539598428, 0.9980432805630943, 0.9999999999999999),
        ),
    ],
)
def test_the_cross_sections_are_raddose_3ds(  # type: ignore[no-untyped-def]
    material, energy, bethe, total, outer, carbon_k, zinc
) -> None:
    """Bethe's inner shells and the GOS model's subshells, as the Java's."""
    sigma = gos.bethe_shells(material, np.array([energy]))[0]
    assert 1 / sigma.sum() == pytest.approx(bethe, rel=1e-13)
    cumulative = np.cumsum(sigma.reshape(-1)) / sigma.sum()
    zn = _row(material, 30)
    np.testing.assert_allclose(cumulative[zn * 9 + 4 : zn * 9 + 7], zinc, rtol=1e-13)

    shells = inelastic_shells(material, energy)
    assert 1 / shells[..., 3].sum() == pytest.approx(total, rel=1e-13)
    valence = [atom.valence[1] for atom in material.atoms]
    outer_sum = sum(shells[i, v, 3] for i, v in enumerate(valence))
    assert 1 / outer_sum == pytest.approx(outer, rel=1e-13)
    np.testing.assert_allclose(shells[_row(material, 6), 0, :3], carbon_k, rtol=1e-12)


def test_below_every_inner_shell_bethes_chances_are_nan(material) -> None:  # type: ignore[no-untyped-def]
    """At 0.06 keV no inner shell is open: zero cross-section, chances 0/0."""
    medium = gos_medium(material, 0.07)
    free_path, cumulative = medium.inner_shells(np.array([0.06]))
    assert free_path[0] == 0.0
    assert np.isnan(cumulative).all()


def test_the_table_is_exact_at_its_grid_points(material) -> None:  # type: ignore[no-untyped-def]
    table = gos.shell_table(material, 0.25, 0.6)
    for energy in table.energies[::97]:
        np.testing.assert_array_equal(
            table.at(np.array([energy]))[0], inelastic_shells(material, float(energy))
        )


def test_the_tables_single_reads_are_its_whole_ones(material) -> None:  # type: ignore[no-untyped-def]
    """``rows`` and ``column`` read what ``at`` gives, to the last bit."""
    table = gos.shell_table(material, 0.25, 0.6)
    rng = np.random.default_rng(2)
    energy = rng.uniform(0.2, 0.7, 500)
    whole = table.at(energy)
    elements = len(material.atoms)
    element = rng.integers(0, elements, 500)
    row = rng.integers(0, 10, 500)
    np.testing.assert_array_equal(
        table.rows(energy, element, row), whole[np.arange(500), element, row]
    )
    inner = np.array([atom.valence[1] for atom in material.atoms])
    np.testing.assert_array_equal(
        table.column(energy, inner, 3), whole[:, np.arange(elements), inner, 3]
    )


def test_distant_losses_follow_their_triangle() -> None:
    """``getEnergyLossDistant``: from Uk to W_dis, falling linearly to zero."""
    rng = np.random.default_rng(1)
    uk, w_dis = np.full(100_000, 100.0), np.full(100_000, 400.0)
    w = gos.energy_loss_distant(rng, w_dis, uk)
    assert w.min() >= 100.0 and w.max() <= 400.0
    assert w.mean() == pytest.approx(100.0 + 300.0 / 3, rel=0.01)


def test_close_fractions_lie_between_the_cutoff_and_a_half() -> None:
    """``samplek``: k from Qk / (E + Qk) to 1/2; with Qk zero, 0 every time."""
    rng = np.random.default_rng(2)
    energy = np.full(20_000, 5.0)
    qk = np.full(20_000, 284.0)
    k = gos.sample_k(rng, energy, qk)
    assert k.min() >= 284.0 / (5000.0 + 284.0) and k.max() <= 0.5
    assert (gos.sample_k(rng, np.full(5, 5.0), np.zeros(5)) == 0).all()


class _Collect:
    """A sink that only adds up."""

    def __init__(self) -> None:
        self.resolved_total = 0.0
        self.dose_total = 0.0
        self.ions = 0

    def dose(self, points, owner, energy) -> None:  # type: ignore[no-untyped-def]
        self.dose_total += float(np.sum(energy))

    def resolved(self, points, owner, energy, time) -> None:  # type: ignore[no-untyped-def]
        self.resolved_total += float(np.sum(energy))

    def ionise(self, points, owner, element, count, time) -> None:  # type: ignore[no-untyped-def]
        self.ions += int(np.sum(count))


def test_a_cascade_conserves_a_vacancys_energy(material) -> None:  # type: ignore[no-untyped-def]
    """A sulfur K vacancy: what it leaves, its Auger electrons and its
    fluorescence photons come to the 2.472 keV it holds, to rounding. Each
    vacancy a transition leaves deposited its own binding energy as well
    until RADDOSE-3D #54."""
    medium = gos_medium(material, 3.0)
    sink = _Collect()
    cascade = Cascade(np.random.default_rng(3), medium, sink)
    photons: list[float] = []
    transition = cascade._transition

    def counting(table, z, shell):  # type: ignore[no-untyped-def]
        index = transition(table, z, shell)
        if table is cascade.fluorescence:
            photons.extend(table.energy_kev[z, shell, index].tolist())
        return index

    cascade._transition = counting  # type: ignore[method-assign]
    n = 4000
    s = _row(material, 16)
    born = _Born()
    cascade.relax(
        np.zeros((n, 3)),
        np.zeros(n, dtype=np.int64),
        np.full(n, s),
        np.zeros(n, dtype=np.int64),
        born,
    )
    freed = float(np.sum(np.concatenate(born.energies)))
    per_vacancy = (sink.resolved_total + freed + sum(photons)) / n
    assert per_vacancy == pytest.approx(2.472, rel=1e-12)
    assert photons and sink.ions > 0


def test_a_valence_collision_has_its_subshells_binding_energy() -> None:
    """Phosphorus's and selenium's valence subshells have no photoelectric
    edge. A collision with one took 0, recoiled by nothing, and deflected its
    electron by an undefined angle; it now takes the subshell's binding
    energy, as the model's cross-sections do, and the deflection is defined."""
    experiment = Experiment.from_yaml(CASE)
    data = experiment.model_dump()
    data["crystal"]["composition"]["protein_heavy_atoms"]["Se"] = 4.0
    material = monte_carlo_setup(Experiment.model_validate(data)).material
    medium = gos_medium(material, 12.1)
    for z in (15, 34, 16, 6):
        row = _row(material, z)
        valence = int(medium.inner[row])
        uk = medium.valence_binding[row] * 1000
        assert uk == material.atoms[row].subshell_binding(valence) * 1000 > 0
        wk = medium.wk[row, valence]
        energy = np.full(1000, 12.1)
        q = gos.recoil_distant(
            np.random.default_rng(z), energy, np.full(1000, wk), np.full(1000, uk)
        )
        theta = gos.primary_angle_distant(energy, q, np.full(1000, wk))
        assert np.isfinite(theta).all(), z


def _atoms(material: ElectronMaterial, zs: list[int]) -> list[_Atom]:
    """Material's first atom made each element of ``zs`` in turn."""
    table = _shell_table()
    return [
        dataclasses.replace(material.atoms[0], z=z, symbol=str(z), shells=table[z])
        for z in zs
    ]


def test_every_element_has_as_many_electrons_as_its_atomic_number(
    material: ElectronMaterial,
) -> None:
    """Arsenic and selenium had two more: their 4s counted in M5 and again as
    valence."""
    for atom in _atoms(material, sorted(_shell_table())):
        valence, inner = atom.valence
        assert valence + sum(_SUBSHELL_ELECTRONS[:inner]) == atom.z


def test_heavy_atoms_have_inner_subshells_and_a_valence_subshell(
    material: ElectronMaterial,
) -> None:
    """From bromine on every electron was valence, bound by the K edge."""
    heavy = [z for z in sorted(_shell_table()) if z >= 35]
    for atom in _atoms(material, heavy):
        assert atom.valence == (atom.z - 28, 9)
        binding = [atom.subshell_binding(i) for i in range(10)]
        assert all(a > b for a, b in itertools.pairwise(binding)), atom.z
    # Iodine's valence electrons, 4s to 5p, are bound by 193 to 10 eV.
    (iodine,) = _atoms(material, [53])
    assert 0.010 < iodine.subshell_binding(9) < 0.193


def test_a_heavy_atom_leaves_the_sternheimer_factor_to_the_composition(
    material: ElectronMaterial,
) -> None:
    """1 M iodide, its 53 electrons bound by its K edge, took the factor from
    1.65 to 1.2, and iodine had no collisions below 33 keV."""
    data = Experiment.from_yaml(CASE).model_dump()
    data["crystal"]["composition"]["solvent_heavy_conc"]["I"] = 1000.0
    iodide = monte_carlo_setup(Experiment.model_validate(data)).material
    assert iodide.sternheimer_a == pytest.approx(material.sternheimer_a, abs=0.02)
    shells = inelastic_shells(iodide, 10.0)
    assert shells[_row(iodide, 53), 9, 3] > 0
