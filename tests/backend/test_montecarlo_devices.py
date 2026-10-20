"""The Monte Carlo on a torch device, held to itself on NumPy.

A run on a device draws torch's random numbers, so it is a different run
from NumPy's, and is compared with RADDOSE-3D's statistically as NumPy's is
(``PYRADDOSE_MC_DEVICE`` in ``tests/conformance/test_montecarlo_files.py``).
Here the device's arithmetic is held to NumPy's: the namespace function by
function, the physics kernel by kernel, and whole runs made with NumPy's
draws mirrored onto the device, which must follow NumPy's electron for
electron. They run on torch's CPU; ``PYRADDOSE_TEST_DEVICE=cuda`` adds a GPU.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys

import numpy as np
import pytest

from pyraddose_3d.backend._array import is_torch
from pyraddose_3d.backend.montecarlo import gos, material, ops, photons
from pyraddose_3d.backend.montecarlo.angles import deflection_angles
from pyraddose_3d.backend.montecarlo.geometry import (
    Placement,
    entry_depth,
    exit_distance,
    inside,
    voxel_indices,
)
from pyraddose_3d.backend.montecarlo.gos_transport import gos_medium
from pyraddose_3d.backend.montecarlo.simple import simulate_simple
from pyraddose_3d.backend.montecarlo.transport import new_directions
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import monte_carlo, monte_carlo_setup

torch = pytest.importorskip("torch", reason="torch is an optional extra")

CASES = "tests/conformance/cases"
DEVICES = ["cpu", *filter(None, [os.environ.get("PYRADDOSE_TEST_DEVICE")])]


def _setup(case: str, photons: int = 5000, runs: int = 2):  # type: ignore[no-untyped-def]
    setup = monte_carlo_setup(Experiment.from_yaml(f"{CASES}/{case}/config.yaml"))
    return dataclasses.replace(setup, photons=photons, runs=runs)


@pytest.fixture(scope="module")
def cylinder():  # type: ignore[no-untyped-def]
    """Case 152: a 124-triangle cylinder in a surrounding."""
    return _setup("152-montecarlo-cylinder-surrounding")


def _same(expected, got, rtol: float = 0.0, atol: float = 0.0) -> None:  # type: ignore[no-untyped-def]
    """NumPy's result: exactly, or to ``rtol`` -- and ``atol``, for components
    near zero -- where torch rounds a function or orders a sum differently."""
    got = ops.to_numpy(got)
    expected = np.asarray(expected)
    assert got.dtype == expected.dtype
    if rtol:
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_array_equal(got, expected)


# --- the namespace ------------------------------------------------------------

_A = np.random.default_rng(0).random((6, 4))
_B = np.random.default_rng(1).random(6)

#: (name, function of a namespace and two arrays): NumPy's result, exactly.
NAMESPACE = [
    ("where between floats", lambda xp, a, b: xp.where(b > 0.5, 0.0, 1.0)),
    (
        "where of ints and a float",
        lambda xp, a, b: xp.where(b > 0.5, xp.arange(6), 0.5),
    ),
    ("argmax of booleans", lambda xp, a, b: xp.argmax(a > 0.5, axis=1)),
    ("any along an axis", lambda xp, a, b: xp.any(a > 0.9, axis=1)),
    ("flatnonzero", lambda xp, a, b: xp.flatnonzero(b > 0.3)),
    (
        "max with an initial",
        lambda xp, a, b: xp.max(xp.zeros(0, dtype=xp.int64), initial=-1),
    ),
    ("min along an axis", lambda xp, a, b: xp.min(a, axis=1)),
    ("full of a flag", lambda xp, a, b: xp.full(3, True)),
    ("full of ints", lambda xp, a, b: xp.full(3, -1, dtype=xp.int64)),
    ("arange", lambda xp, a, b: xp.arange(5)),
    ("asarray of a list", lambda xp, a, b: xp.asarray([1.0, 2.5])),
    ("asarray to float", lambda xp, a, b: xp.asarray([1, 2], dtype=float)),
    ("astype", lambda xp, a, b: xp.astype(b * 10, xp.int64)),
    (
        "clip by an array",
        lambda xp, a, b: xp.clip(xp.asarray([[5, -1, 2]]), 0, np.array([3, 3, 1])),
    ),
    ("maximum with a number", lambda xp, a, b: xp.maximum(xp.asarray([-1, 2]), 0)),
    ("minimum with a number", lambda xp, a, b: xp.minimum(xp.asarray([-1, 9]), 3)),
    ("cross, unfused", lambda xp, a, b: xp.cross(a[:, None, :3], a[None, :2, 1:])),
    (
        "searchsorted, right",
        lambda xp, a, b: xp.searchsorted(xp.sort(b), a[:, 0], side="right"),
    ),
    (
        "unique's inverse",
        lambda xp, a, b: xp.unique(xp.asarray([3, 1, 3]), return_inverse=True)[1],
    ),
    ("isin a list", lambda xp, a, b: xp.isin(xp.asarray([1, 6, 8]), [6, 7, 8])),
    (
        "concatenate of mixed",
        lambda xp, a, b: xp.concatenate([xp.zeros(1, dtype=xp.int64), b]),
    ),
    ("stack", lambda xp, a, b: xp.stack([b, b], axis=1)),
    ("sort", lambda xp, a, b: xp.sort(a, axis=1)),
    ("trunc", lambda xp, a, b: xp.trunc(a * 7 - 3)),
]

#: As :data:`NAMESPACE`, but sums torch may add in another order, and a power
#: it may round differently -- on a GPU, if not on the CPU: to a few ulp.
NEAR = [
    ("sum, kept", lambda xp, a, b: xp.sum(a, axis=1, keepdims=True)),
    ("cumsum", lambda xp, a, b: xp.cumsum(a, axis=1)),
    ("accumulate", lambda xp, a, b: xp.add.accumulate(a, axis=0)),
    ("power of a number", lambda xp, a, b: xp.power(2.0, b)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("name", "function"), NAMESPACE, ids=[n for n, _ in NAMESPACE])
def test_the_namespace_gives_numpys_results(device: str, name: str, function) -> None:  # type: ignore[no-untyped-def]
    xp = ops.torch_namespace(device)
    _same(function(np, _A, _B), function(xp, xp.asarray(_A), xp.asarray(_B)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("name", "function"), NEAR, ids=[n for n, _ in NEAR])
def test_the_namespace_sums_as_numpy_does(device: str, name: str, function) -> None:  # type: ignore[no-untyped-def]
    xp = ops.torch_namespace(device)
    _same(function(np, _A, _B), function(xp, xp.asarray(_A), xp.asarray(_B)), 1e-15)


@pytest.mark.parametrize("device", DEVICES)
def test_add_at_adds_every_repeat(device: str) -> None:
    xp = ops.torch_namespace(device)
    expected, got = np.zeros((2, 3)), xp.zeros((2, 3))
    index = (np.array([0, 1, 0, 0]), np.array([2, 2, 2, 1]))
    np.add.at(expected, index, np.array([1, 2, 3, 4]))
    xp.add.at(got, tuple(xp.asarray(i) for i in index), xp.asarray([1, 2, 3, 4]))
    _same(expected, got)
    np.add.at(expected, index[0], 1)
    xp.add.at(got, xp.asarray(index[0]), 1)
    _same(expected, got)


def test_floats_stay_float64_on_a_device() -> None:
    """torch would make float32 of these, and of a Python float list."""
    xp = ops.torch_namespace("cpu")
    with ops.float64_by_default(xp):
        assert (xp.arange(3) / 2).dtype == torch.float64
        assert (xp.arange(3) * 0.5).dtype == torch.float64
        assert torch.is_inference_mode_enabled()
    assert torch.get_default_dtype() == torch.float32


def test_on_moves_tables_and_meshes_but_not_materials(cylinder) -> None:  # type: ignore[no-untyped-def]
    xp = ops.torch_namespace("cpu")
    medium = ops.on(xp, gos_medium(cylinder.material, 12.4))
    assert is_torch(medium.table.values)
    assert is_torch(medium.binding)
    assert medium.material is cylinder.material
    mesh = ops.on(xp, cylinder.mesh)
    assert is_torch(mesh.vertices)
    assert ops.on(xp, cylinder.mesh) is mesh
    assert ops.on(np, cylinder.mesh) is cylinder.mesh


# --- the physics, kernel by kernel --------------------------------------------


def _points(n: int, mesh, rng) -> np.ndarray:  # type: ignore[no-untyped-def]
    vertices = np.asarray(mesh.vertices)
    low, high = vertices.min(axis=0) * 1200, vertices.max(axis=0) * 1200
    return rng.uniform(low, high, (n, 3))


def _placement(n: int, rng) -> Placement:  # type: ignore[no-untyped-def]
    return Placement(rng.uniform(0, 2 * math.pi, n), rng.normal(0, 0.5, (n, 3)))


@pytest.mark.parametrize("device", DEVICES)
def test_the_geometry_is_numpys(device: str, cylinder) -> None:  # type: ignore[no-untyped-def]
    """Points at random, never on a face: the same answers, the same distances."""
    xp = ops.torch_namespace(device)
    rng = np.random.default_rng(3)
    n = 2000
    points, place = _points(n, cylinder.mesh, rng), _placement(n, rng)
    direction = new_directions(
        np.tile([0.0, 0.0, 1.0], (n, 1)), rng.uniform(0, 3, n), rng.uniform(0, 6, n)
    )
    on = Placement(xp.asarray(place.angle), xp.asarray(place.offset))
    p, d = xp.asarray(points), xp.asarray(direction)
    mesh = cylinder.mesh

    _same(inside(mesh, points, place), inside(mesh, p, on))
    _same(
        entry_depth(mesh, points[:, 0], points[:, 1], place),
        entry_depth(mesh, p[:, 0], p[:, 1], on),
        rtol=1e-12,
    )
    _same(
        exit_distance(mesh, points, direction, place),
        exit_distance(mesh, p, d, on),
        1e-12,
    )
    minimum = np.asarray(mesh.vertices).min(axis=0)
    _same(
        voxel_indices(points, place, minimum, 2.0), voxel_indices(p, on, minimum, 2.0)
    )
    _same(
        new_directions(direction, points[:, 0], points[:, 1]),
        new_directions(d, p[:, 0], p[:, 1]),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_the_electron_physics_is_numpys(device: str, cylinder) -> None:  # type: ignore[no-untyped-def]
    """To 1e-11: the stopping power's terms cancel, so an ulp of torch's log
    comes out at 3e-13."""
    xp = ops.torch_namespace(device)
    rng = np.random.default_rng(4)
    energy = np.concatenate([rng.uniform(0.05, 20, 500), rng.uniform(20, 300, 100)])
    e = xp.asarray(energy)
    m = cylinder.material

    _same(material.stopping_power(m, energy), material.stopping_power(m, e), 1e-11)
    sigma = material.elastic_cross_sections(m, energy)
    sigma_on = material.elastic_cross_sections(m, e)
    _same(sigma, sigma_on, 1e-11)
    _same(
        material.elastic_mean_free_path(m, sigma),
        material.elastic_mean_free_path(m, sigma_on),
        1e-11,
    )
    _same(
        material.elastic_element_cdf(m, sigma),
        material.elastic_element_cdf(m, sigma_on),
        1e-11,
    )
    _same(gos.bethe_shells(m, energy), gos.bethe_shells(m, e), 1e-11)

    z = np.array([atom.z for atom in m.atoms])[rng.integers(0, len(m.atoms), 600)]
    u = rng.random(600)
    elements = tuple(atom.z for atom in m.atoms)
    _same(
        deflection_angles(z, energy, u, elements),
        deflection_angles(xp.asarray(z), e, xp.asarray(u), elements),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_the_gos_table_and_shells_are_numpys(device: str, cylinder) -> None:  # type: ignore[no-untyped-def]
    xp = ops.torch_namespace(device)
    rng = np.random.default_rng(5)
    m = cylinder.material
    table = gos.shell_table(m, 0.05, 12.5)
    table_on = ops.on(xp, table)
    energy = rng.uniform(0.06, 12, 400)
    element = rng.integers(0, len(m.atoms), 400)
    row = rng.integers(0, 10, 400)
    inner = np.array([atom.valence[1] for atom in m.atoms])
    e, el, r = xp.asarray(energy), xp.asarray(element), xp.asarray(row)
    _same(table.rows(energy, element, row), table_on.rows(e, el, r), 1e-14)
    _same(
        table.column(energy, inner, 3), table_on.column(e, xp.asarray(inner), 3), 1e-14
    )

    elements = (1, 6, 16, 30, 53)
    stacked = photons.stacked_shells(elements)
    which = rng.integers(0, len(elements), 400)
    zs = np.array(elements)[which]
    _same(
        photons.shell_cdfs(zs, stacked.edges[which], stacked.ionisation[which], energy),
        photons.shell_cdfs(
            xp.asarray(zs),
            xp.asarray(stacked.edges[which]),
            xp.asarray(stacked.ionisation[which]),
            e,
        ),
        1e-15,
    )


# --- whole runs ---------------------------------------------------------------

#: Simple mode, a 124-triangle cylinder, GOS mode, and XFEL's Gaussian pulse.
#: None with a surrounding: one moves electrons onto the crystal's faces, where
#: whether a point is inside turns on its last bit, and torch's sines need
#: not round as NumPy's do on every machine. Every case follows on this one.
RUNS = [
    "140-montecarlo-baseline",
    "146-montecarlo-cylinder",
    "160-gos-baseline",
    "187-xfel-gaussian-pulse",
]


def _fields(run) -> dict[str, float]:  # type: ignore[no-untyped-def]
    values = {}
    for f in dataclasses.fields(run):
        value = getattr(run, f.name)
        if isinstance(value, dict):
            values.update({f"{f.name}[{k}]": v for k, v in value.items()})
        elif not isinstance(value, bool):
            values[f.name] = value
    return values


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", RUNS)
def test_a_run_with_numpys_draws_follows_numpys(device: str, case: str) -> None:
    """Electron for electron: the sums alone may differ, in their last bits."""
    setup = _setup(case)
    expected = simulate_simple(setup, np.random.default_rng(11))
    got = simulate_simple(setup, ops.MirroredRandom(np.random.default_rng(11), device))
    for a, b in zip(expected, got, strict=True):
        ours, theirs = _fields(a), _fields(b)
        assert ours.keys() == theirs.keys()
        for name, value in ours.items():
            assert theirs[name] == pytest.approx(value, rel=1e-12, nan_ok=True), name


@pytest.mark.parametrize("device", DEVICES)
def test_a_seed_repeats_on_a_device(device: str) -> None:
    setup = _setup("160-gos-baseline", photons=3000)
    first = simulate_simple(setup, ops.TorchRandom(5, device))
    second = simulate_simple(setup, ops.TorchRandom(5, device))
    for a, b in zip(first, second, strict=True):
        # The same electrons; a GPU sums their tallies in no fixed order.
        assert _fields(b) == pytest.approx(_fields(a), rel=1e-12, nan_ok=True)
    assert first[0].diagnostics["electrons"] > 0


def test_monte_carlo_runs_on_a_device() -> None:
    base = Experiment.from_yaml(f"{CASES}/180-xfel-baseline/config.yaml")
    simulation = base.crystal.simulation.model_copy(update={"photons": 3000, "runs": 1})
    experiment = base.model_copy(
        update={
            "seed": 2,
            "crystal": base.crystal.model_copy(update={"simulation": simulation}),
        }
    )
    (run,) = monte_carlo(experiment, device="cpu")
    assert run.xfel and math.isfinite(run.xfel_ader) and run.xfel_ader > 0


def test_a_device_without_torch_says_what_to_install(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    experiment = Experiment.from_yaml(f"{CASES}/140-montecarlo-baseline/config.yaml")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"pyRADDOSE-3D\[torch\]"):
        monte_carlo(experiment, device="cuda")
