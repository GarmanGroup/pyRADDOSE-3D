"""Differentiating a whole experiment, which is why the port exists.

:func:`~pyraddose_3d.solve.solve` takes a validated configuration and returns
the sixteen reported numbers. Hand it a tensor for any tunable setting and
the whole calculation runs in torch, so a data-collection strategy can be
optimised by gradient descent instead of by scanning a grid of input files.

Three things are worth testing separately, because they fail in different
ways:

1. **The gradient exists.** Easy to break by accident: one NumPy call in the
   middle of the chain detaches everything after it, silently, and the result
   is still a perfectly good number.
2. **The gradient is right.** Checked against central finite differences,
   away from the points where the image count changes.
3. **The boundary is enforced.** Settings that cannot be optimised are
   rejected rather than quietly returning a wrong number, which is what
   autograd does for crystal geometry.

The NumPy path is unaffected by all of this and is covered by the conformance
corpus, which must produce bit-identical results with no overrides.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pyraddose_3d.backend.exposure import exposure_angles
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import FIXED, TUNABLE, solve

CASE = Path(__file__).parent / "conformance" / "cases" / "000-baseline" / "config.yaml"


@pytest.fixture(scope="module")
def experiment() -> Experiment:
    return Experiment.from_yaml(CASE)


def _leaf(torch: object, value: object) -> object:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _max_dose(experiment: Experiment, path: str, value: object) -> float:
    torch = pytest.importorskip("torch")
    tensor = torch.tensor(value, dtype=torch.float64)
    return float(solve(experiment, {path: tensor})[0].max_dose)


# --- the gradient exists ------------------------------------------------


def test_no_overrides_stays_on_numpy(experiment: Experiment) -> None:
    """The CCP4 install has no torch, so the default path must not need it.

    Also the reason the conformance corpus is still bit-identical: with no
    overrides nothing in the chain has changed.
    """
    torch = pytest.importorskip("torch")
    summary = solve(experiment)[0]

    assert not torch.is_tensor(summary.max_dose)
    assert isinstance(float(summary.max_dose), float)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("beam.flux", 2.0e12),
        ("beam.energy", 12.1),
        ("beam.fwhm", [20.0, 20.0]),
        ("beam.collimation.x", 40.0),
        ("beam.collimation.y", 40.0),
        ("wedges.0.start", 1.0),
        ("wedges.0.exposure_time", 10.0),
        ("wedges.0.rot_axis_beam_offset", 3.0),
        ("wedges.0.start_offset", [1.0, 0.0, 2.0]),
        ("wedges.0.translate_per_degree", [0.01, 0.0, 0.02]),
    ],
)
def test_every_tunable_setting_reaches_the_dose(
    experiment: Experiment, path: str, value: object
) -> None:
    """One NumPy call anywhere in the chain would break this silently."""
    torch = pytest.importorskip("torch")
    leaf = _leaf(torch, value)

    summary = solve(experiment, {path: leaf})[0]
    assert torch.is_tensor(summary.max_dose), f"{path} did not reach torch"

    summary.max_dose.backward()
    assert leaf.grad is not None, f"no gradient reached {path}"
    assert torch.all(torch.isfinite(leaf.grad)), f"non-finite gradient at {path}"


def test_the_tunable_list_is_exactly_what_is_tested() -> None:
    """A setting added to TUNABLE without a test here would not be checked."""
    tested = {
        "beam.flux",
        "beam.energy",
        "beam.fwhm",
        "beam.collimation.x",
        "beam.collimation.y",
        "wedges.*.start",
        "wedges.*.exposure_time",
        "wedges.*.rot_axis_beam_offset",
        "wedges.*.start_offset",
        "wedges.*.translate_per_degree",
    }
    assert set(TUNABLE) == tested


def test_every_summary_field_is_differentiable(experiment: Experiment) -> None:
    """Including the three that are quantised.

    ``dose_threshold`` is a dose some voxel actually has, and ``tad`` divides
    by a count of voxels. Both select rather than interpolate, so the
    gradient has to be routed through the selected element; returning the
    value the sort produced would detach it.
    """
    torch = pytest.importorskip("torch")
    flux = _leaf(torch, 2.0e12)
    summary = solve(experiment, {"beam.flux": flux})[0]

    for field in (
        "average_dwd",
        "last_dwd",
        "elastic_yield",
        "ad_wc",
        "max_dose",
        "dose_threshold",
        "tad",
        "wedge_absorbed_energy",
    ):
        flux.grad = None
        getattr(summary, field).backward(retain_graph=True)
        assert flux.grad is not None, f"{field} detached the gradient"
        assert torch.isfinite(flux.grad), f"{field} gave a non-finite gradient"


# --- the gradient is right ----------------------------------------------


@pytest.mark.parametrize(
    ("path", "value", "step"),
    [
        ("beam.flux", 2.0e12, 1.0e9),
        ("beam.energy", 12.1, 1.0e-4),
        ("wedges.0.exposure_time", 10.0, 1.0e-4),
        ("wedges.0.rot_axis_beam_offset", 3.0, 1.0e-4),
        # Away from zero, where the image count changes -- see below.
        ("wedges.0.start", 1.0, 1.0e-3),
    ],
)
def test_gradient_matches_a_finite_difference(
    experiment: Experiment, path: str, value: float, step: float
) -> None:
    torch = pytest.importorskip("torch")
    leaf = _leaf(torch, value)
    solve(experiment, {path: leaf})[0].max_dose.backward()

    difference = (
        _max_dose(experiment, path, value + step)
        - _max_dose(experiment, path, value - step)
    ) / (2 * step)

    assert float(leaf.grad) == pytest.approx(difference, rel=1e-6)


def test_a_symmetric_offset_has_zero_gradient(experiment: Experiment) -> None:
    """Not a broken gradient: the beam is symmetric about the rotation axis.

    Worth pinning, because "the gradient is zero" and "the gradient never
    arrived" look identical from the outside and this is the one place a
    zero is the right answer.
    """
    torch = pytest.importorskip("torch")
    at_zero = _leaf(torch, 0.0)
    solve(experiment, {"wedges.0.rot_axis_beam_offset": at_zero})[0].max_dose.backward()
    assert float(at_zero.grad) == 0.0

    off_axis = _leaf(torch, 3.0)
    solve(experiment, {"wedges.0.rot_axis_beam_offset": off_axis})[
        0
    ].max_dose.backward()
    assert float(off_axis.grad) != 0.0


def test_the_image_count_is_a_step_the_gradient_cannot_see(
    experiment: Experiment,
) -> None:
    """Measures the one place autograd and a finite difference disagree.

    The wedge start enters twice: it shifts every angle, which is smooth,
    and it enters the integer image count, which is not. At a start of zero
    the count falls from 46 to 45 as soon as the angle moves, so a finite
    difference across that point measures the step rather than the slope.

    Autograd reports the slope at fixed count, which is the honest answer
    and the useful one. This is recorded rather than asserted away, because
    an optimiser stepping over such a boundary will see its objective jump
    and someone will need to know why.
    """
    torch = pytest.importorskip("torch")
    resolution = math.radians(2.0)

    def images(start_degrees: float) -> int:
        return len(
            exposure_angles(
                start_degrees * math.pi / 180, 90 * math.pi / 180, resolution
            )
        )

    assert images(-0.001) == 46
    assert images(0.001) == 45, (
        "the count must change here for this test to mean anything"
    )

    at_zero = _leaf(torch, 0.0)
    solve(experiment, {"wedges.0.start": at_zero})[0].max_dose.backward()
    across = (
        _max_dose(experiment, "wedges.0.start", 0.001)
        - _max_dose(experiment, "wedges.0.start", -0.001)
    ) / 0.002

    assert float(at_zero.grad) != pytest.approx(across, rel=0.5)

    # Away from the boundary the two agree closely, which is what shows the
    # disagreement above is the step and not a broken gradient.
    assert images(0.999) == images(1.001) == 45


# --- the boundary is enforced -------------------------------------------


@pytest.mark.parametrize("path", sorted(FIXED))
def test_settings_that_cannot_be_optimised_are_refused(
    experiment: Experiment, path: str
) -> None:
    """Autograd returns a number for crystal geometry and it is wrong.

    Measured 20% out against a finite difference for a cuboid's thickness,
    because the occupancy test is a boolean parity count and autograd sees
    only the attenuation path. A confident wrong number is worse than an
    error, so these are refused with the reason.
    """
    torch = pytest.importorskip("torch")
    concrete = path.replace("wedges.*.", "wedges.0.")

    with pytest.raises(KeyError) as raised:
        solve(experiment, {concrete: _leaf(torch, 1.0)})

    assert "cannot be optimised" in str(raised.value)
    assert len(str(raised.value)) > 80, "the refusal must say why"


def test_an_unknown_setting_lists_what_is_tunable(experiment: Experiment) -> None:
    with pytest.raises(KeyError, match="not a tunable setting"):
        solve(experiment, {"beam.flx": 1.0})


def test_a_wedge_index_is_matched_against_the_pattern(
    experiment: Experiment,
) -> None:
    """``wedges.3.end`` must be refused for the same reason as ``wedges.0.end``."""
    with pytest.raises(KeyError, match="cannot be optimised"):
        solve(experiment, {"wedges.3.end": 90.0})


# --- it actually optimises ----------------------------------------------


def test_gradient_descent_reduces_the_peak_dose(experiment: Experiment) -> None:
    """The point of all of it: follow the gradient and the dose comes down.

    Widening the beam at fixed flux spreads the same photons over more
    crystal, so the peak dose falls. A handful of steps is enough to show
    the loop closes; this is not a claim about convergence.
    """
    torch = pytest.importorskip("torch")
    fwhm = _leaf(torch, [20.0, 20.0])
    optimiser = torch.optim.SGD([fwhm], lr=200.0)

    first = None
    for _ in range(5):
        optimiser.zero_grad()
        peak = solve(experiment, {"beam.fwhm": fwhm})[0].max_dose
        first = float(peak.detach()) if first is None else first
        peak.backward()
        optimiser.step()

    final = float(solve(experiment, {"beam.fwhm": fwhm.detach()})[0].max_dose)

    assert final < first, f"peak dose rose, {first} -> {final}"
    assert torch.all(fwhm.detach() > 20.0), "the beam should have widened"
