"""An override must mean exactly what editing the configuration means.

An override is how an optimiser changes a setting: ``solve`` reads the
override wherever it would read the configured value. Miss one of those
places and the setting is only half changed -- and nothing else notices,
because a finite difference taken through the override is consistent with a
gradient taken through it. That happened: overriding ``beam.energy`` changed
the photon energy but left the crystal's absorption at the configured
energy, so every energy gradient described a crystal that did not exist.

So each tunable setting is checked against the plain configuration with the
same value written in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import TUNABLE, apply_overrides, simulate, solve

CASE = Path(__file__).parent / "conformance" / "cases" / "000-baseline" / "config.yaml"

#: A value for each tunable setting, away from the baseline's own.
VALUES = {
    "beam.flux": 1.5e12,
    "beam.energy": 9.0,
    "beam.fwhm": (18.0, 23.0),
    "beam.collimation.x": 35.0,
    "beam.collimation.y": 30.0,
    "wedges.0.start": 5.0,
    "wedges.0.exposure_time": 12.0,
    "wedges.0.rot_axis_beam_offset": 1.5,
    "wedges.0.start_offset": (1.0, 2.0, 0.0),
    "wedges.0.translate_per_degree": (0.05, 0.0, 0.02),
}


@pytest.fixture(scope="module")
def experiment() -> Experiment:
    return Experiment.from_yaml(CASE)


def test_every_tunable_setting_has_a_value_here() -> None:
    assert {path.replace(".0.", ".*.") for path in VALUES} == TUNABLE


@pytest.mark.parametrize("path", sorted(VALUES))
def test_an_override_is_the_same_as_editing_the_configuration(
    experiment: Experiment, path: str
) -> None:
    edited = apply_overrides(experiment, {path: VALUES[path]})
    overridden = solve(experiment, {path: VALUES[path]})
    assert [s.as_dict() for s in overridden] == [s.as_dict() for s in solve(edited)]
    # And the edit must actually change something, or the comparison is empty.
    assert [s.as_dict() for s in overridden] != [s.as_dict() for s in solve(experiment)]


def test_apply_overrides_writes_the_values_in(experiment: Experiment) -> None:
    edited = apply_overrides(experiment, VALUES)
    assert edited.beam.flux == 1.5e12
    assert edited.beam.energy == 9.0
    assert edited.beam.fwhm == (18.0, 23.0)
    assert edited.beam.collimation is not None
    assert (edited.beam.collimation.x, edited.beam.collimation.y) == (35.0, 30.0)
    wedge = edited.wedges[0]
    assert wedge.start == 5.0
    assert wedge.exposure_time == 12.0
    assert wedge.rot_axis_beam_offset == 1.5
    assert wedge.start_offset == (1.0, 2.0, 0.0)
    assert wedge.translate_per_degree == (0.05, 0.0, 0.02)
    # Everything else is untouched.
    assert edited.crystal == experiment.crystal
    assert apply_overrides(experiment, {}) is experiment


def test_tensors_are_written_in_as_plain_numbers(experiment: Experiment) -> None:
    torch = pytest.importorskip("torch")
    energy = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
    fwhm = torch.tensor([18.0, 23.0], dtype=torch.float64, requires_grad=True)
    edited = apply_overrides(experiment, {"beam.energy": energy, "beam.fwhm": fwhm})
    assert type(edited.beam.energy) is float
    assert edited.beam.fwhm == (18.0, 23.0)


def test_the_simulation_records_the_experiment_as_run(experiment: Experiment) -> None:
    simulation = simulate(experiment, {"beam.energy": 9.0})
    assert simulation.experiment.beam.energy == 9.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"wedges.1.start": 5.0}, "1 wedge"),
        ({"beam.collimation.x": 30.0}, "no collimation"),
    ],
)
def test_an_override_that_would_be_ignored_is_refused(
    experiment: Experiment, overrides: dict[str, float], message: str
) -> None:
    """Before, simulate never read these, and so ignored them silently."""
    uncollimated = experiment.model_copy(
        update={"beam": experiment.beam.model_copy(update={"collimation": None})}
    )
    with pytest.raises(KeyError, match=message):
        solve(uncollimated, overrides)


def test_a_width_the_beam_does_not_have_is_refused() -> None:
    measured = Experiment.from_yaml(
        Path(__file__).parent
        / "conformance"
        / "cases"
        / "080-beam-experimental"
        / "config.yaml"
    )
    with pytest.raises(KeyError, match="this experimental beam has no fwhm"):
        solve(measured, {"beam.fwhm": (10.0, 10.0)})


def test_an_invalid_value_is_refused(experiment: Experiment) -> None:
    with pytest.raises(ValidationError):
        solve(experiment, {"beam.flux": -1.0})
