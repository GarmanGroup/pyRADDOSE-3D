"""The output writers' behaviour away from the conformance corpus.

``tests/conformance/test_output_files.py`` compares every file with the Java's,
case by case. These cover what the corpus cannot reach: a decay model the
port cannot run yet, histogram edges no case lands on, and writing without
per-image records or from torch tensors.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pytest

from pyraddose_3d import outputs
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import Simulation, simulate

CASES = Path(__file__).parent / "conformance" / "cases"

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@pytest.fixture(scope="module")
def experiment() -> Experiment:
    return Experiment.from_yaml(CASES / "000-baseline" / "config.yaml")


@pytest.fixture(scope="module")
def simulation(experiment: Experiment) -> Simulation:
    return simulate(experiment)


def test_histogram_bins_as_the_java_bins_them() -> None:
    """Zero is left out; 0.1 opens bin 2 and 30 opens the last.

    The step is (30 - 0.1) / 9 = 3.3222..., so 3.42 is still in bin 2: the
    Java truncates (3.42 - 0.1) / step = 0.9993 to 0.
    """
    doses = np.array([0.0, 0.05, 0.1, 3.42, 29.99, 30.0, 31.0])
    counts = np.array(outputs.dose_histogram(doses)) * 6
    np.testing.assert_allclose(counts, [1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2])


def test_an_empty_histogram_prints_as_java_prints_zero_over_zero() -> None:
    """No dosed voxels: the Java divides 0 by 0, and prints NaN."""
    lines = outputs._histogram_lines(np.zeros(4)).splitlines()
    assert lines[1] == "Bin  1,  0.0 to  0.1 MGy:  NaN % "
    assert lines[-1] == "Bin 11, 30.0 MGy upwards:  NaN %"


def test_leal_description_matches_java(experiment: Experiment) -> None:
    """The Leal model cannot run yet, but its description is known."""
    crystal = experiment.crystal.model_copy(
        update={"decay_model": "leal", "decay_parameters": (0.3, 10.0, 0.03)}
    )
    reference = (CASES / "041-ddm-leal" / "expected" / "Summary.txt").read_text()
    assert (
        outputs.decay_description(crystal).splitlines() == (reference.splitlines()[1:6])
    )


def test_per_voxel_values_at_or_below_float_min_print_as_zero() -> None:
    """The DoseState writers' bare ``0``, including for single underflow."""
    assert outputs._clamped(0.0) == "0"
    assert outputs._clamped(-1.0) == "0"
    assert outputs._clamped(1e-46) == "0"
    assert outputs._clamped(1.0) == "1.0"


def test_without_images_the_per_image_files_are_skipped(
    simulation: Simulation, tmp_path: Path
) -> None:
    """Without per-image records the four whole-run files are still written."""
    written = outputs.write_outputs(simulation, prefix=str(tmp_path / "run-"))
    assert [p.name for p in written] == [
        "run-Summary.csv",
        "run-Summary.txt",
        "run-DoseState.csv",
        "run-DoseState.R",
    ]
    assert written[1].read_text("utf-8") == outputs.summary_text(simulation)
    with pytest.raises(ValueError, match="record_images=True"):
        outputs.dwds_csv(simulation)


def test_the_r_script_is_stamped_with_the_time_given(simulation: Simulation) -> None:
    script = outputs.dose_state_r(simulation, datetime.datetime(2026, 1, 2, 3, 4, 5))
    assert "# Code generated 2026-01-02 03:04:05\n" in script


def test_writes_from_a_differentiable_run(experiment: Experiment) -> None:
    """A run carrying gradients writes the files a NumPy run writes.

    Not digit for digit: torch's cos and sin can differ from NumPy's in the
    last bit, and on a voxel lying on a crystal face that flips its path
    length -- the knife edge the corpus's tolerances are set by, which a
    still has too. So the text must match with the numbers masked, and the
    numbers agree within the corpus's loosest tier A tolerance.
    """
    torch = pytest.importorskip("torch")
    flux = torch.tensor(experiment.beam.flux, dtype=torch.float64, requires_grad=True)
    tensors = simulate(experiment, {"beam.flux": flux}, record_images=True)
    arrays = simulate(experiment, record_images=True)

    for writer in (outputs.summary_text, outputs.summary_csv, outputs.dwds_csv):
        masked = [_NUMBER.sub("#", writer(run)) for run in (tensors, arrays)]
        assert masked[0] == masked[1]
    for got, want in zip(tensors.summaries, arrays.summaries, strict=True):
        got_values, want_values = got.as_dict(), want.as_dict()
        for name, value in want_values.items():
            assert got_values[name] == pytest.approx(value, rel=5e-3), name
