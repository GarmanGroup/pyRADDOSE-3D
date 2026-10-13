"""The run scripts in programs/, run as a user would run them."""

from __future__ import annotations

import runpy
import warnings
from contextlib import nullcontext
from pathlib import Path

import pytest

from pyraddose_3d.models.experiment import Experiment

DOSE = Path(__file__).parent.parent / "programs" / "dose"


def test_the_two_example_inputs_describe_the_same_experiment() -> None:
    """The YAML is the .txt, setting for setting, so either can be the example."""
    from_yaml = Experiment.from_file(DOSE / "dose_example_config.yaml")
    from_txt = Experiment.from_file(DOSE / "dose_example_input.txt")
    assert from_yaml == from_txt


@pytest.mark.parametrize(
    "config", ["dose_example_config.yaml", "dose_example_input.txt"]
)
def test_run_dose_writes_what_raddose_writes(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Edit the parameters block, run, and get RADDOSE-3D's files and output."""
    script = runpy.run_path(str(DOSE / "run_dose.py"))
    script["main"].__globals__.update(
        CONFIG_PATH=str(DOSE / config), OUTPUT_PREFIX=str(tmp_path / "output-")
    )
    script["main"]()

    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == sorted(
        f"output-{name}"
        for name in (
            "Summary.csv", "Summary.txt", "DoseState.csv", "DoseState.R",
            "RDE.csv", "DWDs.csv", "VoxelDose.csv", "VoxelFluences.csv",
        )
    )  # fmt: skip
    summary = (tmp_path / "output-Summary.txt").read_text("utf-8")
    assert summary in capsys.readouterr().out
    assert summary.count("Wedge ") == 2


def test_the_two_microed_examples_describe_the_same_experiment() -> None:
    from_yaml = Experiment.from_file(DOSE / "microed_example_config.yaml")
    from_txt = Experiment.from_file(DOSE / "microed_example_input.txt")
    assert from_yaml == from_txt


def test_run_dose_runs_microed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A MicroED experiment prints RADDOSE-3D's report and writes its two files."""
    script = runpy.run_path(str(DOSE / "run_dose.py"))
    script["main"].__globals__.update(
        CONFIG_PATH=str(DOSE / "microed_example_config.yaml"),
        OUTPUT_PREFIX=str(tmp_path / "output-"),
    )
    script["main"]()

    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["outputMicroED.CSV", "outputMicroED_slices.csv"]
    printed = capsys.readouterr().out
    assert "The Dose in the exposed area by stopping power" in printed
    assert "The optimal accelerating voltage is" in printed


OPTIMISE = Path(__file__).parent.parent / "programs" / "optimise"


def test_run_optimise_energy_writes_a_runnable_strategy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The optimised configuration is an ordinary experiment at a new energy."""
    pytest.importorskip("torch")
    script = runpy.run_path(str(OPTIMISE / "run_optimise_energy.py"))
    optimised = tmp_path / "optimised.yaml"
    script["main"].__globals__.update(
        CONFIG_PATH=str(OPTIMISE / "optimise_energy_example_config.yaml"),
        OPTIMISED_CONFIG_PATH=str(optimised),
        ENERGY_RANGE=(10.0, 14.0),
    )
    script["main"]()

    result = Experiment.from_file(optimised)
    original = Experiment.from_file(OPTIMISE / "optimise_energy_example_config.yaml")
    assert 11.0 < result.beam.energy < 12.5
    assert result.crystal == original.crystal
    assert result.wedges == original.wedges
    assert "Best energy" in capsys.readouterr().out


@pytest.mark.parametrize("subprogram", ["MonteCarlo", "GOS", "XFEL"])
def test_run_dose_refuses_the_subprograms_not_yet_ported(
    subprogram: str, tmp_path: Path
) -> None:
    """Read, but refused by name before anything runs."""
    text = (DOSE / "dose_example_input.txt").read_text()
    text = text.replace("Crystal\n", f"Crystal\nSubprogram {subprogram}\n", 1)
    if subprogram == "XFEL":
        text = text.replace("Beam\n", "Beam\nPulseEnergy 1.4\n", 1)
    config = tmp_path / "input.txt"
    config.write_text(text)
    with (
        pytest.warns(match="Flux is ignored") if subprogram == "XFEL" else nullcontext()
    ):
        assert Experiment.from_file(config).crystal.subprogram == subprogram.lower()

    script = runpy.run_path(str(DOSE / "run_dose.py"))
    script["main"].__globals__.update(
        CONFIG_PATH=str(config), OUTPUT_PREFIX=str(tmp_path / "output-")
    )
    with (
        pytest.raises(NotImplementedError, match="Milestone 7"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        script["main"]()
    assert not list(tmp_path.glob("output-*"))
