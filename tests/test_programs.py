"""The run scripts in programs/, run as a user would run them."""

from __future__ import annotations

import runpy
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
