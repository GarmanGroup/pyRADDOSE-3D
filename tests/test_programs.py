"""The run scripts in programs/, run as a user would run them."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from pyraddose_3d.models.experiment import Experiment

DOSE = Path(__file__).parent.parent / "programs" / "dose"
INPUTS = Path(__file__).parent.parent / "docs" / "examples" / "inputs"


def test_the_two_example_inputs_describe_the_same_experiment() -> None:
    """The YAML is the .txt, setting for setting, so either can be the example."""
    from_yaml = Experiment.from_file(INPUTS / "insulin.yaml")
    from_txt = Experiment.from_file(INPUTS / "insulin.txt")
    assert from_yaml == from_txt


@pytest.mark.parametrize("config", ["insulin.yaml", "insulin.txt"])
def test_run_dose_writes_what_raddose_writes(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Edit the parameters block, run, and get RADDOSE-3D's files and output."""
    script = runpy.run_path(str(DOSE / "run_dose.py"))
    script["main"].__globals__.update(
        CONFIG_PATH=str(INPUTS / config), OUTPUT_PREFIX=str(tmp_path / "output-")
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
    from_yaml = Experiment.from_file(INPUTS / "microed.yaml")
    from_txt = Experiment.from_file(INPUTS / "microed.txt")
    assert from_yaml == from_txt


def test_run_dose_runs_microed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A MicroED experiment prints RADDOSE-3D's report and writes its two files."""
    script = runpy.run_path(str(DOSE / "run_dose.py"))
    script["main"].__globals__.update(
        CONFIG_PATH=str(INPUTS / "microed.yaml"),
        OUTPUT_PREFIX=str(tmp_path / "output-"),
    )
    script["main"]()

    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["outputMicroED.CSV", "outputMicroED_slices.csv"]
    printed = capsys.readouterr().out
    assert "The Dose in the exposed area by stopping power" in printed
    assert "The optimal accelerating voltage is" in printed


def test_run_dose_runs_the_input_named_on_the_command_line(tmp_path: Path) -> None:
    """``python run_dose.py MyInput.txt``, as RADDOSE-3D is given its input."""
    result = subprocess.run(
        [sys.executable, str(DOSE / "run_dose.py"), str(INPUTS / "microed.txt")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"Loaded {INPUTS / 'microed.txt'}" in result.stdout
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["outputMicroED.CSV", "outputMicroED_slices.csv"]


def test_run_dose_runs_the_example_unless_given_another() -> None:
    script = runpy.run_path(str(DOSE / "run_dose.py"))
    assert script["CONFIG_PATH"] == INPUTS.resolve() / "insulin.txt"
    assert script["CONFIG_PATH"].is_file()


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


def test_run_dose_runs_the_monte_carlo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Subprogram MONTECARLO: what RADDOSE-3D prints, and its three files."""
    text = (INPUTS / "insulin.txt").read_text()
    text = text.replace(
        "Crystal\n", "Crystal\nSubprogram MonteCarlo\nSimPhotons 2000\nRuns 2\n", 1
    )
    config = tmp_path / "input.txt"
    config.write_text(text)
    script = runpy.run_path(str(DOSE / "run_dose.py"))
    script["main"].__globals__.update(
        CONFIG_PATH=str(config), OUTPUT_PREFIX=str(tmp_path / "output-")
    )
    script["main"]()

    printed = capsys.readouterr().out
    assert printed.count("Monte Carlo average dose whole crystal (ADWC): ") == 2
    files = {p.name: p.read_text() for p in tmp_path.glob("output*.CSV")}
    assert sorted(files) == ["outputIonsPerAtom.CSV", "outputMC.CSV", "outputXFEL.CSV"]
    rows = files["outputMC.CSV"].splitlines()
    assert rows[0] == "Run Number, RD3D-ADWC,RD3D-ADER,MC-ADWC,MC-ADER,numEl"
    assert [r.split(",")[0] for r in rows[1:]] == [" 1", " 2"]
    assert len(files["outputXFEL.CSV"].splitlines()[1].split(",")) == 11


#: A small still for the GOS and XFEL subprograms, whose electrons are slow.
_STILL = """Crystal
Type Cuboid
Dimensions 10 10 10
PixelsPerMicron 0.5
AbsCoefCalc RD3D
UnitCell 78.02 78.02 78.02
NumMonomers 24
NumResidues 51
ProteinHeavyAtoms Zn 0.333 S 6
SolventHeavyConc P 425
SolventFraction 0.64
Subprogram {subprogram}
SimPhotons 3000
Runs 2

Beam
Type Gaussian
{intensity}
FWHM 20 20
Energy 12.1
Collimation Rectangular 40 40

Wedge 0 {end}
ExposureTime 10
"""


def _run_dose(text: str, tmp_path: Path) -> None:
    config = tmp_path / "input.txt"
    config.write_text(text)
    script = runpy.run_path(str(DOSE / "run_dose.py"))
    script["main"].__globals__.update(
        CONFIG_PATH=str(config), OUTPUT_PREFIX=str(tmp_path / "output-")
    )
    script["main"]()


@pytest.mark.parametrize(
    ("subprogram", "intensity", "printed"),
    [
        ("GOS", "Flux 2e12", "RADDOSE-GOS average dose whole crystal (ADWC): "),
        (
            "XFEL",
            "PulseEnergy 1.4",
            "RADDOSE-XFEL average dose whole crystal (ADWC) (MGy): ",
        ),
    ],
)
def test_run_dose_runs_gos_and_xfel(
    subprogram: str,
    intensity: str,
    printed: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What RADDOSE-3D prints, and its two files: not ``outputMC.CSV``."""
    _run_dose(
        _STILL.format(subprogram=subprogram, intensity=intensity, end=0), tmp_path
    )
    assert capsys.readouterr().out.count(printed) == 2
    files = sorted(p.name for p in tmp_path.glob("output*.CSV"))
    assert files == ["outputIonsPerAtom.CSV", "outputXFEL.CSV"]
