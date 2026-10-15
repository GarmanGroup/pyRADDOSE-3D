"""Program for calculating the dose a crystal absorbs during data collection."""

import time
from pathlib import Path

from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import (
    microed_report,
    monte_carlo_report,
    summary_text,
    write_microed_outputs,
    write_monte_carlo_outputs,
    write_outputs,
)
from pyraddose_3d.solve import microed, monte_carlo, simulate

#######################################
### Editable parameters for program ###
#######################################

# The experiment: a YAML configuration like dose_example_config.yaml, or a
# RADDOSE-3D input file like dose_example_input.txt, which is read unchanged.
# Anything not ending in .yaml or .yml is read as a RADDOSE-3D input file.
# A MicroED experiment, like microed_example_config.yaml, runs that engine,
# and one with subprogram montecarlo (Subprogram MONTECARLO) the Monte Carlo.
CONFIG_PATH = "/path/to/dose_example_config.yaml"

# Prepended to every output file name, as RADDOSE-3D's --prefix does, so the
# default writes output-Summary.txt and so on to the current directory. It
# may include a directory, which must already exist. MicroED and the Monte
# Carlo write their files under fixed names, as RADDOSE-3D does, into that
# directory.
OUTPUT_PREFIX = "output-"

# Whether to write the per-image files -- VoxelDose.csv, VoxelFluences.csv,
# DWDs.csv and RDE.csv -- as RADDOSE-3D does by default. They need every
# voxel's dose and fluence at every image kept in memory, so turn this off
# for a very fine voxel grid if memory runs short.
PER_IMAGE_FILES = True

#######################################################
### Main function called to run the dose program ###
#######################################################


def main() -> None:
    """Main function for running the dose program."""
    experiment = Experiment.from_file(CONFIG_PATH)
    print(f"Loaded {CONFIG_PATH}")

    start_time = time.time()
    if experiment.crystal.subprogram == "microed":
        run_microed(experiment, start_time)
        return
    if experiment.crystal.subprogram in ("montecarlo", "gos"):
        run_monte_carlo(experiment, start_time)
        return
    simulation = simulate(experiment, record_images=PER_IMAGE_FILES)
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

    # What RADDOSE-3D prints to the console, and saves as Summary.txt.
    print(summary_text(simulation), end="")
    print(f"Dose calculation wall time: {elapsed_time}")

    if experiment.seed is None and simulation.seed is not None:
        # As RADDOSE-3D does: escape is Monte Carlo, and this repeats it.
        print(f"Escape modelling used seed {simulation.seed}; set seed to repeat it")

    for path in write_outputs(simulation, prefix=OUTPUT_PREFIX):
        print(f"Wrote {path}")


def run_microed(experiment: Experiment, start_time: float) -> None:
    """MicroED: print what RADDOSE-3D prints, and write its two files."""
    result = microed(experiment)
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(microed_report(result), end="")
    print(f"Dose calculation wall time: {elapsed_time}")
    for path in write_microed_outputs(result, Path(OUTPUT_PREFIX).parent):
        print(f"Wrote {path}")


def run_monte_carlo(experiment: Experiment, start_time: float) -> None:
    """The Monte Carlo: print each run's doses, and write its three files."""
    runs = monte_carlo(experiment)
    for run in runs:
        print(monte_carlo_report(run))
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"Monte Carlo wall time, {len(runs)} run(s): {elapsed_time}")
    for path in write_monte_carlo_outputs(runs, Path(OUTPUT_PREFIX).parent):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
