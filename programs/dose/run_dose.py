"""Program for calculating the dose a crystal absorbs during data collection."""

import time

from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import summary_text, write_outputs
from pyraddose_3d.solve import simulate

#######################################
### Editable parameters for program ###
#######################################

# The experiment: a YAML configuration like dose_example_config.yaml, or a
# RADDOSE-3D input file like dose_example_input.txt, which is read unchanged.
# Anything not ending in .yaml or .yml is read as a RADDOSE-3D input file.
CONFIG_PATH = "/path/to/dose_example_config.yaml"

# Prepended to every output file name, as RADDOSE-3D's --prefix does, so the
# default writes output-Summary.txt and so on to the current directory. It
# may include a directory, which must already exist.
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


if __name__ == "__main__":
    main()
