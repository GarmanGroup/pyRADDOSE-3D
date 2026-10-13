# Dose program example script and configs

This directory contains an example script for calculating the dose a crystal absorbs during data collection, and two example inputs describing the same experiment.

## Files

- `run_dose.py` - The Python script for running a dose calculation. Edit the parameters at the top: the path to your input, where to write the outputs, and whether to write the per-image files.
- `dose_example_config.yaml` - An example YAML configuration, commented with each setting's units and its RADDOSE-3D keyword.
- `dose_example_input.txt` - The same experiment as a RADDOSE-3D input file. This is `MyInput.txt` from RADDOSE-3D, unmodified: existing input files are read as they are.
- `microed_example_config.yaml` - A MicroED experiment: a protein microcrystal in a 300 keV electron beam.
- `microed_example_input.txt` - The same, as RADDOSE-3D's own `ED_input_cell.txt`, unmodified.

## Running

```bash
python run_dose.py
```

The summary is printed, as RADDOSE-3D prints it, and the output files are written with the names and formats RADDOSE-3D uses, so scripts that read RADDOSE-3D's outputs read these unchanged:

| file | contents |
|---|---|
| `output-Summary.txt` | the printed summary |
| `output-Summary.csv` | the same per-wedge numbers as a table |
| `output-DoseState.csv` | `x,y,z,dose,fluence,elastic` for every voxel at the end |
| `output-DoseState.R` | an R script drawing dose contours (needs `rgl` and `misc3d`) |
| `output-DWDs.csv` | diffraction-weighted dose and resolution-dependent RDE per image |
| `output-RDE.csv` | relative diffraction efficiency per image |
| `output-VoxelDose.csv` | every voxel's dose in every image of the last wedge |
| `output-VoxelFluences.csv` | every voxel's fluence in every image of the last wedge |

The last four are written only with `PER_IMAGE_FILES = True`.

## MicroED

A crystal with `subprogram: microed` (`Subprogram EMED` in a `.txt` input) and an electron beam runs RADDOSE-3D's MicroED engine instead. It prints what RADDOSE-3D prints -- the dose by stopping power, how many electrons scatter elastically, once, inelastically and productively, the information coefficient, and the voltage and thickness that would maximise it -- and writes RADDOSE-3D's two MicroED files, under their fixed names, into the directory `OUTPUT_PREFIX` points to:

| file | contents |
|---|---|
| `outputMicroED.CSV` | the beam energy, dose, scattering counts, information coefficient, and best voltage and thickness |
| `outputMicroED_slices.csv` | the dose in each slice through the crystal, about 5 nm thick |

MicroED ignores the wedges, escape, the container and the decay model. The calculation is deterministic and takes well under a second.

## Escape modelling and the seed

Photoelectron and fluorescence escape are Monte Carlo, so two runs give slightly different doses. Set `seed` in the YAML -- RADDOSE-3D's `--seed` -- to repeat a run exactly. The port draws its random numbers in RADDOSE-3D's order, so with the same seed it gives the same dose RADDOSE-3D does. With no seed one is chosen, and the script prints it.

## What is not supported

Input that asks for something not ported is refused with a message saying what -- when the input is read, or at the latest when the calculation starts -- rather than ignored:

- the Monte Carlo and XFEL subprograms (`Subprogram MonteCarlo`, `GOS`, `XFEL`)
- `AbsCoefCalc RDV2` and `RDV3`, which run the original Fortran RADDOSE as an external program
- `Type Experimental`, which RADDOSE-3D cannot build from an input file either; `Type ExperimentalPGM` works

See `ROADMAP.md` at the top of the repository for when.
