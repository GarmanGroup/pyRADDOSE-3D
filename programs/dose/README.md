# Dose program example script and configs

This directory contains an example script for calculating the dose a crystal absorbs during data collection, and two example inputs describing the same experiment.

## Files

- `run_dose.py` - The Python script for running a dose calculation. Edit the parameters at the top: the path to your input, where to write the outputs, and whether to write the per-image files.
- `dose_example_config.yaml` - An example YAML configuration, commented with each setting's units and its RADDOSE-3D keyword.
- `dose_example_input.txt` - The same experiment as a RADDOSE-3D input file. This is `MyInput.txt` from RADDOSE-3D, unmodified: existing input files are read as they are.

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

## What is not supported yet

Input that asks for something not yet ported is refused with a message saying what -- when the input is read, or at the latest when the calculation starts -- rather than ignored:

- photoelectron and fluorescence escape, and the surrounding material
- the Leal et al. decay model
- crystals from `.obj` files
- pink beams (`EnergyFWHM`)
- the Monte Carlo, XFEL and MicroED subprograms

See `ROADMAP.md` at the top of the repository for when.
