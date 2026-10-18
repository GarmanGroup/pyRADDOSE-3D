# Dose program example script and configs

This directory contains the script for calculating the dose a crystal absorbs during data collection. Its example inputs are in `docs/examples/inputs`.

## Files

- `run_dose.py` - The Python script for running a dose calculation. Give it an input on the command line, or edit the parameters at the top: the path to your input, where to write the outputs, and whether to write the per-image files.
- `plot_dose_contours.py` - Draws the dose `run_dose.py` wrote, as RADDOSE-3D's `DoseState.R` does, with matplotlib.

In `docs/examples/inputs`:

- `insulin.txt` - An insulin crystal, as a RADDOSE-3D input file. This is `MyInput.txt` from RADDOSE-3D, unmodified: existing input files are read as they are.
- `insulin.yaml` - The same experiment as a YAML configuration, commented with each setting's units and its RADDOSE-3D keyword.
- `microed.txt` - A MicroED experiment, a protein microcrystal in a 300 keV electron beam: RADDOSE-3D's own `ED_input_cell.txt`, unmodified.
- `microed.yaml` - The same, as YAML.

## Running

Give it an input file, as you would give one to RADDOSE-3D:

```bash
python run_dose.py MyInput.txt
```

With no input named it runs `insulin.txt`, so from the top of the repository

```bash
python programs/dose/run_dose.py
```

runs the example. A path ending in `.yaml` or `.yml` is read as a YAML configuration, and anything else as a RADDOSE-3D input file.

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

## Dose contours

`output-DoseState.R` draws the final dose in R, as RADDOSE-3D's does. `plot_dose_contours.py` draws the same with matplotlib (`pip install "pyRADDOSE-3D[plot]"`, or `--extra plot` with uv): a surface where the dose crosses 0.1, 20 and 30 MGy, in light blue, dark blue and red, in a box marking the crystal. Give it the `DoseState.csv` a run wrote:

```bash
python run_dose.py MyInput.txt
python plot_dose_contours.py output-DoseState.csv
```

It saves the R script's five views, as `output-DoseContours_<theta>_<phi>.png`, and opens a window in which the contours can be turned with the mouse. The levels, colours, opacities and views are at the top of the script: the R script's levels suit a crystal that reaches about 30 MGy, so give one that gets much less or more its own. `docs/examples/07_dose_contours.ipynb` draws them from Python.

## MicroED

A crystal with `subprogram: microed` (`Subprogram EMED` in a `.txt` input) and an electron beam runs RADDOSE-3D's MicroED engine instead. It prints what RADDOSE-3D prints -- the dose by stopping power, how many electrons scatter elastically, once, inelastically and productively, the information coefficient, and the voltage and thickness that would maximise it -- and writes RADDOSE-3D's two MicroED files, under their fixed names, into the directory `OUTPUT_PREFIX` points to:

| file | contents |
|---|---|
| `outputMicroED.CSV` | the beam energy, dose, scattering counts, information coefficient, and best voltage and thickness |
| `outputMicroED_slices.csv` | the dose in each slice through the crystal, about 5 nm thick |

MicroED ignores the wedges, escape, the container and the decay model. The calculation is deterministic and takes well under a second.

## Monte Carlo and XFEL

A crystal with `subprogram: montecarlo`, `gos` or `xfel` (`Subprogram MonteCarlo`, `GOS` or `XFEL`) runs RADDOSE-3D's Monte Carlo instead: photon by photon, following every electron they free. `montecarlo` follows electrons by their stopping power; `gos` through every collision, with the secondaries they free and the relaxation of the atoms they ionise; `xfel` the same within one pulse, set by the beam's `pulse_energy` (mJ) and the wedge's exposure time, read as the pulse's length in femtoseconds. The pulse is a top-hat unless the beam's `pulse_shape` is `gaussian`; the exposure time is then its FWHM, and its photons all arrive within `pulse_window` FWHMs about its peak, 2 by default (`PulseShape` and `PulseWindow` in a `.txt` input, as RADDOSE-3D's #62). Each of the crystal's `simulation.runs` prints its doses as RADDOSE-3D does, and the files below are written under their fixed names into the directory `OUTPUT_PREFIX` points to; each wedge reruns the simulation, and the files hold the last.

| file | contents |
|---|---|
| `outputMC.CSV` | `montecarlo` only: a row per run of RADDOSE-3D-style and Monte Carlo doses, and the elastic count |
| `outputXFEL.CSV` | a row per run: the doses -- for `xfel`, those before each voxel's cutoff in time -- the diffraction efficiency, ionisations per atom and diffraction-weighted doses |
| `outputIonsPerAtom.CSV` | the last run's ionisations per atom of each element, in the exposed region |

The runs are random, and compared with RADDOSE-3D's statistically: the same seed repeats the port's runs, not RADDOSE-3D's. They need the ELSEPA data packages (the `montecarlo` extra). XFEL is the Monte Carlo's GOS model with a clock, as RADDOSE-3D's is with its #57, so a surrounding, a rotation and an offset work as in the other subprograms; a pulse lasts femtoseconds, so the crystal stands still at its wedge's start.

## Escape modelling and the seed

Photoelectron and fluorescence escape are Monte Carlo, so two runs give slightly different doses. Set `seed` in the YAML -- RADDOSE-3D's `--seed` -- to repeat a run exactly. The port draws its random numbers in RADDOSE-3D's order, so with the same seed it gives the same dose RADDOSE-3D does. With no seed one is chosen, and the script prints it.

## What is not supported

Input that asks for something not ported is refused with a message saying what -- when the input is read, or at the latest when the calculation starts -- rather than ignored:

- `AbsCoefCalc RDV2` and `RDV3`, which run the original Fortran RADDOSE as an external program
- `Type Experimental`, which RADDOSE-3D cannot build from an input file either; `Type ExperimentalPGM` works
