---
title: pyRADDOSE-3D
description: Radiation dose in macromolecular crystallography, as RADDOSE-3D calculates it, in Python
---

# pyRADDOSE-3D

A Python port of [RADDOSE-3D](https://github.com/GarmanGroup/RADDOSE-3D), which
models the radiation dose a crystal absorbs during an X-ray diffraction
experiment -- voxel by voxel, image by image.

It reads RADDOSE-3D's input files unchanged and writes its output files in
the same formats, so it can stand in for the Java in an existing pipeline.
It also reads a YAML configuration, and it runs on NumPy by default or on
PyTorch, where the calculation becomes differentiable: a data-collection
strategy can be optimised by gradient descent rather than by scanning.

!!! note "Citing this work"

    pyRADDOSE-3D is a derivative of RADDOSE-3D. If you use it, cite the
    original:

    Zeldin, O. B., Gerstel, M. and Garman, E. F. (2013). RADDOSE-3D: time-
    and space-resolved modelling of dose in macromolecular crystallography.
    *J. Appl. Cryst.* **46**, 1225-1230.
    [doi:10.1107/S0021889813011461](https://doi.org/10.1107/S0021889813011461)

## Installing

```bash
pip install pyRADDOSE-3D          # NumPy only
pip install pyRADDOSE-3D[torch]   # with gradients
```

## A first run

```python
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import summary_text, write_outputs
from pyraddose_3d.solve import simulate

experiment = Experiment.from_file("input.txt")  # or a .yaml configuration
simulation = simulate(experiment, record_images=True)
print(summary_text(simulation))
write_outputs(simulation)  # output-Summary.txt, output-DoseState.csv, ...
```

Or edit and run [`programs/dose/run_dose.py`](programs/dose.md). The
[examples](examples/01_getting_started.ipynb) walk through the rest.

## Agreement with RADDOSE-3D

Every number is checked against RADDOSE-3D itself, on a corpus of 95
experiments, with each tolerance measured and justified rather than chosen;
the output files are compared with the Java's line by line. The 22 Monte
Carlo, GOS and XFEL experiments, which are random, are compared
statistically, over 20 runs on each side. See
[Agreement with RADDOSE-3D](conformance.md).

The whole MX calculation is ported, photoelectron and fluorescence escape
included: escape is Monte Carlo, and the port draws its random numbers in
the Java's order, so given RADDOSE-3D's seed it gives RADDOSE-3D's answer.
So is the MicroED engine (`Subprogram EMED`), RADDOSE-3D's dose to a thin
crystal in an electron beam, and its Monte Carlo and XFEL subprograms
(`Subprogram MonteCarlo`, `GOS` and `XFEL`), which follow every electron;
those are random, and agree with RADDOSE-3D statistically. Not ported:
`AbsCoefCalc RDV2`/`RDV3`, which run the original Fortran RADDOSE as an
external program, and three XFEL inputs RADDOSE-3D runs inconsistently -- a
surrounding, and a rotation or offset, which it applies to the voxels but
not the crystal. Asking for any of them is an error, not a silently
different answer.

## Licence

AGPL-3.0-only, as RADDOSE-3D is.
