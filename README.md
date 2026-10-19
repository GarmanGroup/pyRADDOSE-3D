# pyRADDOSE-3D

[![License](https://img.shields.io/pypi/l/pyRADDOSE-3D.svg?color=green)](https://github.com/GarmanGroup/pyRADDOSE-3D/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyRADDOSE-3D.svg?color=green)](https://pypi.org/project/pyRADDOSE-3D)
[![Python Version](https://img.shields.io/pypi/pyversions/pyRADDOSE-3D.svg?color=green)](https://python.org)
[![CI](https://github.com/GarmanGroup/pyRADDOSE-3D/actions/workflows/ci.yml/badge.svg)](https://github.com/GarmanGroup/pyRADDOSE-3D/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-garmangroup.github.io-blue)](https://garmangroup.github.io/pyRADDOSE-3D/)
[![codecov](https://codecov.io/gh/GarmanGroup/pyRADDOSE-3D/branch/main/graph/badge.svg)](https://codecov.io/gh/GarmanGroup/pyRADDOSE-3D)

A Python port of [RADDOSE-3D](https://github.com/GarmanGroup/RADDOSE-3D), which
models radiation dose in macromolecular crystallography.

**Documentation:** https://garmangroup.github.io/pyRADDOSE-3D/, with the example
notebooks run and their results shown.

NumPy by default, so it installs at ~100 MB. Install `pyRADDOSE-3D[torch]` and
the same code runs on PyTorch, making the deterministic dose calculation
differentiable -- so a data-collection strategy can be optimised by gradient
descent rather than by scanning.

Conformance with the Java is a tested property, not a claim: each module is
checked against reference values generated from RADDOSE-3D itself, and the
achieved agreement is recorded rather than merely asserted.

## Installing

pyRADDOSE-3D is not on PyPI yet. Clone it, then install it with uv or with
conda:

```sh
git clone https://github.com/GarmanGroup/pyRADDOSE-3D.git
cd pyRADDOSE-3D
```

### With uv

[uv](https://docs.astral.sh/uv/getting-started/installation/) makes a virtual
environment in the clone, `.venv`, and installs into it:

```sh
uv sync --no-dev --extra montecarlo
```

`--extra montecarlo` adds ELSEPA's electron tables, which the Monte Carlo, GOS
and XFEL subprograms read: two data packages in `packages/`, 2.3 MB up to
20 keV and 23 MB above it. Add `--extra plot` to draw the dose in 3D, and
`--extra torch` for gradients. Plain `uv sync`
installs the development tools as well (see [Development](#development)).

Then activate it, and run:

```sh
source .venv/bin/activate    # .venv\Scripts\activate on Windows
python programs/dose/run_dose.py docs/examples/inputs/insulin.txt
```

`uv run python ...` works without activating, but syncs first and so adds the
development tools unless given `--no-dev`. To open the example notebooks:

```sh
uv run --no-dev --group notebooks --with jupyterlab jupyter lab docs/examples
```

### With conda

[`environment.yml`](environment.yml) makes a conda environment,
`pyraddose-3d`, with the dependencies from conda-forge, the package installed
from the clone (editable, so a `git pull` updates it), the ELSEPA tables, and
JupyterLab and Matplotlib for the example notebooks and the dose contours:

```sh
conda env create -f environment.yml
conda activate pyraddose-3d
python programs/dose/run_dose.py docs/examples/inputs/insulin.txt
```

For gradients, add PyTorch: `conda install -n pyraddose-3d -c conda-forge pytorch`.

## Running a dose calculation

Existing RADDOSE-3D input files are read unchanged, and the output files are
written with RADDOSE-3D's names and formats, so the port can stand in for the
Java in an existing pipeline. Give
[`programs/dose/run_dose.py`](programs/dose/run_dose.py) an input file, as you
would give one to RADDOSE-3D:

```sh
python programs/dose/run_dose.py MyInput.txt
```

It prints the summary RADDOSE-3D prints and writes its files,
`output-Summary.txt`, `output-DoseState.csv` and the rest, to the current
directory; its other settings are at the top of the script. Or, from Python:

```python
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import summary_text, write_outputs
from pyraddose_3d.solve import simulate

experiment = Experiment.from_file("MyInput.txt")  # or a .yaml configuration
simulation = simulate(experiment, record_images=True)
print(summary_text(simulation))
write_outputs(simulation)  # output-Summary.txt, output-DoseState.csv, ...
```

To draw the dose as RADDOSE-3D's `DoseState.R` does, as surfaces where it
crosses 0.1, 20 and 30 MGy, but with matplotlib rather than R:

```sh
python programs/dose/plot_dose_contours.py output-DoseState.csv
```

It saves five views as PNG files and opens a window in which the contours can
be turned with the mouse; `pyraddose_3d.plotting` draws them from Python.

[`docs/examples/inputs`](docs/examples/inputs) holds example inputs, ready to
run: RADDOSE-3D's insulin and MicroED examples, each as a RADDOSE-3D input file
and as a YAML configuration. The notebooks beside them start by running one,
then walk through crystals and beams, composition, decay models, gradients,
choosing a beam energy and drawing the dose. They are committed without their
results; [the documentation](https://garmangroup.github.io/pyRADDOSE-3D/) shows
them run.

## Optimising a strategy

Hand `solve` a tensor for any tunable setting and the whole calculation runs
in torch, so the reported numbers come back differentiable:

```python
import torch
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import solve

experiment = Experiment.from_yaml("beamline.yaml")
fwhm = torch.tensor([20.0, 20.0], dtype=torch.float64, requires_grad=True)

peak = solve(experiment, {"beam.fwhm": fwhm})[0].max_dose
peak.backward()
fwhm.grad          # how the peak dose responds to widening the beam
```

With no overrides it stays on NumPy, which is what a torch-less install runs
and what the conformance corpus checks.

Only settings that carry a real gradient are accepted. Crystal geometry is
refused with the reason: the voxel occupancy test is a boolean parity count,
so autograd sees only the attenuation path and misses voxels entering or
leaving the crystal -- it returns a confident number that is 20% wrong. The
end of a wedge is refused for a different reason: it sets the number of
images and nothing else, so its derivative is zero wherever it exists. See
`TUNABLE` and `FIXED` in `pyraddose_3d.solve`.

Photoelectron and fluorescence escape are the exception to all of this. They
are Monte Carlo, with each electron's direction drawn at random, so they have
no gradient and run on NumPy only. They do reproduce RADDOSE-3D exactly, given
its seed.

## Licence

**AGPL-3.0-only.** Not a preference: RADDOSE-3D is licensed AGPL-3.0, and this
is a derivative work of it, so it inherits that licence. Upstream grants no
"or any later version", so neither can this -- hence `AGPL-3.0-only` rather
than `-or-later`.

See `NOTICE` for what is derived from where.

## Attribution

This is a derivative work of RADDOSE-3D, by the Garman Group at the University
of Oxford and contributors. If you use it, cite the original:

> Zeldin, O. B., Gerstel, M. and Garman, E. F. (2013). *RADDOSE-3D: time- and
> space-resolved modelling of dose in macromolecular crystallography*.
> J. Appl. Cryst. **46**, 1225-1230. doi:10.1107/S0021889813011461

The element data in `src/pyraddose_3d/data/` comes from RADDOSE-3D's
`constants/MuCalcConstants.txt`, which holds Pathikrit Bandyopadhyay's
parameterisation of the McMaster tables of X-ray cross-sections. The values
are unmodified; the layout is not. That file is 87 rows of 80 unlabelled
tab-separated columns, so it is split here into named tables --
`elements.tsv`, `absorption_edges.tsv`, `cross_section_coefficients.tsv`,
`fluorescence.tsv`, `edge_ratios.tsv`, `electron_ionisation.tsv` and
`elsepa_elastic.tsv` -- each with a header and each narrow enough to read.
`tests/backend/data/build_element_tables.py` is the script that produced
them.

Numerical behaviour is deliberately reproduced as-is, including several known
quirks, each marked in the source. Where the port and the Java disagree, the
Java is the specification until a change is made to both.

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork GarmanGroup/pyRADDOSE-3D --clone
# or just
# gh repo clone GarmanGroup/pyRADDOSE-3D
cd pyRADDOSE-3D
uv sync
```

The Monte Carlo and XFEL subprograms read ELSEPA's electron angular tables,
which come in two data packages beside the core, in `packages/`, both committed:
`uv sync` installs them. `tests/backend/data/build_elsepa_packages.py` rebuilds
them from a RADDOSE-3D checkout, and the tests check them against each
package's `MANIFEST.tsv`.

Run tests:

```sh
uv run pytest
```

Run the example notebooks, which are tests too:

```sh
uv run --group notebooks pytest --nbmake docs/examples
```

Build and preview the documentation, executing the examples:

```sh
uv run --group docs mkdocs serve
```

Lint files:

```sh
uv run pre-commit run --all-files
```
