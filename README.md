# pyRADDOSE-3D

[![License](https://img.shields.io/pypi/l/pyRADDOSE-3D.svg?color=green)](https://github.com/jdickerson95/pyRADDOSE-3D/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyRADDOSE-3D.svg?color=green)](https://pypi.org/project/pyRADDOSE-3D)
[![Python Version](https://img.shields.io/pypi/pyversions/pyRADDOSE-3D.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/pyRADDOSE-3D/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/pyRADDOSE-3D/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/pyRADDOSE-3D/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/pyRADDOSE-3D)

A Python port of [RADDOSE-3D](https://github.com/GarmanGroup/RADDOSE-3D), which
models radiation dose in macromolecular crystallography.

NumPy by default, so it installs at ~100 MB. Install `pyRADDOSE-3D[torch]` and
the same code runs on PyTorch, making the deterministic dose calculation
differentiable -- so a data-collection strategy can be optimised by gradient
descent rather than by scanning.

Conformance with the Java is a tested property, not a claim: each module is
checked against reference values generated from RADDOSE-3D itself, and the
achieved agreement is recorded rather than merely asserted.

## Running a dose calculation

Existing RADDOSE-3D input files are read unchanged, and the output files are
written with RADDOSE-3D's names and formats, so the port can stand in for the
Java in an existing pipeline. Edit the parameters at the top of
[`programs/dose/run_dose.py`](programs/dose/run_dose.py) and run it, or:

```python
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import summary_text, write_outputs
from pyraddose_3d.solve import simulate

experiment = Experiment.from_file("input.txt")  # or a .yaml configuration
simulation = simulate(experiment, record_images=True)
print(summary_text(simulation))
write_outputs(simulation)  # output-Summary.txt, output-DoseState.csv, ...
```

The notebooks in [`docs/examples`](docs/examples) walk through crystals and
beams, composition, decay models, gradients and choosing a beam energy.

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
gh repo fork jdickerson95/pyRADDOSE-3D --clone
# or just
# gh repo clone jdickerson95/pyRADDOSE-3D
cd pyRADDOSE-3D
uv sync
```

The Monte Carlo and XFEL subprograms read ELSEPA's electron angular tables,
which come in two data packages beside the core, in `packages/`: the `montecarlo`
extra installs them. The tables above 20 keV, 23 MB compressed, are not
committed, and the tests that need them fail until they are built from a
RADDOSE-3D checkout:

```sh
uv run python tests/backend/data/build_elsepa_packages.py ~/git/RADDOSE-3D high
```

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
