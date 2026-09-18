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

## Attribution

This is a derivative work of RADDOSE-3D, by the Garman Group at the University
of Oxford and contributors. If you use it, cite the original:

> Zeldin, O. B., Gerstel, M. and Garman, E. F. (2013). *RADDOSE-3D: time- and
> space-resolved modelling of dose in macromolecular crystallography*.
> J. Appl. Cryst. **46**, 1225-1230. doi:10.1107/S0021889813011461

`src/pyraddose_3d/data/MuCalcConstants.txt` is taken unmodified from
RADDOSE-3D. It holds Pathikrit Bandyopadhyay's parameterisation of the
McMaster tables of X-ray cross-sections.

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

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
