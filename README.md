# pyRADDOSE-3D

[![License](https://img.shields.io/pypi/l/pyRADDOSE-3D.svg?color=green)](https://github.com/jdickerson95/pyRADDOSE-3D/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyRADDOSE-3D.svg?color=green)](https://pypi.org/project/pyRADDOSE-3D)
[![Python Version](https://img.shields.io/pypi/pyversions/pyRADDOSE-3D.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/pyRADDOSE-3D/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/pyRADDOSE-3D/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/pyRADDOSE-3D/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/pyRADDOSE-3D)

pyTorch port of RADDOSE-3D

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
