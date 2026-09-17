"""Tests that enforce the package's structural invariants."""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).parent.parent / "src" / "pyraddose_3d"
BACKEND = SRC / "backend"

# The numerical core must stay callable with plain tensors, so that it can be
# handed autograd leaves, compiled, and benchmarked without a config object.
FORBIDDEN_IN_BACKEND = {"pydantic", "yaml", "pyraddose_3d.models"}


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Return every module name imported by a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _backend_files() -> list[pathlib.Path]:
    return sorted(BACKEND.rglob("*.py"))


def test_backend_package_exists() -> None:
    assert BACKEND.is_dir(), f"expected a backend package at {BACKEND}"


@pytest.mark.parametrize("path", _backend_files(), ids=lambda p: p.name)
def test_backend_does_not_import_config_machinery(path: pathlib.Path) -> None:
    """The backend takes tensors, not models.

    If this fails, the numerical core has grown a dependency on the
    configuration layer. That would make it impossible to call with bare
    tensors, which is what differentiability and benchmarking both need.
    Move the offending code into a manager and pass the values through
    ``make_backend_kwargs()`` instead.
    """
    imported = _imported_modules(path)
    for module in imported:
        root = module.split(".")[0]
        offending = {module, root} & FORBIDDEN_IN_BACKEND
        assert not offending, (
            f"{path.relative_to(SRC)} imports '{module}'. The backend must not "
            f"depend on the configuration layer -- see backend/__init__.py."
        )
