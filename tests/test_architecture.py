"""Tests that enforce the package's structural invariants."""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).parent.parent / "src" / "pyraddose_3d"
BACKEND = SRC / "backend"

# The numerical core must stay callable with plain tensors, so that it can be
# handed autograd leaves, compiled, and benchmarked without a config object.
FORBIDDEN_IN_BACKEND = {"pydantic", "yaml", "pyraddose_3d.models"}

# torch is an optional extra, so importing it where a plain `import` statement
# executes would make a NumPy-only install fail outright. It may only be
# imported inside a function, reached when a tensor is already in hand.
DEFERRED_IN_BACKEND = {"torch"}


def _names(node: ast.AST) -> set[str]:
    """Return the module names an import statement names."""
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module:
        return {node.module}
    return set()


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Return every module name imported by a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return set().union(*(_names(node) for node in ast.walk(tree)), set())


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Return module names imported where the import runs at import time.

    Only statements directly in the module body count. An import nested in a
    function runs on first call, and one under ``if TYPE_CHECKING`` never
    runs at all; neither can break an install that lacks the package.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return set().union(*(_names(node) for node in tree.body), set())


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


@pytest.mark.parametrize("path", _backend_files(), ids=lambda p: p.name)
def test_backend_defers_optional_imports(path: pathlib.Path) -> None:
    """torch must never be imported at module scope.

    If this fails, ``import pyraddose_3d`` will raise ``ModuleNotFoundError``
    on an install without the ``[torch]`` extra -- which is the install CCP4
    would use. Move the import inside the function that needs it; by the time
    it runs there is a tensor in hand, so torch is necessarily present.
    """
    for module in _module_level_imports(path):
        root = module.split(".")[0]
        assert root not in DEFERRED_IN_BACKEND, (
            f"{path.relative_to(SRC)} imports '{module}' at module scope. "
            f"torch is an optional extra -- import it inside the function "
            f"that needs it, as backend/_array.py does."
        )
