"""Guards on the package's weight: what it depends on, imports and carries.

``ROADMAP.md``, *Keeping it light*, sets the rules; these tests hold them, so
that adding a dependency, an eager import or bulk data is a visible change.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "src" / "pyraddose_3d" / "data"

#: The runtime dependencies, each with its reason in ``pyproject.toml``.
DEPENDENCIES = {"gemmi", "numpy", "pydantic", "pyyaml", "scipy"}
#: The core package's data, in bytes; bulk data goes in an optional package.
DATA_BUDGET = 1_000_000


def _imported_after(statement: str, modules: tuple[str, ...]) -> list[str]:
    """Which of ``modules`` a fresh interpreter has loaded after ``statement``."""
    code = (
        f"import json, sys\n{statement}\n"
        f"print(json.dumps([m for m in {list(modules)!r} if m in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return list(json.loads(result.stdout))


def test_the_runtime_dependencies_are_the_listed_ones() -> None:
    """A new one has to be added here too, with its reason in pyproject.toml."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {
        re.split(r"[\s\[<>=!~;]", dependency, maxsplit=1)[0].lower()
        for dependency in project["project"]["dependencies"]
    }
    assert names == DEPENDENCIES


def test_importing_the_package_asks_nothing_of_its_metadata() -> None:
    """The version is written at build time: ``importlib.metadata`` took
    about 40% of the import."""
    assert _imported_after("import pyraddose_3d", ("importlib.metadata",)) == []


def test_the_solver_imports_neither_torch_nor_scipy() -> None:
    """Both are imported where they are needed: torch for gradients, scipy
    for ``erf`` and the normal quantile."""
    imported = _imported_after("import pyraddose_3d.solve", ("torch", "scipy"))
    assert imported == []


def test_the_core_package_data_stays_under_its_budget() -> None:
    size = sum(path.stat().st_size for path in DATA.rglob("*") if path.is_file())
    assert size < DATA_BUDGET, f"{size:,} bytes"
