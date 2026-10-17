"""Builds the two data packages that hold ELSEPA's electron angular tables.

RADDOSE-3D's Monte Carlo and XFEL subprograms choose each elastic deflection
from ELSEPA's differential cross-sections, one table per element:
``constants/below_20000/{Z}.csv`` for electrons up to 20 keV, every element,
and ``constants/above_20000/{Z}.csv`` above it, up to Z = 52. Together they
are 100 MB, too much for pyRADDOSE-3D's core, so each directory becomes a
package of its own, installed with the ``montecarlo`` extra:

- ``packages/elsepa-low``, ``pyraddose-3d-elsepa-low``, 2.2 MB, committed;
- ``packages/elsepa-high``, ``pyraddose-3d-elsepa-high``, 23 MB, committed
  too: anyone installing from the repository needs it.

Each ``{Z}.csv`` is xz-compressed unchanged, so its SHA-256 can be checked
against RADDOSE-3D's file. ``MANIFEST.tsv`` records the hashes, and is
committed for both packages: a local build is checked against it.

Usage
-----
::

    python tests/backend/data/build_elsepa_packages.py ~/git/RADDOSE-3D [low|high]

Both packages are built unless one is named.
"""

from __future__ import annotations

import hashlib
import lzma
import subprocess
import sys
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[3] / "packages"

#: Package -> (RADDOSE-3D directory, where the tables go).
SOURCES = {
    "low": ("below_20000", PACKAGES / "elsepa-low/src/pyraddose_3d_elsepa_low/tables"),
    "high": (
        "above_20000",
        PACKAGES / "elsepa-high/src/pyraddose_3d_elsepa_high/tables",
    ),
}

#: xz's strongest setting: 2.2 MB and 23 MB, against 3.0 MB and 38 MB gzipped.
_PRESET = 9 | lzma.PRESET_EXTREME


def _commit(raddose: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(raddose), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def build(raddose: Path, which: str) -> int:
    """Compress one directory's tables into its package; return how many."""
    directory, out_dir = SOURCES[which]
    source = raddose / "constants" / directory
    tables = sorted(
        (int(path.stem), path) for path in source.glob("*.csv") if path.stem.isdigit()
    )
    if not tables:
        raise SystemExit(f"no tables in {source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# ELSEPA angular tables: RADDOSE-3D's constants/{directory}/ at "
        f"{_commit(raddose)}, each {{z}}.csv xz-compressed unchanged. sha256 and "
        f"bytes are the uncompressed file's.",
        "z\tsha256\tbytes",
    ]
    for z, path in tables:
        data = path.read_bytes()
        (out_dir / f"{z}.csv.xz").write_bytes(lzma.compress(data, preset=_PRESET))
        lines.append(f"{z}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}")
    (out_dir / "MANIFEST.tsv").write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(tables)


def main() -> None:
    """Build the packages named on the command line."""
    if len(sys.argv) not in (2, 3) or sys.argv[2:] not in ([], ["low"], ["high"]):
        print(__doc__)
        raise SystemExit(2)
    raddose = Path(sys.argv[1]).expanduser()
    for which in sys.argv[2:] or list(SOURCES):
        count = build(raddose, which)
        print(f"  elsepa-{which}: {count} tables")


if __name__ == "__main__":
    main()
