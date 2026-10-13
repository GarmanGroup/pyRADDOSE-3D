"""Gathers the small tables RADDOSE-3D's Monte Carlo and XFEL subprograms read.

Run once to produce four files in ``src/pyraddose_3d/data/``, from a
RADDOSE-3D checkout's ``constants/``:

- ``auger_transitions.tsv``, from ``auger_linewidths/{Z}-{shell}.csv``;
- ``fluorescence_transitions.tsv``, from ``fl_linewidths/{Z}-{shell}.csv``;
- ``overvoltage_low.tsv`` and ``overvoltage_high.tsv``, from
  ``low_overvoltage.csv`` and ``high_overvoltage.csv``.

The 186 transition files become two tables with the element and shell as
columns, rows kept in each file's order, which is the order the Java sums
probabilities in. The overvoltage files hold one row per element with a
group of columns per shell; they are written one row per element per shell.

Usage
-----
::

    python tests/backend/data/build_monte_carlo_tables.py \\
        ~/git/RADDOSE-3D/constants src/pyraddose_3d/data

Numbers are written with :func:`repr`, the shortest string that reads back as
the same double, as ``build_element_tables.py`` does.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HEADERS = {
    "auger_transitions.tsv": (
        "# Auger transitions for each element (z) and initial vacancy (shell: "
        "0 = K, 1-3 = L1-L3, 4-8 = M1-M5), from RADDOSE-3D's "
        "constants/auger_linewidths/{z}-{shell}.csv, rows in file order. "
        "linewidth in eV, energy in keV; exit_shell and drop_shell are the "
        "two vacancies left behind."
    ),
    "fluorescence_transitions.tsv": (
        "# Fluorescence transitions for each element (z) and initial vacancy "
        "(shell), from RADDOSE-3D's constants/fl_linewidths/{z}-{shell}.csv, "
        "rows in file order. linewidth in eV, energy in keV; drop_shell is "
        "the vacancy left behind."
    ),
    "overvoltage_low.tsv": (
        "# Electron impact ionisation at low overvoltage, per element (z) and "
        "shell, from RADDOSE-3D's constants/low_overvoltage.csv: the binding "
        "energy in eV and coefficients a1-a5 (CoefCalcCompute's lowOvervoltages)."
    ),
    "overvoltage_high.tsv": (
        "# Electron impact ionisation at high overvoltage, per element (z) and "
        "shell, from RADDOSE-3D's constants/high_overvoltage.csv: b-, b+, A and "
        "g1-g4 (CoefCalcCompute's highOvervoltages)."
    ),
}

#: Per-shell group widths in the overvoltage files: a label then the numbers.
_OVERVOLTAGE = {
    "overvoltage_low.tsv": (
        "low_overvoltage.csv",
        6,
        ["binding_ev", *[f"a{i}" for i in range(1, 6)]],
    ),
    "overvoltage_high.tsv": (
        "high_overvoltage.csv",
        7,
        ["b_minus", "b_plus", "a", "g1", "g2", "g3", "g4"],
    ),
}

_NAME = re.compile(r"^(\d+)-(\d+)\.csv$")


def _number(text: str) -> str:
    return repr(float(text))


def _transitions(directory: Path, kind: str) -> list[list[str]]:
    rows = []
    files = []
    for path in directory.iterdir():
        match = _NAME.match(path.name)
        if match:
            files.append((int(match[1]), int(match[2]), path))
    for z, shell, path in sorted(files):
        with path.open(encoding="ascii") as handle:
            for record in csv.reader(handle):
                if not record or not record[0].strip():
                    continue
                label, linewidth, probability, energy, *shells = record
                rows.append(
                    [
                        str(z),
                        str(shell),
                        label.strip(),
                        _number(linewidth),
                        _number(probability),
                        _number(energy),
                        *(str(int(float(s))) for s in shells),
                    ]
                )
                width = 8 if kind == "auger" else 7
                if len(rows[-1]) != width:
                    raise ValueError(f"{path}: {record} has the wrong width")
    return rows


def _overvoltage(path: Path, width: int) -> list[list[str]]:
    rows = []
    with path.open(encoding="ascii") as handle:
        for record in csv.reader(handle):
            if not record:
                continue
            z = int(record[0])
            groups = (len(record) - 1) // (width + 1)
            for i in range(groups):
                start = 1 + i * (width + 1)
                label = record[start].strip()
                numbers = record[start + 1 : start + 1 + width]
                rows.append([str(z), str(i), label, *(_number(n) for n in numbers)])
    return rows


def write_tables(constants: Path, out_dir: Path) -> dict[str, int]:
    """Write the four tables, returning the rows written to each."""
    tables = {
        "auger_transitions.tsv": (
            [
                "z",
                "shell",
                "label",
                "linewidth_ev",
                "probability",
                "energy_kev",
                "exit_shell",
                "drop_shell",
            ],
            _transitions(constants / "auger_linewidths", "auger"),
        ),
        "fluorescence_transitions.tsv": (
            [
                "z",
                "shell",
                "label",
                "linewidth_ev",
                "probability",
                "energy_kev",
                "drop_shell",
            ],
            _transitions(constants / "fl_linewidths", "fluorescence"),
        ),
    }
    for name, (source, width, columns) in _OVERVOLTAGE.items():
        tables[name] = (
            ["z", "shell", "label", *columns],
            _overvoltage(constants / source, width),
        )
    written = {}
    for name, (columns, rows) in tables.items():
        lines = [HEADERS[name], "\t".join(columns), *("\t".join(r) for r in rows)]
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="ascii")
        written[name] = len(rows)
    return written


def main() -> None:
    """Convert the directory named on the command line."""
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    constants, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    for name, count in write_tables(constants, out_dir).items():
        print(f"  {name:32s} {count:5d} rows")


if __name__ == "__main__":
    main()
