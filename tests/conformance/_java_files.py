"""Reading RADDOSE-3D's output files, wherever they came from.

``generate.py`` reduces the Java's files with these to build the corpus, and
``test_output_files.py`` reduces the port's files with the same functions, so a
file the port writes is read exactly as the Java's is. Kept apart from
``generate.py`` because that imports ``_cases``, a name ``tests/backend``
also uses: under pytest's default import mode, whichever is imported first
wins.
"""

from __future__ import annotations

import csv
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from _array_stats import dose_state, image_rows, per_image_series

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "R_TIMESTAMP",
    "SIMULATED_FILES",
    "SUMMARY_COLUMNS",
    "parse_summary",
    "reduce_java_outputs",
    "simulated_runs",
]

#: What the timestamp line of a kept DoseState.R is replaced with.
R_TIMESTAMP = "# Code generated <timestamp>"

#: Summary columns, in the order OutputSummaryCSV writes them.
SUMMARY_COLUMNS = [
    "wedge_number",
    "average_dwd",
    "last_dwd",
    "elastic_yield",
    "diffraction_efficiency",
    "ad_wc",
    "ad_exp_region",
    "max_dose",
    "dose_threshold",
    "abs_en_threshold",
    "tad",
    "dose_contrast",
    "used_volume",
    "wedge_absorbed_energy",
    "dose_inefficiency",
    "dose_inefficiency_pe",
]


def parse_summary(path: Path) -> list[dict[str, float]]:
    """Parse out-Summary.csv into one dict per wedge."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        values = [float(v) for v in line.split(",")]
        rows.append(dict(zip(SUMMARY_COLUMNS, values, strict=False)))
    return rows


def _rows(path: Path, *, skip: int) -> list[list[float]]:
    """Numeric rows of a Java CSV, tolerating its trailing commas."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[skip:]:
        fields = [f.strip() for f in line.split(",") if f.strip()]
        if fields:
            rows.append([float(f) for f in fields])
    return rows


def reduce_java_outputs(out_dir: Path) -> dict[str, Any]:
    """Reduce the Java's per-voxel and per-image files to statistics.

    Each file is parsed by its actual layout, then handed to the reducers in
    ``_array_stats`` -- the same ones the test applies to the port's arrays.

    ``DoseState`` has no header: ``x,y,z,dose,fluence,elastic`` per voxel
    inside the crystal. ``VoxelDose`` and ``VoxelFluences`` open with a row
    of ``x_y_z`` labels, then one row per image of the last wedge, every
    value followed by a comma. ``DWDs`` and ``RDE`` have a header and one
    row per image of every wedge, at full double precision.
    """
    stats: dict[str, Any] = {}

    state = np.array(_rows(out_dir / "out-DoseState.csv", skip=0))
    stats["DoseState"] = dose_state(
        state[:, 0:3], state[:, 3], state[:, 4], state[:, 5]
    )

    for name in ("VoxelDose", "VoxelFluences"):
        stats[name] = image_rows(np.array(_rows(out_dir / f"out-{name}.csv", skip=1)))

    dwds = np.array(_rows(out_dir / "out-DWDs.csv", skip=1))
    stats["DWDs"] = per_image_series(
        wedge=dwds[:, 0],
        angle=dwds[:, 2],
        dwd=dwds[:, 3],
        volume=dwds[:, 4],
        rde_1A=dwds[:, 5],
        rde_2A=dwds[:, 6],
        rde_3A=dwds[:, 7],
        rde_4A=dwds[:, 8],
        rde_max_res=dwds[:, 9],
    )

    rde = np.array(_rows(out_dir / "out-RDE.csv", skip=1))
    stats["RDE"] = per_image_series(
        wedge=rde[:, 0], angle=rde[:, 2], weighted=rde[:, 3], minimum=rde[:, 4]
    )
    return stats


#: The files the Monte Carlo and XFEL subprograms write to the working
#: directory, a row per run. MONTECARLO writes both, GOS and XFEL the second.
SIMULATED_FILES = ("outputMC.CSV", "outputXFEL.CSV")

#: What each run prints: "label: value", a %.3f number.
PRINTED = re.compile(r"^(?P<label>[A-Z][^:]*): (?P<value>-?\d+\.\d+|NaN|-?Infinity)$")


def simulated_runs(out_dir: Path, stdout: str, runs: int) -> dict[str, list[float]]:
    """Every value each run wrote or printed, keyed ``file/column`` or
    ``stdout/label``, a list of one value per run."""
    quantities: dict[str, list[float]] = {}
    for name in SIMULATED_FILES:
        path = out_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            header, *rows = [r for r in csv.reader(handle) if r]
        if [int(r[0]) for r in rows] != list(range(1, runs + 1)):
            raise RuntimeError(f"{name} does not have runs 1 to {runs}")
        for j, column in enumerate(header[1:], start=1):
            quantities[f"{name}/{column.strip()}"] = [float(r[j]) for r in rows]
    printed: dict[str, list[float]] = {}
    for line in stdout.splitlines():
        match = PRINTED.match(line.strip())
        if match:
            printed.setdefault(match["label"], []).append(float(match["value"]))
    for label, values in printed.items():
        if len(values) == runs:
            quantities[f"stdout/{label}"] = values
    return quantities
