"""Where the Monte Carlo spends its time: a baseline to measure GPU work by.

Runs a set of the conformance corpus's Monte Carlo, GOS and XFEL cases under
cProfile and splits each run's time into the parts a GPU would treat
differently: the GOS model's setup, the photon stage's geometry, the
electrons' transport, and the processing of the tallies. It also records
how many electrons each run followed and how many passes their tracking
took, since a pass over a few hundred electrons costs mostly overhead.

Usage, from the top of the repository::

    uv run python benchmarks/mc_profile.py                     # the default cases
    uv run python benchmarks/mc_profile.py --cases 152 160 --photons 100000

It prints a table and writes the numbers, with the machine they came from,
as JSON (``--out``, by default ``benchmarks/results/<system>-<machine>-<date>.json``).
"""

from __future__ import annotations

import argparse
import cProfile
import datetime
import json
import platform
import pstats
import time
from pathlib import Path

import numpy as np

from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import monte_carlo

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "tests" / "conformance" / "cases"

#: Cases chosen to cover each mode and its costs: a cuboid, a surrounding, a
#: 124-triangle cylinder in a surrounding, a heavy atom, GOS with and
#: without its surrounding, and XFEL with a top-hat and a Gaussian pulse.
DEFAULT_CASES = ["140", "148", "152", "154", "160", "162", "180", "187"]

#: The parts, as (label, module file, function name). A function's cumulative
#: time includes what it calls, so the parts are chosen not to nest.
PARTS = [
    ("GOS setup", "gos.py", "shell_table"),
    ("photon coefficients", "photons.py", "photon_coefficients"),
    ("geometry: entry_depth", "geometry.py", "entry_depth"),
    ("simple transport", "transport.py", "track"),
    ("GOS transport", "gos_transport.py", "track_gos"),
    ("processing (MC)", "simple.py", "_process"),
    ("processing (XFEL)", "xfel.py", "xfel_runs"),
]


def profile_case(config: Path, photons: int, runs: int, seed: int) -> dict:
    """One case's wall time, its parts, and its electrons and passes."""
    base = Experiment.from_yaml(config)
    simulation = base.crystal.simulation.model_copy(
        update={"photons": photons, "runs": runs}
    )
    experiment = base.model_copy(
        update={
            "seed": seed,
            "crystal": base.crystal.model_copy(update={"simulation": simulation}),
        }
    )
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    results = monte_carlo(experiment)
    profiler.disable()
    wall = time.perf_counter() - start

    stats = pstats.Stats(profiler).stats  # type: ignore[attr-defined]
    parts = {}
    for label, module, name in PARTS:
        seconds = sum(
            cumulative
            for (path, _line, function), (_, _, _, cumulative, _) in stats.items()
            if function == name and path.endswith(f"montecarlo/{module}")
        )
        if seconds:
            parts[label] = round(seconds, 3)
    return {
        "case": config.parent.name,
        "subprogram": base.crystal.subprogram,
        "photons": photons,
        "runs": runs,
        "wall_s": round(wall, 3),
        "parts_s": parts,
        "electrons": sum(r.diagnostics.get("electrons", 0) for r in results),
        "passes": max((r.diagnostics.get("passes", 0) for r in results), default=0),
    }


def main() -> None:
    """Profile the cases and report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES, metavar="PREFIX")
    parser.add_argument("--photons", type=int, default=100_000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    options = parser.parse_args()

    configs = [next(CASES.glob(f"{prefix}*/config.yaml")) for prefix in options.cases]
    rows = []
    for config in configs:
        row = profile_case(config, options.photons, options.runs, options.seed)
        rows.append(row)
        parts = ", ".join(f"{k} {v:.2f}" for k, v in row["parts_s"].items())
        print(
            f"{row['case']:40s} {row['wall_s']:7.2f} s  "
            f"{row['electrons']:8.0f} electrons  {row['passes']:4.0f} passes  "
            f"[{parts}]"
        )

    out = options.out or (
        ROOT
        / "benchmarks"
        / "results"
        / f"{platform.system()}-{platform.machine()}-{datetime.date.today()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "machine": {
                    "system": platform.platform(),
                    "processor": platform.processor() or platform.machine(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                },
                "date": datetime.datetime.now().isoformat(timespec="seconds"),
                "cases": rows,
            },
            indent=1,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
