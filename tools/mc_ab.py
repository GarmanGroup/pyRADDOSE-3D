"""The Monte Carlo's results, bit for bit, against another commit's.

The Monte Carlo, GOS and XFEL subprograms are being reworked so that they can
run on a GPU, and on NumPy every rework must leave their results exactly as
they were. This runs every one of the conformance corpus's Monte Carlo, GOS
and XFEL cases, at fewer photons, from fixed seeds, with the code in the
working tree and with the code at a base commit, checked out in a temporary
git worktree, and compares every field of every run exactly.

Only a comparison on one machine can be exact: NumPy's results move in the
last bit between NumPy versions and platforms, so goldens would not hold.

Usage, from the top of the repository::

    uv run python tools/mc_ab.py                    # against origin/main's merge base
    uv run python tools/mc_ab.py --base HEAD~1 --cases 160 180

It exits 1, listing what differs, if any field of any run does.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "tests" / "conformance" / "cases"
SUBPROGRAMS = ("montecarlo", "gos", "xfel")

#: Run in a fresh interpreter, with the tree's own src first on the path.
PROBE = r"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import monte_carlo

photons, runs, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[5]
seeds = [int(seed) for seed in sys.argv[4].split(",")]
results = {}
for path in sys.argv[6:]:
    base = Experiment.from_yaml(path)
    simulation = base.crystal.simulation.model_copy(
        update={"photons": photons, "runs": runs}
    )
    crystal = base.crystal.model_copy(update={"simulation": simulation})
    for seed in seeds:
        experiment = base.model_copy(update={"seed": seed, "crystal": crystal})
        results[f"{Path(path).parent.name}, seed {seed}"] = [
            {
                name: repr(sorted(value.items()) if isinstance(value, dict) else value)
                for name, value in vars(run).items()
            }
            for run in monte_carlo(experiment)
        ]
Path(out).write_text(json.dumps(results, indent=1, sort_keys=True))
"""


def monte_carlo_cases(prefixes: list[str]) -> list[Path]:
    """The corpus's Monte Carlo, GOS and XFEL cases' configurations."""
    found = []
    for config in sorted(CASES.glob("*/config.yaml")):
        if prefixes and not config.parent.name.startswith(tuple(prefixes)):
            continue
        crystal = yaml.safe_load(config.read_text("utf-8"))["crystal"]
        if crystal.get("subprogram") in SUBPROGRAMS:
            found.append(config)
    return found


def git(*args: str) -> str:
    """A git command's output, run at the top of the repository."""
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    """Run both sides, compare them, and say what differs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base",
        default="origin/main",
        help="compare with the merge base of HEAD and this (default origin/main)",
    )
    parser.add_argument("--photons", type=int, default=20000)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--cases", nargs="*", default=[], metavar="PREFIX")
    options = parser.parse_args()

    base = git("merge-base", "HEAD", options.base)
    cases = monte_carlo_cases(options.cases)
    print(f"{len(cases)} cases, {options.photons} photons x {options.runs} runs,")
    print(f"seeds {options.seeds}: the working tree against {base[:7]}")

    with tempfile.TemporaryDirectory() as scratch:
        tree = Path(scratch) / "base"
        git("worktree", "add", "--detach", str(tree), base)
        try:
            runs = {}
            for side, src in (("new", ROOT / "src"), ("old", tree / "src")):
                out = Path(scratch) / f"{side}.json"
                command = [
                    sys.executable,
                    "-c",
                    PROBE,
                    str(src),
                    str(options.photons),
                    str(options.runs),
                    options.seeds,
                    str(out),
                    *map(str, cases),
                ]
                runs[side] = (subprocess.Popen(command), out)
            for side, (job, _) in runs.items():
                if job.wait() != 0:
                    print(f"the {side} code's run failed")
                    return 1
            new, old = (json.loads(out.read_text()) for _, out in runs.values())
        finally:
            git("worktree", "remove", "--force", str(tree))

    differences = [
        f"{case}, run {index + 1}, {field}: {before.get(field)} -> {value}"
        for case in sorted(new)
        for index, (after, before) in enumerate(
            zip(new[case], old.get(case, []), strict=False)
        )
        for field, value in after.items()
        if value != before.get(field)
    ]
    missing = sorted(set(new) ^ set(old))
    for line in differences + [f"{case}: run by one side only" for case in missing]:
        print(line)
    if differences or missing:
        print(f"{len(differences)} fields differ")
        return 1
    print("every field of every run is the same")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
