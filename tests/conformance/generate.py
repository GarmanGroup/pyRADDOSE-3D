"""Regenerate the conformance corpus from the Java reference implementation.

Run deliberately, never in CI -- the committed reference outputs are the point
of the corpus, and regenerating them is a claim that behaviour changed on
purpose. Review the resulting diff.

Usage
-----
    RADDOSE3D_DIR=/path/to/RADDOSE-3D python tests/conformance/generate.py

The reference implementation is invoked via its compiled classes rather than
the distributed jar, so the corpus tracks the source tree being ported.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from _cases import SEED, Case, all_cases, config_for

from pyraddose_3d.models.experiment import Experiment

HERE = Path(__file__).parent
CASES_DIR = HERE / "cases"
MANIFEST = HERE / "MANIFEST.yaml"

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


def raddose_dir() -> Path:
    """Locate the Java implementation."""
    env = os.environ.get("RADDOSE3D_DIR")
    default = Path.home() / "git" / "RADDOSE-3D"
    path = Path(env) if env else default
    if not (path / "bin" / "se" / "raddose3D").is_dir() and not (path / "bin").is_dir():
        raise SystemExit(
            f"No compiled RADDOSE-3D at {path}. Set RADDOSE3D_DIR, and run "
            f"`ant build` there first."
        )
    return path


def java_classpath(root: Path) -> str:
    """Classpath for running the reference implementation."""
    return os.pathsep.join(
        [
            str(root / "bin"),
            str(root / "lib/commons-math3-3.6.1/commons-math3-3.6.1.jar"),
            str(root / "lib/antlr-3.4/antlr-3.4-cutdown.jar"),
            str(root),
        ]
    )


def reference_version(root: Path) -> str:
    """Version string the reference implementation reports."""
    out = subprocess.run(
        ["java", "-cp", str(root / "bin"), "se.raddo.raddose3D.Version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def run_reference(root: Path, input_text: str, out_dir: Path) -> None:
    """Run the reference implementation on one input file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.txt"
        input_path.write_text(input_text, encoding="utf-8")
        result = subprocess.run(
            [
                "java",
                "-cp",
                java_classpath(root),
                "se.raddo.raddose3D.RD3D",
                "--seed",
                str(SEED),
                "-i",
                str(input_path),
                "-p",
                str(out_dir / "out-"),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "reference run failed:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )


#: Directives whose argument is a path relative to this repository.
PATH_DIRECTIVES = ("PDB ", "CIF ", "SeqFile ", "ModelFile ")

REPO_ROOT = HERE.parent.parent


def resolve_paths(input_text: str) -> str:
    """Make repository-relative file paths absolute for the reference run."""
    out = []
    for line in input_text.splitlines():
        for directive in PATH_DIRECTIVES:
            if line.startswith(directive):
                keyword, _, value = line.partition(" ")
                candidate = REPO_ROOT / value.strip()
                if candidate.is_file():
                    line = f"{keyword} {candidate}"
                break
        out.append(line)
    return "\n".join(out) + "\n"


def parse_summary(path: Path) -> list[dict[str, float]]:
    """Parse out-Summary.csv into one dict per wedge."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        values = [float(v) for v in line.split(",")]
        rows.append(dict(zip(SUMMARY_COLUMNS, values, strict=False)))
    return rows


def summarise_array_csv(path: Path) -> dict[str, float] | None:
    """Reduce a large per-voxel output to statistics.

    The voxel-level files are megabytes; committing them would bloat the repo
    for no benefit. Statistics still catch gross error.
    """
    if not path.is_file():
        return None
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        for field in line.split(","):
            field = field.strip()
            try:
                values.append(float(field))
            except ValueError:
                continue
    if not values:
        return None
    values.sort()
    n = len(values)
    return {
        "count": float(n),
        "min": values[0],
        "max": values[-1],
        "sum": float(sum(values)),
        "mean": float(sum(values) / n),
        "median": values[n // 2],
        "p05": values[int(0.05 * n)],
        "p95": values[int(0.95 * n)],
    }


def generate_case(case: Case, root: Path, version: str) -> None:
    """Write config.yaml, metadata and reference outputs for one case."""
    case_dir = CASES_DIR / case.case_id
    expected = case_dir / "expected"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    expected.mkdir(parents=True)

    experiment = Experiment(**config_for(case))
    experiment.to_yaml(case_dir / "config.yaml")

    # Paths in a config are relative to this repository, but the reference is
    # run with its own tree as the working directory. Absolutise for the run
    # only -- the committed config stays portable.
    input_text = resolve_paths(experiment.to_raddose_txt())

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        run_reference(root, input_text, out_dir)

        summary = parse_summary(out_dir / "out-Summary.csv")
        (expected / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

        stats: dict[str, Any] = {}
        for name in ("DoseState", "VoxelDose", "VoxelFluences", "RDE", "DWDs"):
            reduced = summarise_array_csv(out_dir / f"out-{name}.csv")
            if reduced is not None:
                stats[name] = reduced
        (expected / "array_stats.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )

    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(
            {
                "case_id": case.case_id,
                "tier": case.tier,
                "exercises": case.exercises,
                "seed": SEED,
                "reference_version": version,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"  {case.case_id:38s} tier {case.tier}  {len(summary)} wedge(s)")


def main() -> None:
    """Regenerate every case."""
    root = raddose_dir()
    version = reference_version(root)
    java = subprocess.run(
        ["java", "-version"], capture_output=True, text=True
    ).stderr.splitlines()[0]

    print(f"reference: RADDOSE-3D {version}")
    print(f"jvm:       {java}")
    print(f"seed:      {SEED}\n")

    cases = all_cases()
    for case in cases:
        generate_case(case, root, version)

    MANIFEST.write_text(
        yaml.safe_dump(
            {
                "reference_version": version,
                "jvm": java,
                "seed": SEED,
                "case_count": len(cases),
                "tolerances": {
                    "tier_A_relative": 1.0e-5,
                    "tier_A_ratio_quantities_relative": 1.0e-4,
                    "tier_B_relative": 5.0e-2,
                },
                "notes": (
                    "Reference values are produced by the Java implementation on "
                    "the JVM named above. The same Java code varies by ~1e-6 "
                    "between JVMs, so regenerate deliberately and review the diff."
                ),
                "excluded": {
                    "AbsCoefCalc RDV2": "shells out to an unshipped binary",
                    "ContainerElemental/Mixture": "scrape NIST at runtime",
                },
                "caveats": {
                    "AbsCoefCalc EXPSM (CIF)": (
                        "Covered by case 034, but note CoefCalcFromCIF crashes "
                        "when the last token of _chemical_formula_sum is a bare "
                        "single-letter symbol (urea's 'C H4 N2 O' fails). COD "
                        "1008123 was chosen because its formula ends in 'Sb2'. "
                        "A protein mmCIF will not work at all: this is the "
                        "small-molecule route and needs _chemical_formula_sum."
                    ),
                },
                "keyword_notes": {
                    "AbsCoefCalc EXP": (
                        "PDB-based composition. The grammar token *named* PDB "
                        "matches the literal 'EXP', so 'AbsCoefCalc PDB' can "
                        "never parse."
                    ),
                    "AbsCoefCalc EXPSM": (
                        "CIF-based composition. The token named CIF matches "
                        "the literal 'EXPSM'."
                    ),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {len(cases)} cases and {MANIFEST.name}")


if __name__ == "__main__":
    main()
