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
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from _cases import SEED, Case, all_cases, config_for
from _java_files import R_TIMESTAMP, parse_summary, reduce_java_outputs

from pyraddose_3d.models.experiment import Experiment

HERE = Path(__file__).parent
CASES_DIR = HERE / "cases"
MANIFEST = HERE / "MANIFEST.yaml"


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


def reference_commit(root: Path) -> str:
    """The RADDOSE-3D commit the reference was built from.

    The version string alone does not say: several fixes can share one.
    Marked ``-dirty`` when tracked files differ from the commit.
    """
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{commit}-dirty" if changed else commit


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


#: The files MicroED writes, under fixed names, to the working directory.
MICROED_FILES = ("outputMicroED.CSV", "outputMicroED_slices.csv")


def run_microed_reference(root: Path, input_text: str, out_dir: Path) -> str:
    """Run the reference on a MicroED input, returning what it printed.

    MicroED writes its files to the working directory rather than under the
    output prefix, so it runs in ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.txt"
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
        cwd=out_dir,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    written = all((out_dir / name).is_file() for name in MICROED_FILES)
    if result.returncode != 0 or not written:
        raise RuntimeError(
            f"reference run failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return result.stdout


#: The first line ``MicroED.CalculateEM`` prints: a warning, if the GOS
#: model lacks binding energies, and otherwise the density.
REPORT_STARTS = ("Warning: the element tables", "The density is")

#: Its last: the optimal thickness, or that there was no search.
REPORT_ENDS = ("The optimal thickness", "The optimal accelerating voltage and")


def microed_report(stdout: str) -> str:
    """The lines ``MicroED.CalculateEM`` prints, from the first to the
    optimal thickness."""
    lines = stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(REPORT_STARTS))
    end = next(i for i, line in enumerate(lines) if line.startswith(REPORT_ENDS))
    return "\n".join(lines[start : end + 1]) + "\n"


#: Directives whose argument is a path relative to this repository.
PATH_DIRECTIVES = ("PDB ", "CIF ", "SeqFile ", "ModelFile ", "File ")

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


def generate_case(case: Case, root: Path, version: str, commit: str) -> None:
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

    if experiment.crystal.subprogram == "microed":
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            stdout = run_microed_reference(root, input_text, out_dir)
            for name in MICROED_FILES:
                shutil.copyfile(out_dir / name, expected / name)
            (expected / "microed_report.txt").write_text(
                microed_report(stdout), encoding="utf-8"
            )
        write_case_metadata(case, case_dir, version, commit)
        print(f"  {case.case_id:38s} tier {case.tier}  MicroED")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        run_reference(root, input_text, out_dir)

        summary = parse_summary(out_dir / "out-Summary.csv")
        (expected / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

        # The human-readable summary, the summary CSV and the R script are
        # kept whole, so the port's writers can be compared with them line by
        # line. They are a few kilobytes; the per-voxel files are megabytes,
        # and are checked through their statistics instead, plus their first
        # line. The R script's timestamp is blanked, or every regeneration
        # would rewrite every one of them.
        # A path the reference was given absolute is made relative again, as
        # the configuration has it: an experimental beam names its image.
        for name in ("Summary.txt", "Summary.csv"):
            text = (out_dir / f"out-{name}").read_text(encoding="utf-8")
            (expected / name).write_text(
                text.replace(f"{REPO_ROOT}{os.sep}", ""), encoding="utf-8"
            )
        script = (out_dir / "out-DoseState.R").read_text(encoding="utf-8")
        (expected / "DoseState.R").write_text(
            re.sub(r"^# Code generated .*$", R_TIMESTAMP, script, flags=re.MULTILINE),
            encoding="utf-8",
        )
        first_lines = {}
        for name in ("DWDs", "RDE", "DoseState", "VoxelDose", "VoxelFluences"):
            with (out_dir / f"out-{name}.csv").open(encoding="utf-8") as handle:
                first_lines[name] = handle.readline().rstrip("\n")[:200]
        (expected / "first_lines.json").write_text(
            json.dumps(first_lines, indent=2) + "\n", encoding="utf-8"
        )

        stats = reduce_java_outputs(out_dir)
        (expected / "array_stats.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )

    write_case_metadata(case, case_dir, version, commit)
    print(f"  {case.case_id:38s} tier {case.tier}  {len(summary)} wedge(s)")


def write_case_metadata(case: Case, case_dir: Path, version: str, commit: str) -> None:
    """``case.yaml``: what the case exercises, and the reference it came from."""
    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(
            {
                "case_id": case.case_id,
                "tier": case.tier,
                "exercises": case.exercises,
                "seed": SEED,
                "reference_version": version,
                "reference_commit": commit,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    """Regenerate every case, or those named.

    ``--only`` and ``--skip`` take case-id prefixes. A skipped case keeps
    the outputs it has, and its ``case.yaml`` keeps the RADDOSE-3D commit
    they came from -- so a partial run is visible as one.

    The JVM must be the one the manifest records: the same Java code gives
    different face-voxel depths on different JVMs, and so a different
    corpus. ``--new-jvm`` accepts a deliberate change. (An older JVM fails
    in another way too: NIST refuses Java 8's requests, so the container
    cases cannot run on it.)
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--only", nargs="+", default=None, metavar="PREFIX")
    parser.add_argument("--skip", nargs="+", default=[], metavar="PREFIX")
    parser.add_argument("--new-jvm", action="store_true")
    options = parser.parse_args(argv)

    root = raddose_dir()
    version = reference_version(root)
    commit = reference_commit(root)
    java = subprocess.run(
        ["java", "-version"], capture_output=True, text=True
    ).stderr.splitlines()[0]

    recorded = yaml.safe_load(MANIFEST.read_text("utf-8")).get("jvm")
    if recorded and java != recorded and not options.new_jvm:
        raise SystemExit(
            f"java is {java!r}, but the corpus was generated with {recorded!r}. "
            f"Put that JVM first on PATH, or pass --new-jvm to change it."
        )

    print(f"reference: RADDOSE-3D {version}")
    print(f"jvm:       {java}")
    print(f"seed:      {SEED}\n")

    cases = all_cases()
    for case in cases:
        wanted = options.only is None or case.case_id.startswith(tuple(options.only))
        if wanted and not case.case_id.startswith(tuple(options.skip)):
            generate_case(case, root, version, commit)
        else:
            print(f"  {case.case_id:38s} skipped")

    MANIFEST.write_text(
        yaml.safe_dump(
            {
                "reference_version": version,
                "reference_commit": commit,
                "jvm": java,
                "seed": SEED,
                "case_count": len(cases),
                "tolerances": {
                    "tier_A_relative": 1.0e-4,
                    "tier_A_quantised_quantities_relative": 5.0e-3,
                    "tier_B_relative": 5.0e-2,
                    "histogram_percentage_points": 1.0,
                },
                "array_tolerances": {
                    "DoseState/totals": 1.0e-4,
                    "DoseState/extremes": 5.0e-3,
                    "DoseState/position_um": 5.0e-3,
                    "VoxelDose/sum": 5.0e-3,
                    "VoxelDose/max": 5.0e-4,
                    "VoxelFluences/sum": 5.0e-2,
                    "VoxelFluences/max": 5.0e-2,
                    "DWDs/dwd": 5.0e-3,
                    "DWDs/volume": 1.0e-12,
                    "DWDs/rde": 5.0e-4,
                    "DoseState.R/voxel": 5.0e-2,
                    "RDE/weighted": 1.0e-3,
                    "RDE/minimum": 5.0e-4,
                    "counts": 0,
                    "angle_degrees": 1.0e-9,
                },
                "microed_tolerances": {
                    "counts_relative": 1.0e-12,
                    "inelastic_mean_free_path_relative": 1.0e-13,
                    "elastic_mean_free_path_relative": 1.0e-13,
                    "slice_dose_relative": 1.0e-12,
                },
                "microed_tolerance_rationale": (
                    "MicroED is deterministic, and the port agrees with the "
                    "Java to the last printed digit or two: the productive "
                    "count to 3.1e-14 (case 131), the inelastic mean free "
                    "path to 5.9e-15, the elastic to 4.0e-16, and every slice "
                    "exactly. What remains comes from summing the elements in "
                    "atomic-number order, where the Java iterates a HashSet. "
                    "The tolerances allow 15x or more, for other platforms' "
                    "maths libraries. The grid searches' answers must be "
                    "exact."
                ),
                "array_tolerance_rationale": (
                    "Set by the same rule as the summary tolerances: the "
                    "worst disagreement with the reference across the corpus, "
                    "with headroom, justified because it is smaller than the "
                    "port's disagreement with itself when every rotation angle "
                    "is moved by one ulp. Statistics that average over the "
                    "crystal agree to 5e-5 and share the summary's 1e-4. The "
                    "rest are dominated by voxels whose centres lie exactly on "
                    "a crystal face: at some angles their path length through "
                    "the crystal flips between zero and the full chord on the "
                    "last bit of a cosine, and Math.cos is not bit-reproducible "
                    "across JVMs. Per-image fluence maxima are the most exposed "
                    "because the brightest voxel in an image is one on the "
                    "upstream face. How much a flip matters scales with "
                    "absorption: the worst case is the antimony CIF crystal, "
                    "at 3.6e-2 against the reference and 4.8e-2 against itself. "
                    "The stills, which do not rotate, and the AngleP 45 case, "
                    "which has no face on a voxel plane, agree to about 1e-8 "
                    "on every statistic -- the control showing the "
                    "disagreement is the knife edge and not the physics. "
                    "Counts of voxels, images and exposed voxels are exact. "
                    "The five RDE columns of DWDs are compared absolutely: "
                    "they are fractions of 1 that fall to exp(-40) on the "
                    "antimony crystal, where any relative difference is the "
                    "knife edge in the DWD multiplied by DWD over the 1/e "
                    "dose. Worst 2.6e-4 against the reference, on that "
                    "crystal, and 7.9e-4 against itself; the stills agree "
                    "to 2e-15. DoseState.R is compared voxel by voxel, which "
                    "is the knife edge undiluted: worst 1.6e-2 against the "
                    "reference and 4.6e-2 against itself, both on the "
                    "antimony crystal, so it shares the fluence maxima's 5e-2. "
                    "The weighted RDE is amplified the same way under the "
                    "Leal model, whose Gaussian term makes log RDE fall by "
                    "2 gamma^2 D per MGy: worst 7.1e-4 against the reference, "
                    "where the efficiency is down to 1e-4, and 2.6e-3 against "
                    "itself. Escape modelling does the same to each image's "
                    "maximum dose, which is one voxel's: a photoelectron's "
                    "deposit lands in the voxel its track position rounds to, "
                    "and a last-bit difference in a cosine moves it to the "
                    "next. Worst 3.1e-4 against the reference and 6.7e-4 "
                    "against itself, so VoxelDose/max goes from 1e-4 to 5e-4."
                ),
                "quantised_quantities": [
                    "dose_threshold",
                    "tad",
                    "dose_contrast",
                ],
                "tolerance_rationale": {
                    "tier_A_relative": (
                        "Set by how well conditioned the voxel occupancy test "
                        "is, not by how carefully the port was written. "
                        "calculateCrystalOccupancy decides membership by ray "
                        "parity, so a voxel whose centre lies exactly on a "
                        "face can go either way, and a crystal face aligned "
                        "with the voxel grid puts a whole plane of them in "
                        "that position. Moving every mesh vertex by one ulp "
                        "moves the reported sums by as much as 9.3e-2 "
                        "(baseline); the port's disagreement with the "
                        "reference is at worst 6.5e-5 (composition-cif, "
                        "diffraction_efficiency). Java cannot "
                        "reproduce itself to better than this either: "
                        "Math.cos and Math.sin are not required to be "
                        "bit-reproducible across JVMs."
                    ),
                    "tier_A_quantised_quantities_relative": (
                        "These three are quantised and cannot take a value "
                        "between two rungs. dose_threshold is a dose some "
                        "voxel actually has, so it moves in steps of the dose "
                        "histogram's bin gap -- median 1e-4 to 3e-3 relative "
                        "across the corpus. tad and dose_contrast divide by a "
                        "count of voxels above the threshold, so they move in "
                        "steps of one voxel in about five hundred, 2e-3. "
                        "Every disagreement measured is exactly one step: the "
                        "adjacent histogram bin, or one voxel. 5e-3 is a "
                        "couple of steps."
                    ),
                    "tier_B_relative": (
                        "For a calculation the port cannot reproduce draw for "
                        "draw, where only gross agreement is meaningful. No "
                        "case uses it: escape modelling draws from "
                        "java.util.Random in the Java's order, so its cases "
                        "are tier A."
                    ),
                    "histogram_percentage_points": (
                        "Summary.txt's dose histogram counts voxels, so a bin "
                        "moves in steps of one voxel -- about 0.2 percentage "
                        "points in these crystals -- whenever a voxel's dose "
                        "sits close to a bin edge. Worst 0.40 against the "
                        "reference (two voxels) and 0.42 against the port "
                        "itself when every angle moves by one ulp."
                    ),
                },
                "notes": (
                    "Reference values are produced by the Java implementation on "
                    "the JVM named above. The same Java code varies by ~1e-6 "
                    "between JVMs, so regenerate deliberately and review the diff."
                ),
                "excluded": {
                    "AbsCoefCalc RDV2": "shells out to an unshipped binary",
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
