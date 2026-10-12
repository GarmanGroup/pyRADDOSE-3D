"""The port's output files against the Java's own, case by case.

``test_conformance`` checks the numbers; this checks the files they are
written into. Anything that reads RADDOSE-3D's output should read the
port's unchanged, so the text around the numbers must be identical, and each
number must be printed the same way -- the same decimal places, the same
exponent style -- even where its last digits differ.

Four comparisons, from strictest to loosest:

- ``Summary.txt``, ``Summary.csv`` and ``DoseState.R`` are kept whole in the
  corpus and compared line by line: text exact, number layout exact, values
  within the manifest's tolerances.
- For the two stills, which do not rotate and so agree with the Java to
  about 1e-15, ``Summary.txt`` is compared byte for byte.
- The per-voxel and per-image CSVs are written, read back with the same
  parser ``generate.py`` applies to the Java's, and reduced to the corpus's
  statistics.
- Their first lines -- headers, and voxel coordinates -- are compared
  exactly.
"""

from __future__ import annotations

import functools
import json
import re
from typing import TYPE_CHECKING, Any

import pytest
from _array_stats import compare as compare_arrays
from _array_stats import from_simulation
from _java_files import R_TIMESTAMP, parse_summary, reduce_java_outputs
from test_conformance import CASE_DIRS, CASE_PARAMS, load_case, manifest, tolerance_for

from pyraddose_3d import outputs
from pyraddose_3d.solve import simulate

if TYPE_CHECKING:
    from pathlib import Path

    from pyraddose_3d.solve import Simulation

pytestmark = pytest.mark.conformance

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_LAYOUT = re.compile(
    r"[-+]?\d+(?:\.(?P<fraction>\d+))?(?:(?P<e>[eE])(?P<sign>[-+]?)\d+)?"
)


@functools.cache
def _simulation(case_dir: Path) -> Simulation:
    """One run per case, shared by every test here."""
    _, experiment, _ = load_case(case_dir)
    return simulate(experiment, record_images=True)


def _layout(token: str) -> tuple[int, str]:
    """What a format pattern fixes: decimal places and exponent style."""
    match = _LAYOUT.fullmatch(token)
    assert match is not None, token
    places = len(match["fraction"] or "")
    style = (
        "" if match["e"] is None else match["e"] + ("signed" if match["sign"] else "")
    )
    return places, style


def _unit(token: str) -> float:
    """One unit in the last printed place of a number."""
    places, style = _layout(token)
    exponent = int(re.split("[eE]", token)[1]) if style else 0
    return 10.0 ** (exponent - places)


def _compare_lines(
    expected: list[str],
    actual: list[str],
    relative: float,
    histogram_points: float,
) -> list[str]:
    """Line-by-line comparison: text exact, numbers by layout and value.

    A number may differ by ``relative`` of itself plus one printed unit,
    since two values a hair apart can round either side of a last digit.
    The dose histogram's percentages are compared in percentage points.
    """
    failures = []
    if len(expected) != len(actual):
        failures.append(f"{len(actual)} lines, the Java has {len(expected)}")
    for number, (want, got) in enumerate(zip(expected, actual, strict=False), 1):
        if _NUMBER.sub("#", want) != _NUMBER.sub("#", got):
            failures.append(f"line {number} text:\n  java: {want}\n  port: {got}")
            continue
        for java, port in zip(_NUMBER.findall(want), _NUMBER.findall(got), strict=True):
            if _layout(java) != _layout(port):
                failures.append(f"line {number}: {port!r} laid out unlike {java!r}")
                continue
            deviation = abs(float(port) - float(java))
            if want.startswith("Bin "):
                limit = histogram_points
            else:
                limit = relative * abs(float(java)) + _unit(java)
            if deviation > limit:
                failures.append(f"line {number}: {port} against {java}\n  {want}")
    return failures


@pytest.mark.parametrize("case_dir", CASE_PARAMS)
def test_summary_text_matches_java(case_dir: Path) -> None:
    """Every line of Summary.txt, with each number in its tolerance.

    Numbers are allowed the quantised quantities' tolerance, the loosest in
    tier A: this test is about the file, and ``test_conformance`` already
    holds each summary field to its own.
    """
    tolerances = manifest()["tolerances"]
    failures = _compare_lines(
        (case_dir / "expected" / "Summary.txt").read_text("utf-8").splitlines(),
        outputs.summary_text(_simulation(case_dir)).splitlines(),
        relative=tolerances["tier_A_quantised_quantities_relative"],
        histogram_points=tolerances["histogram_percentage_points"],
    )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("case_id", ["064-wedge-still", "065-wedge-still-decaying"])
def test_summary_text_is_identical_for_stills(case_id: str) -> None:
    """With no rotation there is no knife edge, and nothing to round apart.

    The stills agree with the Java to about 1e-15, so every printed digit
    should be the same: a byte-for-byte check on the formatting of every
    line, the histogram included.
    """
    case_dir = next(d for d in CASE_DIRS if d.name == case_id)
    expected = (case_dir / "expected" / "Summary.txt").read_text("utf-8")
    assert outputs.summary_text(_simulation(case_dir)) == expected


@pytest.mark.parametrize("case_dir", CASE_PARAMS)
def test_summary_csv_matches_java(case_dir: Path, tmp_path: Path) -> None:
    """Summary.csv's header, its layout, and each field read back.

    Reading it back with the parser ``generate.py`` applies to the Java's
    file, and comparing field by field at each field's own tolerance, checks
    that the columns are in the Java's order.
    """
    meta, _, reference = load_case(case_dir)
    expected = (case_dir / "expected" / "Summary.csv").read_text("utf-8").splitlines()
    text = outputs.summary_csv(_simulation(case_dir))
    actual = text.splitlines()

    assert actual[0] == expected[0]
    assert len(actual) == len(expected)
    for want, got in zip(expected[1:], actual[1:], strict=True):
        assert [_layout(t) for t in want.split(", ")] == [
            _layout(t) for t in got.split(", ")
        ]

    path = tmp_path / "Summary.csv"
    path.write_text(text, encoding="utf-8")
    tolerances = manifest()
    for want_row, got_row in zip(reference, parse_summary(path), strict=True):
        for field, value in want_row.items():
            limit = tolerance_for(meta["tier"], field, tolerances)
            assert abs(got_row[field] - value) <= limit * max(abs(value), 1e-30), field


def _r_dose(line: str) -> tuple[str, list[float]]:
    head, _, values = line.partition("(")
    return head, [float(v) for v in values.rstrip(")").split(",")]


@pytest.mark.parametrize("case_dir", CASE_PARAMS)
def test_dose_state_r_matches_java(case_dir: Path) -> None:
    """DoseState.R line by line, and its dose array voxel by voxel."""
    expected = (case_dir / "expected" / "DoseState.R").read_text("utf-8").splitlines()
    script = outputs.dose_state_r(_simulation(case_dir))
    actual = re.sub(
        r"^# Code generated .*$", R_TIMESTAMP, script, flags=re.MULTILINE
    ).splitlines()
    limit = manifest()["array_tolerances"]["DoseState.R/voxel"]

    assert len(actual) == len(expected)
    for want, got in zip(expected, actual, strict=True):
        if not want.startswith("dose[,,"):
            assert got == want
            continue
        want_head, want_values = _r_dose(want)
        got_head, got_values = _r_dose(got)
        assert got_head == want_head
        assert len(got_values) == len(want_values)
        for java, port in zip(want_values, got_values, strict=True):
            assert (java == 0) == (port == 0)
            assert abs(port - java) <= limit * max(abs(java), abs(port)), want_head


@pytest.mark.parametrize("case_dir", CASE_PARAMS)
def test_per_voxel_files_reduce_to_the_java_statistics(
    case_dir: Path, tmp_path: Path
) -> None:
    """Write the files, parse them as the Java's are parsed, and compare.

    Proves the column order, the masking and the parseability of the files,
    not just of the arrays behind them. The reduction must also match the
    one taken straight from the arrays to within single precision: values
    printed as a Java ``float`` read back as the shortest decimal that
    names it, not as the float itself, which is all writing should lose.
    """
    simulation = _simulation(case_dir)
    written = outputs.write_outputs(simulation, prefix=str(tmp_path / "out-"))
    assert sorted(p.name for p in written) == sorted(
        f"out-{name}"
        for name in (
            "Summary.csv", "Summary.txt", "DoseState.csv", "DoseState.R",
            "RDE.csv", "DWDs.csv", "VoxelDose.csv", "VoxelFluences.csv",
        )
    )  # fmt: skip

    reduced = reduce_java_outputs(tmp_path)
    expected = json.loads((case_dir / "expected" / "array_stats.json").read_text())
    failures, _ = compare_arrays(expected, reduced, manifest()["array_tolerances"])
    assert not failures, "\n".join(failures)

    single = dict.fromkeys(manifest()["array_tolerances"], 1e-6)
    failures, _ = compare_arrays(from_simulation(simulation), reduced, single)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("case_dir", CASE_PARAMS)
def test_first_lines_match_java(case_dir: Path) -> None:
    """Headers exactly, and the coordinates that open each per-voxel file.

    The voxel coordinates sit on the grid, so they print identically; the
    values after them on DoseState's first line are a face voxel's, and are
    left to the statistics.
    """
    expected: dict[str, Any] = json.loads(
        (case_dir / "expected" / "first_lines.json").read_text("utf-8")
    )
    simulation = _simulation(case_dir)
    texts = {
        "DWDs": outputs.dwds_csv(simulation),
        "RDE": outputs.rde_csv(simulation),
        "DoseState": outputs.dose_state_csv(simulation),
        "VoxelDose": outputs.voxel_dose_csv(simulation),
        "VoxelFluences": outputs.voxel_fluences_csv(simulation),
    }
    assert sorted(texts) == sorted(expected)
    for name, want in expected.items():
        got = texts[name].split("\n", 1)[0][:200]
        if name == "DoseState":
            assert got.split(",")[:3] == want.split(",")[:3]
        else:
            assert got == want, name
