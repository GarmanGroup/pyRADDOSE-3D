"""MicroED against RADDOSE-3D: its two files and what it prints.

Each case's ``expected`` holds the Java's ``outputMicroED.CSV`` and
``outputMicroED_slices.csv``, and ``microed_report.txt``, the lines
``MicroED.CalculateEM`` prints -- which give the mean free paths to full
precision. The tolerances, and why, are in ``MANIFEST.yaml``.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import pytest
from test_conformance import CASE_DIRS, is_microed, manifest

from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import microed_csv, microed_report, microed_slices_csv
from pyraddose_3d.solve import microed

if TYPE_CHECKING:
    from pathlib import Path

    from pyraddose_3d.backend.microed import MicroEDResult

pytestmark = pytest.mark.conformance

MICROED_PARAMS = [
    pytest.param(case_dir, id=case_dir.name)
    for case_dir in CASE_DIRS
    if is_microed(case_dir)
]

TOLERANCES = manifest()["microed_tolerances"]


@functools.cache
def run(case_dir: Path) -> MicroEDResult:
    return microed(Experiment.from_yaml(case_dir / "config.yaml"))


def _numbers(line: str) -> list[float]:
    return [float(field) for field in line.split(",")]


def _close(java: float, port: float, relative: float) -> bool:
    if math.isnan(java) or math.isnan(port):
        return math.isnan(java) and math.isnan(port)
    return abs(java - port) <= relative * abs(java)


def test_there_are_microed_cases() -> None:
    assert len(MICROED_PARAMS) >= 14


@pytest.mark.parametrize("case_dir", MICROED_PARAMS)
def test_results_match_java(case_dir: Path) -> None:
    """``outputMicroED.CSV``: same header, same values, same grid answers."""
    java = (case_dir / "expected" / "outputMicroED.CSV").read_text().splitlines()
    port = microed_csv(run(case_dir)).splitlines()
    assert port[0] == java[0]
    names = [name.strip() for name in java[0].split(",")]
    for name, expected, got in zip(
        names, _numbers(java[1]), _numbers(port[1]), strict=True
    ):
        relative = (
            0.0
            if name in ("Beam_en", "Best_en", "Best_t")
            else (TOLERANCES["counts_relative"])
        )
        assert _close(expected, got, relative), f"{name}: {expected} -> {got}"


@pytest.mark.parametrize("case_dir", MICROED_PARAMS)
def test_slices_match_java(case_dir: Path) -> None:
    """``outputMicroED_slices.csv``: every slice, its depth and its dose."""
    java = (case_dir / "expected" / "outputMicroED_slices.csv").read_text().splitlines()
    port = microed_slices_csv(run(case_dir)).splitlines()
    assert port[0] == java[0]
    assert len(port) == len(java)
    for expected_line, got_line in zip(java[1:], port[1:], strict=True):
        expected, got = _numbers(expected_line), _numbers(got_line)
        assert got[:2] == expected[:2]
        assert _close(expected[2], got[2], TOLERANCES["slice_dose_relative"]), (
            f"{expected_line} -> {got_line}"
        )


@pytest.mark.parametrize("case_dir", MICROED_PARAMS)
def test_report_matches_java(case_dir: Path) -> None:
    """What ``CalculateEM`` prints: the same lines, the same numbers."""
    java = (case_dir / "expected" / "microed_report.txt").read_text().splitlines()
    port = microed_report(run(case_dir)).splitlines()
    assert len(port) == len(java)
    relative = {
        "Elastic Lambda": TOLERANCES["elastic_mean_free_path_relative"],
        "GOS Inelastic Lambda": TOLERANCES["inelastic_mean_free_path_relative"],
    }
    for expected_line, got_line in zip(java, port, strict=True):
        if ": " not in expected_line or expected_line.startswith("Warning: "):
            assert got_line == expected_line
            continue
        label, expected = expected_line.split(": ", 1)
        got_label, got = got_line.split(": ", 1)
        assert got_label == label
        expected_text, _, unit = expected.partition(" ")
        got_text, _, got_unit = got.partition(" ")
        assert got_unit == unit, label
        assert _close(
            float(expected_text),
            float(got_text),
            relative.get(label, TOLERANCES["counts_relative"]),
        ), f"{label}: {expected_text} -> {got_text}"
