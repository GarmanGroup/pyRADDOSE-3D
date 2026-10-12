"""Reports the achieved agreement at the end of a conformance run."""

from __future__ import annotations

from typing import Any

from _agreement import AGREEMENT, ARRAY_AGREEMENT


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Print how closely each case that ran agreed with the reference.

    Parameters
    ----------
    terminalreporter : Any
        pytest's reporter, used for its section formatting.
    """
    if AGREEMENT:
        terminalreporter.write_sep("-", "conformance: worst relative deviation")
        for case_id, (deviation, field) in sorted(
            AGREEMENT.items(), key=lambda item: -item[1][0]
        ):
            terminalreporter.write_line(f"  {deviation:8.2e}  {case_id:40s} {field}")

    if ARRAY_AGREEMENT:
        terminalreporter.write_sep(
            "-", "conformance arrays: largest fraction of tolerance used"
        )
        for case_id, (fraction, statistic) in sorted(
            ARRAY_AGREEMENT.items(), key=lambda item: -item[1][0]
        ):
            terminalreporter.write_line(f"  {fraction:8.2f}  {case_id:40s} {statistic}")
