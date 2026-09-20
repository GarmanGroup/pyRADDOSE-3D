"""Reports the achieved agreement at the end of a conformance run."""

from __future__ import annotations

from typing import Any

from _agreement import AGREEMENT


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Print the worst relative deviation for each case that ran.

    Parameters
    ----------
    terminalreporter : Any
        pytest's reporter, used for its section formatting.
    """
    if not AGREEMENT:
        return

    terminalreporter.write_sep("-", "conformance: worst relative deviation")
    for case_id, (deviation, field) in sorted(
        AGREEMENT.items(), key=lambda item: -item[1][0]
    ):
        terminalreporter.write_line(f"  {deviation:8.2e}  {case_id:40s} {field}")
