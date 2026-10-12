"""Numbers formatted as Java formats them, against Java's own output."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyraddose_3d._java_format import java_double, java_e, java_f, java_float

FIXTURE = Path(__file__).parent / "data" / "java_format.tsv"

#: The fixture's columns after the value, and how the port produces each.
FORMATTERS = {
    "%.0f": lambda x: java_f(x, 0),
    "%.1f": lambda x: java_f(x, 1),
    "%.2f": lambda x: java_f(x, 2),
    "%.3f": lambda x: java_f(x, 3),
    "%f": java_f,
    "%4.1f": lambda x: f"{java_f(x, 1):>4}",
    "%.1e": lambda x: java_e(x, 1),
    "%.2e": lambda x: java_e(x, 2),
    "%e": java_e,
    "Double.toString": java_double,
    "Float.toString": java_float,
}


def _rows() -> list[list[str]]:
    lines = FIXTURE.read_text("utf-8").splitlines()
    rows = [line.split("\t") for line in lines if line and not line.startswith("#")]
    header, body = rows[0], rows[1:]
    assert header == ["value", *FORMATTERS]
    return body


ROWS = _rows()


@pytest.mark.parametrize("pattern", list(FORMATTERS))
def test_matches_java(pattern: str) -> None:
    """Every value in the fixture, formatted as the JDK formatted it."""
    column = list(FORMATTERS).index(pattern) + 1
    formatter = FORMATTERS[pattern]
    wrong = [
        f"{row[0]}: java {row[column]!r}, port {formatter(float(row[0]))!r}"
        for row in ROWS
        if formatter(float(row[0])) != row[column]
    ]
    assert not wrong, "\n".join(wrong[:20])


def test_fixture_covers_the_cases_that_matter() -> None:
    """The cases Python gets wrong are in the fixture, so they are tested."""
    values = {row[0] for row in ROWS}
    # Half-up on the shortest decimal; Python would print 1.00 and 0.12.
    assert {"1.005", "0.125"} <= values
    # Layout boundaries, subnormals and specials.
    assert {"0.001", "10000000.0", "5e-324", "nan", "inf", "-0.0"} <= values


@pytest.mark.parametrize(
    ("value", "expected"), [(1.005, "1.01"), (0.125, "0.13"), (2.675, "2.68")]
)
def test_rounds_the_shortest_decimal_half_up(value: float, expected: str) -> None:
    """Where Python's own %.2f differs, for the reasons the module gives."""
    assert java_f(value, 2) == expected
    assert f"{value:.2f}" != expected
