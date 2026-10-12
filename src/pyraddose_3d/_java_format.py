"""Numbers formatted exactly as RADDOSE-3D's Java prints them.

The output files are meant to be the same files, so a script that reads
RADDOSE-3D's output reads the port's unchanged. Python's ``%`` formatting
gets the last digit wrong surprisingly often, for two reasons:

**Java rounds the shortest decimal, not the binary value.** ``String.format``
starts from the digits ``Double.toString`` would print and rounds those
half-up. Python rounds the exact binary value half-even. They part company
whenever the shortest form ends in a 5, which is common::

    value    Java %.2f   Python %.2f
    1.005    1.01        1.00         (1.005 is 1.00499999... in binary)
    0.125    0.13        0.12         (exact, but Python rounds half-even)

**``Double.toString`` and ``Float.toString`` have their own layout.**
Plain decimals between 1e-3 and 1e7, otherwise ``d.dddE±n`` with a capital
E, no plus sign and no padding: ``1.0E-5``, ``2.6469176E9``. Both always
show at least one digit after the point.

Since JDK 19 the digits are the shortest that round-trip, as Python's
``repr`` gives, with one refinement: where a single digit would do, Java
considers two-digit candidates too and takes whichever is closest to the
exact value. That only changes the answer where the gap between adjacent
values is enormous -- the smallest subnormal double is ``4.9E-324``, not
``5.0E-324``.

Checked against every pattern RADDOSE-3D uses, for about four hundred
values, in ``tests/data/java_format.tsv``.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Context, Decimal

import numpy as np

__all__ = ["java_double", "java_e", "java_f", "java_float"]

#: Enough digits for any double printed in full, which ``%f`` of 1e308 does.
_WIDE = Context(prec=800)


def _round_trips(candidate: Decimal, value: float, single: bool) -> bool:
    if single:
        return float(np.float32(float(candidate))) == value
    return float(candidate) == value


def _digits(value: float, single: bool) -> tuple[str, int]:
    """Java's digits for a finite, non-zero value, and its decimal exponent.

    Returns ``(digits, exponent)`` with the magnitude equal to
    ``d1.d2d3... * 10**exponent``.
    """
    text = np.format_float_scientific(
        np.float32(value) if single else np.float64(value), unique=True, trim="-"
    )
    mantissa, exponent_text = text.lstrip("-").split("e")
    digits = mantissa.replace(".", "").rstrip("0") or "0"
    exponent = int(exponent_text)
    if len(digits) == 1:
        exact = abs(Decimal(value))  # a float converts exactly
        two = exact.scaleb(-exponent).quantize(Decimal("0.1"), ROUND_HALF_EVEN)
        if (
            two < 10
            and two != int(digits)
            and _round_trips(two.scaleb(exponent), abs(value), single)
            and abs(two.scaleb(exponent) - exact)
            < abs(Decimal(int(digits)).scaleb(exponent) - exact)
        ):
            digits = str(two).replace(".", "").rstrip("0")
    return digits, exponent


def _non_finite(value: float) -> str:
    """How Java spells NaN and the infinities, whatever the pattern."""
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _to_string(value: float, single: bool) -> str:
    if not math.isfinite(value):
        return _non_finite(value)
    if value == 0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"

    digits, exponent = _digits(value, single)
    sign = "-" if value < 0 else ""
    if 1e-3 <= abs(value) < 1e7:
        if exponent >= 0:
            whole = digits[: exponent + 1].ljust(exponent + 1, "0")
            fraction = digits[exponent + 1 :] or "0"
        else:
            whole = "0"
            fraction = "0" * (-exponent - 1) + digits
        return f"{sign}{whole}.{fraction}"
    fraction = digits[1:] or "0"
    return f"{sign}{digits[0]}.{fraction}E{exponent}"


def java_double(value: float) -> str:
    """``Double.toString``: what ``"" + someDouble`` prints in Java."""
    return _to_string(float(value), single=False)


def java_float(value: float) -> str:
    """``Float.toString``: what ``"" + (float) someDouble`` prints in Java.

    The cast comes first, so a value that rounds to a power of ten, to zero
    or to infinity in single precision is laid out as that.
    """
    with np.errstate(over="ignore"):
        return _to_string(float(np.float32(value)), single=True)


def _decimal(value: float) -> Decimal:
    """The decimal Java's Formatter rounds: ``Double.toString``'s digits."""
    if value == 0:
        return Decimal(0)
    digits, exponent = _digits(value, single=False)
    sign = "-" if value < 0 else ""
    return Decimal(f"{sign}{digits[0]}.{digits[1:] or '0'}E{exponent}")


def java_f(value: float, places: int = 6) -> str:
    """``String.format("%.Nf", value)``, rounding half-up as Java does."""
    value = float(value)
    if not math.isfinite(value):
        return _non_finite(value)
    quantum = Decimal(1).scaleb(-places)
    rounded = _decimal(value).quantize(quantum, ROUND_HALF_UP, context=_WIDE)
    text = f"{rounded.copy_abs():f}"
    return "-" + text if math.copysign(1.0, value) < 0 else text


def java_e(value: float, places: int = 6) -> str:
    """``String.format("%.Ne", value)``: ``d.ddde±XX``, rounding half-up."""
    value = float(value)
    if not math.isfinite(value):
        return _non_finite(value)
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    if value == 0:
        return f"{sign}{0:.{places}f}e+00"
    number = _decimal(value).copy_abs()
    exponent = number.adjusted()
    quantum = Decimal(1).scaleb(-places)
    mantissa = number.scaleb(-exponent).quantize(quantum, ROUND_HALF_UP, context=_WIDE)
    if mantissa >= 10:
        exponent += 1
        mantissa = number.scaleb(-exponent).quantize(
            quantum, ROUND_HALF_UP, context=_WIDE
        )
    exponent_sign = "-" if exponent < 0 else "+"
    return f"{sign}{mantissa:f}e{exponent_sign}{abs(exponent):02d}"
