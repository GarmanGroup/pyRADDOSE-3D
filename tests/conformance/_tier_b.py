"""Tier B: comparing the port with RADDOSE-3D as two samples.

The Monte Carlo and XFEL subprograms follow each electron with random
numbers whose order depends on its history, so the port cannot reproduce
the Java draw for draw. Each tier-B case is instead run K times on each
side, and the means of a few headline quantities are compared with Welch's
t-test, which does not assume the two sides vary equally.

The tests are a family: across the corpus, the chance that any one of them
fails when the port and the Java agree is held to ``FAMILY_ALPHA``, by
Bonferroni's correction. A false alarm is not a flaky test, though. The
seeds are fixed, so the same code gives the same t every run; a false alarm
can only appear when a change alters how random numbers are used, and then
it stays until the change is understood.

How small a disagreement the tests can see is set by K and by each
quantity's run-to-run spread, and is recorded per case by ``generate.py``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FAMILY_ALPHA",
    "GATED",
    "POWER",
    "Comparison",
    "compare",
    "critical_t",
    "detectable_bias",
    "welch",
]

#: The chance that any tier-B test in the corpus fails when nothing is wrong.
FAMILY_ALPHA = 1e-3

#: The chance a bias of the recorded detectable size is caught.
POWER = 0.8

#: What each subprogram's test compares, keyed as ``generate.py`` keys a
#: run's values: ``file/column`` for what it writes, ``stdout/label`` for
#: what it prints. For each, the dose to the whole crystal and to its
#: exposed region, RADDOSE-3D's own estimate of the dose, the elastic
#: scattering, and the ionisations per atom.
GATED: dict[str, tuple[str, ...]] = {
    "montecarlo": (
        "outputMC.CSV/MC-ADWC",
        "outputMC.CSV/MC-ADER",
        "outputMC.CSV/RD3D-ADWC",
        "outputMC.CSV/numEl",
        "outputXFEL.CSV/ions per atom",
    ),
    # GOS writes its doses in outputXFEL.CSV's vResolvedNoCut column, and
    # prints its exposed-region dose without writing it.
    "gos": (
        "outputXFEL.CSV/vResolvedNoCut",
        "stdout/RADDOSE-GOS average dose exposed region (ADER)",
        "outputXFEL.CSV/RD3D-ADWC",
        "outputXFEL.CSV/Diffraction efficiency",
        "outputXFEL.CSV/ions per atom",
    ),
    "xfel": (
        "outputXFEL.CSV/XFEL-ADWC",
        "outputXFEL.CSV/XFEL-ADER",
        "outputXFEL.CSV/RD3D-ADWC",
        "outputXFEL.CSV/Diffraction efficiency",
        "outputXFEL.CSV/ions per atom",
    ),
}


def welch(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's t statistic for the means of ``b`` and ``a``, and its df.

    Parameters
    ----------
    a, b : Sequence[float]
        The two samples, at least two values each.

    Returns
    -------
    tuple[float, float]
        ``t``, positive when ``b``'s mean is the larger, and the
        Welch-Satterthwaite degrees of freedom. Two samples with no spread
        give ``t`` 0 if their means are equal and infinite if not.
    """
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each sample needs at least two values")
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    se2_a = statistics.variance(a) / len(a)
    se2_b = statistics.variance(b) / len(b)
    se2 = se2_a + se2_b
    if se2 == 0.0:
        if mean_a == mean_b:
            return 0.0, float(len(a) + len(b) - 2)
        return math.copysign(math.inf, mean_b - mean_a), float(len(a) + len(b) - 2)
    df = se2**2 / (se2_a**2 / (len(a) - 1) + se2_b**2 / (len(b) - 1))
    return (mean_b - mean_a) / math.sqrt(se2), df


def critical_t(df: float, tests: int, alpha: float = FAMILY_ALPHA) -> float:
    """The two-sided threshold for one of ``tests`` comparisons.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    tests : int
        How many comparisons share the family-wise rate.
    alpha : float
        The family-wise false-alarm rate.

    Returns
    -------
    float
        ``|t|`` above this fails.
    """
    return float(stats.t.isf(alpha / tests / 2, df))


def detectable_bias(
    cv: float, k: int, tests: int, alpha: float = FAMILY_ALPHA, power: float = POWER
) -> float:
    """The relative bias caught with probability ``power``, K runs a side.

    With both sides spreading by ``cv`` of the mean, a bias of ``delta``
    gives a t of about ``delta / (cv * sqrt(2 / k))``, which must clear the
    threshold by the ``power`` quantile.

    Parameters
    ----------
    cv : float
        Run-to-run standard deviation over the mean.
    k : int
        Runs on each side.
    tests : int
        How many comparisons share the family-wise rate.
    alpha : float
        The family-wise false-alarm rate.
    power : float
        The chance of catching the bias.

    Returns
    -------
    float
        The bias, relative to the mean.
    """
    df = 2 * k - 2
    margin = critical_t(df, tests, alpha) + float(stats.t.ppf(power, df))
    return margin * cv * math.sqrt(2 / k)


@dataclass(frozen=True)
class Comparison:
    """One quantity, the reference's runs against the port's.

    Attributes
    ----------
    quantity : str
        What was compared.
    reference_mean, port_mean : float
        The two means.
    t, df : float
        Welch's statistic and degrees of freedom.
    threshold : float
        ``|t|`` above this fails.
    variance_ratio : float
        The port's variance over the reference's. A diagnostic, not a test:
        it says whether the port is as noisy as the Java.
    """

    quantity: str
    reference_mean: float
    port_mean: float
    t: float
    df: float
    threshold: float
    variance_ratio: float

    @property
    def passed(self) -> bool:
        """Whether the means agree within the threshold."""
        return abs(self.t) <= self.threshold

    def __str__(self) -> str:
        """One line, for a failure message or a report."""
        return (
            f"{self.quantity}: reference {self.reference_mean:.6g}, port "
            f"{self.port_mean:.6g}, t = {self.t:+.2f} (|t| <= {self.threshold:.2f} "
            f"passes, df {self.df:.1f}), variance ratio {self.variance_ratio:.2f}"
        )


def compare(
    quantity: str, reference: Sequence[float], port: Sequence[float], tests: int
) -> Comparison:
    """Compare one quantity's runs.

    Parameters
    ----------
    quantity : str
        What is compared, for messages.
    reference, port : Sequence[float]
        Each side's runs.
    tests : int
        How many comparisons share the family-wise rate.

    Returns
    -------
    Comparison
        The outcome.
    """
    t, df = welch(reference, port)
    var_ref = statistics.variance(reference)
    var_port = statistics.variance(port)
    if var_ref:
        ratio = var_port / var_ref
    else:
        ratio = 1.0 if var_port == 0.0 else math.inf
    return Comparison(
        quantity=quantity,
        reference_mean=statistics.fmean(reference),
        port_mean=statistics.fmean(port),
        t=t,
        df=df,
        threshold=critical_t(df, tests),
        variance_ratio=ratio,
    )
