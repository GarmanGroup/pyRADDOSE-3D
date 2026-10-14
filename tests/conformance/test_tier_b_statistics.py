"""The tier-B statistics, checked before they are trusted with the port.

Welch's t against SciPy's, and by simulation with a fixed seed: the
false-alarm rate is what it is set to, and a bias of the size
``detectable_bias`` claims is caught as often as it claims.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _tier_b import compare, critical_t, detectable_bias, welch
from scipy import stats

#: Simulated comparisons per check: a rate of 0.2 or 0.8 is then known to
#: about +-0.006 (one standard error).
TRIALS = 4000

K = 20


def test_welch_is_scipys() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 1, 12), rng.normal(10.5, 2, 20)
    t, df = welch(a.tolist(), b.tolist())
    expected = stats.ttest_ind(b, a, equal_var=False)
    assert t == pytest.approx(expected.statistic, rel=1e-12)
    assert df == pytest.approx(expected.df, rel=1e-12)


def test_samples_without_spread() -> None:
    assert welch([1.0, 1.0], [1.0, 1.0])[0] == 0.0
    assert welch([1.0, 1.0], [2.0, 2.0])[0] == math.inf
    assert compare("x", [1.0, 1.0], [1.0, 1.0], tests=1).passed


def test_the_threshold_splits_the_rate_between_the_tests() -> None:
    """Bonferroni: each of 75 tests gets 1/75 of the family's 1e-3."""
    threshold = critical_t(38, tests=75)
    assert 2 * stats.t.sf(threshold, 38) == pytest.approx(1e-3 / 75)
    assert threshold == pytest.approx(5.0, abs=0.01)


def _null_and_biased(bias: float, alpha: float) -> float:
    """The fraction of simulated comparisons that fail."""
    rng = np.random.default_rng(20260917)
    cv = 0.015
    failed = 0
    for _ in range(TRIALS):
        reference = rng.normal(1.0, cv, K)
        port = rng.normal(1.0 + bias, cv, K)
        t, df = welch(reference.tolist(), port.tolist())
        failed += abs(t) > critical_t(df, tests=1, alpha=alpha)
    return failed / TRIALS


def test_the_false_alarm_rate_is_as_set() -> None:
    """At a rate large enough to see in 4000 trials."""
    assert _null_and_biased(0.0, alpha=0.2) == pytest.approx(0.2, abs=0.02)


def test_the_detectable_bias_is_caught_as_often_as_claimed() -> None:
    bias = detectable_bias(0.015, K, tests=1, alpha=0.05)
    assert _null_and_biased(bias, alpha=0.05) == pytest.approx(0.8, abs=0.03)


def test_the_corpus_detects_about_twice_the_spread() -> None:
    """With 20 runs a side and 75 tests, about 1.9 run-to-run deviations."""
    assert detectable_bias(1.0, K, tests=75) == pytest.approx(1.9, abs=0.1)
