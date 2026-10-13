"""java.util.Random, against the JDK's own draws."""

from __future__ import annotations

import numpy as np
import pytest

from pyraddose_3d.backend.java_random import JavaRandom

#: ``new Random(20260917)``: ``nextInt`` cycling through ``BOUNDS``, then five
#: ``nextDouble``s. Printed by ``tests/backend/data/DumpRandom.java``, JDK 25.
BOUNDS = [1, 2, 7, 100, 128, 1000, 1500, 2147483647, 1073741825]
INTS = [
    0, 0, 6, 23, 92, 482, 48, 49541319, 596806285, 0, 1, 2, 0, 6, 935, 1314,
    1738868368, 345459484, 0, 0, 1, 32, 71, 729, 294, 1404681237, 231579150,
    0, 0, 5, 17, 96, 40, 229, 431331404, 524058390, 0, 0, 4, 79,
]  # fmt: skip
DOUBLES = [
    0.5167248883063394, 0.5888581299352071, 0.688051529377141,
    0.44006939497067166, 0.8544995952579331,
]  # fmt: skip


def test_matches_the_jdk() -> None:
    random = JavaRandom(20260917)
    assert [random.next_int(BOUNDS[i % len(BOUNDS)]) for i in range(40)] == INTS
    assert [random.next_double() for _ in range(5)] == DOUBLES


def test_a_negative_seed_matches_the_jdk() -> None:
    random = JavaRandom(-123456789)
    assert (random.next_int(1000), random.next_int(3)) == (933, 2)


def test_vectorised_draws_are_the_same_draws() -> None:
    """Including rejections, which shift every later draw along by one.

    A bound just over 2**30 rejects about half its candidates, so this
    exercises the rejection path thousands of times.
    """
    bounds = np.random.default_rng(1).choice([*BOUNDS, 2**30 + 1], size=20_000)
    one_at_a_time = JavaRandom(20260917)
    vectorised = JavaRandom(20260917)
    expected = [one_at_a_time.next_int(int(b)) for b in bounds]
    assert vectorised.next_ints(bounds).tolist() == expected
    assert vectorised.next_int(1000) == one_at_a_time.next_int(1000)


def test_vectorised_draws_start_from_the_jdks_first() -> None:
    assert (
        JavaRandom(20260917).next_ints(np.array(BOUNDS * 4 + BOUNDS[:4])).tolist()
        == INTS
    )


def test_refuses_a_bound_java_refuses() -> None:
    with pytest.raises(ValueError, match="positive"):
        JavaRandom(1).next_int(0)
