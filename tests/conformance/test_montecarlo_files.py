"""The Monte Carlo and XFEL subprograms against RADDOSE-3D: tier B.

Each case's ``expected/runs.json`` holds every value RADDOSE-3D wrote or
printed in each of its runs, and ``case.yaml`` the quantities the tests
compare, with the bias each can detect. ``_tier_b.py`` has the statistics,
and ``MANIFEST.yaml`` why.

Before the comparison with the port is trusted, the tests are calibrated on
the reference alone: its runs split in two must agree, and must disagree
once one half is shifted by the bias the test claims to see.
"""

from __future__ import annotations

import functools
import json
import statistics
from typing import TYPE_CHECKING

import pytest
import yaml
from _tier_b import compare, detectable_bias
from test_conformance import CASE_DIRS, is_simulated, manifest

from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import monte_carlo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.conformance

TIER_B = [case_dir for case_dir in CASE_DIRS if is_simulated(case_dir)]
TIER_B_PARAMS = [pytest.param(case_dir, id=case_dir.name) for case_dir in TIER_B]


def _subprogram(case_dir: Path) -> str:
    return Experiment.from_yaml(case_dir / "config.yaml").crystal.subprogram


#: Being ported: the comparison is expected to fail, and must start passing
#: when the subprogram is.
_UNPORTED = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="being ported: ROADMAP.md, Milestone 7",
)
PORT_PARAMS = [
    pytest.param(
        case_dir,
        id=case_dir.name,
        marks=() if _subprogram(case_dir) == "montecarlo" else _UNPORTED,
    )
    for case_dir in TIER_B
]

#: How many comparisons share the family-wise false-alarm rate.
TESTS = manifest()["tier_b"]["comparisons"] if TIER_B else 0


def gated(case_dir: Path) -> list[str]:
    """The quantities the case's test compares."""
    meta = yaml.safe_load((case_dir / "case.yaml").read_text("utf-8"))
    return list(meta["gated"])


def reference(case_dir: Path) -> dict[str, list[float]]:
    """Each quantity's value in each of the reference's runs."""
    runs = json.loads((case_dir / "expected" / "runs.json").read_text("utf-8"))
    return runs["quantities"]


def halves(values: list[float]) -> tuple[list[float], list[float]]:
    """The first runs and the last."""
    half = len(values) // 2
    return values[:half], values[half:]


def test_every_comparison_is_counted() -> None:
    """The family-wise rate is shared by exactly the comparisons made."""
    assert sum(len(gated(case_dir)) for case_dir in TIER_B) == TESTS


@pytest.mark.parametrize("case_dir", TIER_B_PARAMS)
def test_the_references_rounding_is_negligible(case_dir: Path) -> None:
    """RADDOSE-3D writes %f to its files and prints %.3f.

    Rounding to a step adds ``step**2 / 12`` to a variance, and nothing to
    a mean. It must stay under 2% of each tested quantity's variance.
    """
    quantities = reference(case_dir)
    for quantity in gated(case_dir):
        step = 1e-3 if quantity.startswith("stdout/") else 1e-6
        variance = statistics.variance(quantities[quantity])
        assert step**2 / 12 < 0.02 * variance, quantity


@pytest.mark.parametrize("case_dir", TIER_B_PARAMS)
def test_the_reference_agrees_with_itself(case_dir: Path) -> None:
    """Its first runs against its last: the test must pass them.

    So the tests do not fail when nothing differs, and RADDOSE-3D's runs are
    alike: a subprogram whose later runs inherited state from earlier ones
    would fail here.
    """
    quantities = reference(case_dir)
    failures = []
    for quantity in gated(case_dir):
        first, last = halves(quantities[quantity])
        comparison = compare(quantity, first, last, TESTS)
        if not comparison.passed:
            failures.append(str(comparison))
    assert not failures, "\n".join(failures)


def test_a_bias_of_the_claimed_size_is_caught() -> None:
    """The halves again, the last moved up and down by what a test sees.

    The move is ``detectable_bias`` times the mean, added: the spread stays
    as it was, as the statistics assume. Each should be caught four times
    in five. A case's quantities rise and fall together -- its doses and
    ionisations all follow how many photons were absorbed -- so whole cases
    are caught or missed together, and the share caught varies more than
    the count suggests: hence 60%, not 80%. Heavy-tailed runs would catch
    far fewer.
    """
    caught = []
    for case_dir in TIER_B:
        quantities = reference(case_dir)
        for quantity in gated(case_dir):
            values = quantities[quantity]
            first, last = halves(values)
            mean = statistics.fmean(values)
            cv = statistics.stdev(values) / abs(mean)
            shift = detectable_bias(cv, len(first), TESTS) * mean
            for sign in (1, -1):
                moved = [v + sign * shift for v in last]
                caught.append(not compare(quantity, first, moved, TESTS).passed)
    assert sum(caught) >= 0.6 * len(caught), f"caught {sum(caught)} of {len(caught)}"


@functools.cache
def port(case_dir: Path) -> dict[str, list[float]]:
    """The port's runs of a case, keyed as the reference's quantities."""
    experiment = Experiment.from_yaml(case_dir / "config.yaml")
    if experiment.crystal.subprogram != "montecarlo":
        raise NotImplementedError(f"Subprogram {experiment.crystal.subprogram}")
    runs = monte_carlo(experiment)
    return {
        "outputMC.CSV/MC-ADWC": [r.mc_adwc for r in runs],
        "outputMC.CSV/MC-ADER": [r.mc_ader for r in runs],
        "outputMC.CSV/RD3D-ADWC": [r.rd3d_adwc for r in runs],
        "outputMC.CSV/numEl": [r.elastic for r in runs],
        "outputXFEL.CSV/ions per atom": [r.ions_per_atom for r in runs],
    }


@pytest.mark.parametrize("case_dir", PORT_PARAMS)
def test_the_port_agrees_with_the_reference(case_dir: Path) -> None:
    """Each tested quantity's mean, the port's runs against RADDOSE-3D's."""
    ours, theirs = port(case_dir), reference(case_dir)
    comparisons = [compare(q, theirs[q], ours[q], TESTS) for q in gated(case_dir)]
    for comparison in comparisons:
        print(comparison)
    failures = [str(c) for c in comparisons if not c.passed]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("case_dir", PORT_PARAMS)
def test_the_port_agrees_with_itself(case_dir: Path) -> None:
    """Its first runs against its last, as the reference's are compared."""
    ours = port(case_dir)
    failures = []
    for quantity in gated(case_dir):
        first, last = halves(ours[quantity])
        comparison = compare(quantity, first, last, TESTS)
        if not comparison.passed:
            failures.append(str(comparison))
    assert not failures, "\n".join(failures)
