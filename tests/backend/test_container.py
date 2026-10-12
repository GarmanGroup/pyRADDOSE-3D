"""Tests for sample container attenuation.

The conformance corpus checks two containers end to end against RADDOSE-3D
fetching NIST live. These check the parts of the lookup a single energy
cannot isolate.
"""

from __future__ import annotations

import itertools
import math

import pytest

from pyraddose_3d.backend.container import (
    _tables,
    container_transmission,
    elemental_mass_attenuation,
    known_mixtures,
    mixture_mass_attenuation,
)

COPPER = 29
COPPER_K_EDGE_KEV = 8.97890


def test_every_nist_table_is_vendored() -> None:
    assert len(_tables("nist_elements.tsv")) == 92
    assert len(known_mixtures()) == 48
    assert {"water", "mylar", "pmma", "polyethylene", "pyrex"} <= known_mixtures()


def test_an_edge_is_two_rows_at_one_energy() -> None:
    """Below then above, in that order -- the walk depends on it."""
    rows = [
        mu
        for energy, mu in _tables("nist_elements.tsv")[str(COPPER)]
        if energy == pytest.approx(COPPER_K_EDGE_KEV)
    ]
    assert len(rows) == 2
    assert rows[0] < rows[1]


def test_exactly_at_an_edge_takes_the_value_above_it() -> None:
    """The Java keeps the *last* row at or below the energy."""
    below, above = (
        mu
        for energy, mu in _tables("nist_elements.tsv")[str(COPPER)]
        if energy == pytest.approx(COPPER_K_EDGE_KEV)
    )
    at_edge = elemental_mass_attenuation({"Cu": 1}, COPPER_K_EDGE_KEV)
    assert at_edge == pytest.approx(above)
    assert at_edge != pytest.approx(below)


def test_interpolation_is_linear_not_log_log() -> None:
    """Halfway between two rows gives the arithmetic mean of their values.

    NIST's own advice is to interpolate log-log, which would give the
    geometric mean here; RADDOSE-3D interpolates linearly, and so does this.
    """
    rows = _tables("nist_mixtures.tsv")["water"]
    (e1, mu1), (e2, mu2) = next(
        (a, b) for a, b in itertools.pairwise(rows) if a[0] <= 12.1 < b[0]
    )
    halfway = mixture_mass_attenuation("water", (e1 + e2) / 2)
    assert halfway == pytest.approx((mu1 + mu2) / 2)
    assert halfway != pytest.approx(math.sqrt(mu1 * mu2), rel=1e-3)


def test_elements_combine_by_mass_fraction() -> None:
    """Water by formula is close to, not equal to, NIST's water table.

    Close because both are water; not equal because NIST's compound table is
    its own calculation, and the formula route combines per-element tables
    with RADDOSE-3D's atomic weights.
    """
    by_formula = elemental_mass_attenuation({"H": 2, "O": 1}, 12.1)
    by_name = mixture_mass_attenuation("water", 12.1)
    assert by_formula == pytest.approx(by_name, rel=0.02)


def test_transmission() -> None:
    assert container_transmission(0.0, 1.0, 100.0) == 1.0
    assert container_transmission(2.0, 1.5, 100.0) == pytest.approx(
        math.exp(-2.0 * 1.5 * 0.01)
    )


def test_an_unknown_mixture_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="water"):
        mixture_mass_attenuation("kapton", 12.1)
