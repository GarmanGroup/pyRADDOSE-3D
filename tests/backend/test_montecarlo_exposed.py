"""The Monte Carlo's exposed region: the voxels the beam crosses.

RADDOSE-3D issue #17, fixed in both by #66. Case 151's crystal and beam:
a 4 x 6 x 8 um cuboid at 2 voxels per um, in a top-hat 2 um wide and 6 um
tall, which covers the crystal's height and half its width. The same tests
are ``MonteCarloExposedRegionTest`` in RADDOSE-3D.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from pyraddose_3d.backend.montecarlo.simple import (
    SimpleSetup,
    _exposed_region,
    _exposed_volume,
    _grid,
)
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import monte_carlo_setup

CASE = "tests/conformance/cases/151-montecarlo-still-at-60/config.yaml"


@pytest.fixture(scope="module")
def setup() -> SimpleSetup:
    return monte_carlo_setup(Experiment.from_yaml(CASE))


def _placed(
    setup: SimpleSetup, start: float, end: float, offset_x: float = 0.0
) -> SimpleSetup:
    return dataclasses.replace(
        setup,
        start_angle=math.radians(start),
        end_angle=math.radians(end),
        start_offset=(offset_x, 0.0, 0.0),
    )


def _at(exposed: np.ndarray, x: float, y: float, z: float) -> bool:
    """The voxel at x, y, z um in the crystal's own frame."""
    return bool(exposed[round((x + 2) * 2), round((y + 3) * 2), round((z + 4) * 2)])


def test_the_case_is_as_described(setup: SimpleSetup) -> None:
    assert _grid(setup).size == (9, 13, 17)
    assert setup.beam.collimation == (2.0, 6.0)


def test_a_centred_unturned_crystal_s_region_is_as_it_was(
    setup: SimpleSetup,
) -> None:
    """The column |x| <= 1 um, as the old test chose, and an unscaled
    volume: such a crystal's results are unchanged."""
    centred = _placed(setup, 0, 0)
    exposed, share = _exposed_region(centred, _grid(centred))
    x = np.arange(9) / 2 - 2
    assert np.array_equal(
        exposed, np.broadcast_to((np.abs(x) <= 1)[:, None, None], exposed.shape)
    )
    assert share == 1.0


def test_an_offset_moves_the_region_and_hangs_the_beam_off_the_edge(
    setup: SimpleSetup,
) -> None:
    """A StartOffset of 2 um along x puts the beam at x from -3 to -1 um in
    the crystal: half of it off the crystal, so the volume halves."""
    offset = _placed(setup, 0, 0, offset_x=2.0)
    exposed, share = _exposed_region(offset, _grid(offset))
    assert _at(exposed, -2, 0, 0)
    assert _at(exposed, -1, 0, 0)
    assert not _at(exposed, -0.5, 0, 0)
    assert not _at(exposed, 0, 0, 0)
    assert share == 0.5


def test_a_turned_still_s_region_is_the_slab_the_beam_crosses(
    setup: SimpleSetup,
) -> None:
    """At 60 degrees, |x cos 60 + z sin 60| <= 1: 4 by 2.31 um of the
    section against 2 by 8 centred, a share of 0.577, within 1%."""
    turned = _placed(setup, 60, 60)
    exposed, share = _exposed_region(turned, _grid(turned))
    assert _at(exposed, 2, 0, -2)
    assert not _at(exposed, 2, 0, 0.5)
    assert _at(exposed, -1, 0, 1.5)
    assert not _at(exposed, 0, 0, 2)
    expected = 4 * (2 / math.sin(math.radians(60))) / (2 * 8)
    assert share == pytest.approx(expected, rel=0.01)


def test_a_rotation_s_region_is_every_voxel_the_beam_crosses(
    setup: SimpleSetup,
) -> None:
    """From 0 to 90 degrees the corner at x = 2, z = 0 comes into the beam
    and the one at x = 2, z = 4 never does."""
    rotation = _placed(setup, 0, 90)
    exposed, share = _exposed_region(rotation, _grid(rotation))
    assert _at(exposed, 2, 0, 0)
    assert not _at(exposed, 2, 0, 4)
    assert share > 1


def test_an_xfel_pulse_stands_at_its_start(setup: SimpleSetup) -> None:
    """Its photons do: a wedge from 60 to 90 degrees is a still at 60."""
    pulse = dataclasses.replace(_placed(setup, 60, 90), xfel=True)
    still = _placed(setup, 60, 60)
    assert np.array_equal(
        _exposed_region(pulse, _grid(pulse))[0],
        _exposed_region(still, _grid(still))[0],
    )


def test_the_exposed_volume_is_scaled_and_no_more_than_the_crystal(
    setup: SimpleSetup,
) -> None:
    """The aperture by the depth, 2 x 6 x 8 um, scaled; never past the
    crystal's 192 um^3."""
    volume = 4 * 6 * 8 * 1e-12
    assert _exposed_volume(setup, 8000, volume, 1.0) == pytest.approx(96e-12)
    assert _exposed_volume(setup, 8000, volume, 0.5) == pytest.approx(48e-12)
    assert _exposed_volume(setup, 8000, volume, 3.0) == volume
