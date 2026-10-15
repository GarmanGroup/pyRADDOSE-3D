"""The XFEL subprogram's clock, and what it changes in a vacancy's relaxation.

Its whole runs are compared with RADDOSE-3D statistically, in
``tests/conformance/test_montecarlo_files.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.montecarlo.gos_transport import (
    Cascade,
    Clock,
    _Born,
    gos_medium,
)
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import monte_carlo_setup

#: Case 140's crystal.
CASE = "tests/conformance/cases/140-montecarlo-baseline/config.yaml"
#: Case 180, an XFEL pulse.
XFEL_CASE = "tests/conformance/cases/180-xfel-baseline/config.yaml"

_C = 299792458.0
_REST_KEV = 9.10938356e-31 * _C**2 / KEV_TO_JOULES


def test_an_electron_at_twice_its_rest_energy_flies_at_root_three_halves_c() -> None:
    """``getTimeToDistance``: gamma 2 is beta squared 3/4."""
    fs = Clock.flight(np.array([_REST_KEV]), np.array([1000.0]))[0]
    assert fs == pytest.approx(1000 / (math.sqrt(0.75) * _C * 1e9 / 1e15), rel=1e-12)


def test_a_one_ev_linewidth_lives_0_66_fs() -> None:
    """hbar over the linewidth: 6.582e-16 eV s over 1 eV."""
    assert Clock.lifetime(np.array([1.0]))[0] == pytest.approx(0.6582, rel=1e-3)


@pytest.fixture(scope="module")
def selenium():  # type: ignore[no-untyped-def]
    experiment = Experiment.from_yaml(CASE)
    data = experiment.model_dump()
    data["crystal"]["composition"]["protein_heavy_atoms"]["Se"] = 4.0
    setup = monte_carlo_setup(Experiment.model_validate(data))
    medium = gos_medium(setup.material, 13.0)
    row = next(i for i, z in enumerate(medium.z) if z == 34)
    return medium, row


class _Collect:
    """A sink that adds up, keeping the times of what it was given."""

    def __init__(self) -> None:
        self.resolved_total = 0.0
        self.times: list[float] = []
        self.ions = 0

    def dose(self, points, owner, energy) -> None:  # type: ignore[no-untyped-def]
        pass

    def resolved(self, points, owner, energy, time) -> None:  # type: ignore[no-untyped-def]
        self.resolved_total += float(np.sum(energy))
        self.times.extend(np.asarray(time).tolist())

    def ionise(self, points, owner, element, count, time) -> None:  # type: ignore[no-untyped-def]
        self.ions += int(np.sum(count))


def _relax(medium, row, clock, n=2000, time=0.0):  # type: ignore[no-untyped-def]
    sink = _Collect()
    born = _Born()
    Cascade(np.random.default_rng(5), medium, sink, clock).relax(
        np.zeros((n, 3)),
        np.zeros(n, dtype=np.int64),
        np.full(n, row),
        np.zeros(n, dtype=np.int64),
        born,
        np.full(n, time),
    )
    return sink, born


def test_gos_and_xfel_relax_a_vacancy_alike(selenium) -> None:  # type: ignore[no-untyped-def]
    """A selenium K vacancy fluoresces about half the time, with an 11 keV
    photon, which escapes in both. GOS mode left it at the atom until
    RADDOSE-3D #53. From the same draws, the two now leave the same."""
    medium, row = selenium
    gos, gos_born = _relax(medium, row, None)
    xfel, xfel_born = _relax(medium, row, Clock(math.inf))
    np.testing.assert_array_equal(
        np.concatenate(gos_born.energies), np.concatenate(xfel_born.energies)
    )
    assert gos.resolved_total == pytest.approx(xfel.resolved_total, rel=1e-12)


def test_a_vacancy_relaxes_only_before_last_time(selenium) -> None:  # type: ignore[no-untyped-def]
    medium, row = selenium
    sink, born = _relax(medium, row, Clock(1.0), time=1.0)
    assert sink.resolved_total == 0 and sink.ions == 0 and not born.energies


def test_what_a_transition_leaves_comes_its_lifetime_later(selenium) -> None:  # type: ignore[no-untyped-def]
    """The K vacancy's own deposit is at once; the Auger electron, its
    ionisation and the vacancies after it a lifetime later, and not at all
    past ``lastTime``."""
    medium, row = selenium
    sink, born = _relax(medium, row, Clock(1e-6))
    assert sink.resolved_total > 0 and set(sink.times) == {0.0}
    assert sink.ions == 0 and not born.energies
    sink, born = _relax(medium, row, Clock(math.inf))
    later = np.concatenate(born.times)
    assert later.min() > 0 and sink.ions > 0


def test_an_xfel_pulse_is_a_still_at_the_wedges_start() -> None:
    """A pulse lasts femtoseconds, so the crystal stands still through it, at
    the wedge's start, however far the wedge turns: as RADDOSE-3D has it
    since #56. XFEL turned the voxels, not the crystal, by whole radians."""
    from pyraddose_3d.backend.montecarlo.simple import _placements

    experiment = Experiment.from_yaml(XFEL_CASE)
    data = experiment.model_dump()
    data["wedges"][-1].update(start=30.0, end=120.0)
    setup = monte_carlo_setup(Experiment.model_validate(data))
    placement = _placements(setup, np.arange(0, setup.photons, 997))
    np.testing.assert_array_equal(placement.angle, math.radians(30.0))
