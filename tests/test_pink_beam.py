"""A pink beam shares each image between the energies it is sampled at.

RADDOSE-3D exposes a pink beam once per sampled energy, 200 times an image,
each with a 200th of the exposure. With a negligible spread it must report
what a monochromatic beam does -- RADDOSE-3D's ``PinkBeamTest`` checks the
same. It used not to: the exposed volume was 200 times too large, the
voxel fluences 200 times too small, and the surrounding gave 200 times its
dose (UPSTREAM_FIXES.md item 6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import image_series
from pyraddose_3d.solve import simulate

BASELINE = (
    Path(__file__).parent / "conformance" / "cases" / "000-baseline" / "config.yaml"
)


@pytest.fixture(scope="module")
def runs() -> tuple:
    config = Experiment.from_yaml(BASELINE).model_dump()
    config["wedges"] = [{**config["wedges"][0], "start": 0.0, "end": 30.0}]
    mono = simulate(Experiment.model_validate(config), record_images=True)
    config["beam"]["energy_fwhm"] = 1e-6
    pink = simulate(Experiment.model_validate(config), record_images=True)
    return mono, pink


def test_a_narrow_pink_beam_reports_as_a_monochromatic_one(runs: tuple) -> None:
    mono, pink = runs
    for field, value in mono.summaries[0].as_dict().items():
        assert pink.summaries[0].as_dict()[field] == pytest.approx(value, rel=1e-5)
    mono_series, pink_series = image_series(mono), image_series(pink)
    np.testing.assert_array_equal(pink_series["volume"], mono_series["volume"])
    np.testing.assert_allclose(pink_series["dwd"], mono_series["dwd"], rtol=1e-5)


def test_a_narrow_pink_beam_sums_its_fluence_over_its_energies(runs: tuple) -> None:
    mono, pink = runs
    np.testing.assert_allclose(
        np.asarray(pink.exposures[0].images.fluence),
        np.asarray(mono.exposures[0].images.fluence),
        rtol=1e-5,
        atol=1e-30,
    )


def test_coefficients_are_reported_at_the_beams_energy(runs: tuple) -> None:
    mono, pink = runs
    assert pink.coefficients == mono.coefficients
