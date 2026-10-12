"""Minimal valid models for tests, since nothing has a made-up default now.

Every field RADDOSE-3D requires is required by the models, so a test that
only cares about one setting still has to supply the rest. These supply them,
with the values of the conformance baseline, and let a test override what it
is about.
"""

from __future__ import annotations

from typing import Any

from pyraddose_3d.models.beam import GaussianBeam
from pyraddose_3d.models.crystal import CuboidCrystal
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.models.wedge import Wedge

__all__ = ["INSULIN", "beam", "crystal", "experiment", "wedge"]

#: The unit cell composition RADDOSE-3D's own example input uses.
INSULIN: dict[str, Any] = {
    "method": "unit_cell",
    "unit_cell": (78.02, 78.02, 78.02),
    "num_monomers": 24,
    "num_residues": 51,
}


def crystal(**overrides: Any) -> CuboidCrystal:
    """A 20 x 16 x 12 um cuboid of insulin."""
    return CuboidCrystal(
        **{"dimensions": (20.0, 16.0, 12.0), "composition": INSULIN, **overrides}
    )


def beam(**overrides: Any) -> GaussianBeam:
    """A 20 um Gaussian at 12.1 keV, uncollimated."""
    return GaussianBeam(
        **{"flux": 2e12, "energy": 12.1, "fwhm": (20.0, 20.0), **overrides}
    )


def wedge(**overrides: Any) -> Wedge:
    """0 to 90 degrees over 10 seconds."""
    return Wedge(**{"start": 0.0, "end": 90.0, "exposure_time": 10.0, **overrides})


def experiment(**overrides: Any) -> Experiment:
    """One of each of the above."""
    return Experiment(
        **{"crystal": crystal(), "beam": beam(), "wedges": [wedge()], **overrides}
    )
