"""Beam profile and collimation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from pyraddose_3d.models._base import RD3DModel

__all__ = ["Beam", "Collimation", "GaussianBeam", "TophatBeam"]


class Collimation(RD3DModel):
    """Aperture placed in front of the beam.

    Attributes
    ----------
    shape : {"rectangular", "circular"}
        Aperture shape. A circular aperture is an ellipse inscribed in the
        same bounding box, so it always transmits less than the rectangular
        one for the same dimensions.
    x, y : float
        Full width of the aperture in each direction, in micrometres.
    """

    shape: Literal["rectangular", "circular"] = "rectangular"
    x: Annotated[float, Field(gt=0.0)] = 100.0
    y: Annotated[float, Field(gt=0.0)] = 100.0


class _BeamBase(RD3DModel):
    """Fields common to every beam profile."""

    flux: Annotated[float, Field(gt=0.0)] = 2.0e12
    energy: Annotated[float, Field(gt=0.0)] = 12.1
    collimation: Collimation = Field(default_factory=Collimation)


class GaussianBeam(_BeamBase):
    """A bivariate Gaussian profile.

    Attributes
    ----------
    fwhm : tuple[float, float]
        Full width at half maximum in x and y, in micrometres. For a
        horizontal goniometer x is vertical and y is horizontal.
    """

    type: Literal["gaussian"] = "gaussian"
    fwhm: tuple[float, float] = (20.0, 70.0)


class TophatBeam(_BeamBase):
    """A uniform beam filling its collimation aperture."""

    type: Literal["tophat"] = "tophat"


Beam = Annotated[GaussianBeam | TophatBeam, Field(discriminator="type")]
