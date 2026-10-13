"""Beam profile and collimation.

A field has a default only where RADDOSE-3D has one, and it is RADDOSE-3D's,
so that leaving a setting out of a YAML file means the same as leaving it out
of a ``.txt`` input. Flux, energy and a Gaussian's widths are required by the
Java and are required here.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

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
    x: Annotated[float, Field(gt=0.0)]
    y: Annotated[float, Field(gt=0.0)]


class _BeamBase(RD3DModel):
    """Fields common to every beam profile.

    Attributes
    ----------
    flux : float
        Photons per second.
    energy : float
        Photon energy in keV.
    collimation : Collimation | None
        The aperture, or ``None`` for an uncollimated beam -- which is what
        RADDOSE-3D does when an input file has no ``Collimation`` line.
    energy_fwhm : float | None
        ``EnergyFWHM``: a pink beam, Gaussian in energy with this full width
        at half maximum, in keV, about ``energy``. ``None`` is
        monochromatic.
    """

    flux: Annotated[float, Field(gt=0.0)]
    energy: Annotated[float, Field(gt=0.0)]
    collimation: Collimation | None = None
    energy_fwhm: Annotated[float, Field(gt=0.0)] | None = None


class GaussianBeam(_BeamBase):
    """A bivariate Gaussian profile.

    Attributes
    ----------
    fwhm : tuple[float, float]
        Full width at half maximum in x and y, in micrometres. For a
        horizontal goniometer x is vertical and y is horizontal.
    """

    type: Literal["gaussian"] = "gaussian"
    fwhm: tuple[float, float]


class TophatBeam(_BeamBase):
    """A uniform beam filling its collimation aperture.

    The aperture *is* the beam, so a tophat must have one; RADDOSE-3D's
    ``BeamTophat`` refuses to construct without it.
    """

    type: Literal["tophat"] = "tophat"

    @model_validator(mode="after")
    def _needs_an_aperture(self) -> Self:
        if self.collimation is None:
            raise ValueError(
                "a tophat beam fills its collimation aperture, so it needs one"
            )
        return self


Beam = Annotated[GaussianBeam | TophatBeam, Field(discriminator="type")]
