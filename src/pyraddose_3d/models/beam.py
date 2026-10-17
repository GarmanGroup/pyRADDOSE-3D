"""Beam profile and collimation.

A field has a default only where RADDOSE-3D has one, and it is RADDOSE-3D's,
so that leaving a setting out of a YAML file means the same as leaving it out
of a ``.txt`` input. Flux, energy, a Gaussian's widths and an experimental
beam's image and pixel size are required by the Java and are required here.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from pyraddose_3d.models._base import RD3DModel

__all__ = [
    "Beam",
    "Collimation",
    "ElectronBeam",
    "ExperimentalBeam",
    "GaussianBeam",
    "TophatBeam",
    "XrayBeam",
]


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
    flux : float | None
        Photons per second. Every engine needs it but the XFEL subprogram,
        which takes ``pulse_energy`` instead.
    energy : float
        Photon energy in keV.
    collimation : Collimation | None
        The aperture, or ``None`` for an uncollimated beam -- which is what
        RADDOSE-3D does when an input file has no ``Collimation`` line.
    energy_fwhm : float | None
        ``EnergyFWHM``: a pink beam, Gaussian in energy with this full width
        at half maximum, in keV, about ``energy``. ``None`` is
        monochromatic.
    pulse_energy : float | None
        ``PulseEnergy``: the energy of one X-ray free-electron laser pulse,
        in mJ, which only the XFEL subprogram reads.
    pulse_shape : {"tophat", "gaussian"} | None
        ``PulseShape``: the XFEL pulse's shape in time. A top-hat, the
        default, spreads its photons evenly over the wedge's exposure time; a
        Gaussian's FWHM is the exposure time, and its photons all arrive
        within ``pulse_window``.
    pulse_window : float | None
        ``PulseWindow``: the width of the window a Gaussian pulse's photons
        arrive in, centred on its peak, in FWHMs; 2 if not given.
    """

    flux: Annotated[float, Field(gt=0.0)] | None = None
    energy: Annotated[float, Field(gt=0.0)]
    collimation: Collimation | None = None
    energy_fwhm: Annotated[float, Field(gt=0.0)] | None = None
    pulse_energy: Annotated[float, Field(gt=0.0)] | None = None
    pulse_shape: Literal["tophat", "gaussian"] | None = None
    pulse_window: Annotated[float, Field(gt=0.0)] | None = None

    @model_validator(mode="after")
    def _a_window_is_a_gaussians(self) -> Self:
        """Only a Gaussian pulse has a window; a top-hat lasts the exposure."""
        if self.pulse_window is not None and self.pulse_shape != "gaussian":
            raise ValueError(
                "pulse_window is the window a Gaussian pulse's photons arrive in, "
                "in FWHMs; set pulse_shape to gaussian, or leave it out"
            )
        return self

    @model_validator(mode="after")
    def _needs_photons(self) -> Self:
        """A flux, or for XFEL a pulse energy, as RADDOSE-3D's beams require."""
        if self.flux is None and self.pulse_energy is None:
            raise ValueError(
                "a beam needs its flux, or for the XFEL subprogram its pulse energy"
            )
        return self


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


class ExperimentalBeam(_BeamBase):
    """A measured beam profile, read from an image (``Type ExperimentalPGM``).

    The image is a plain (``P2``) PGM file whose pixel values are relative
    intensities; the flux sets their total. RADDOSE-3D lays the image's
    horizontal axis along y, the rotation axis, and its vertical axis along
    x, both reversed: a peak right of the image's centre is at negative y.

    Attributes
    ----------
    file : str
        Path to the PGM file, resolved relative to the working directory of
        the run.
    pixel_size : tuple[float, float]
        Pixel size in x and y, in micrometres: the first down the image, the
        second across it.
    collimation : Collimation | None
        An aperture, which clips the measured profile with its own sizes.
        The flux, the flux after collimation, is spread over the pixels it
        lets through. RADDOSE-3D used to ignore the sizes.
    """

    type: Literal["experimental"] = "experimental"
    file: str
    pixel_size: tuple[Annotated[float, Field(gt=0.0)], Annotated[float, Field(gt=0.0)]]


class ElectronBeam(RD3DModel):
    """An electron beam, for RADDOSE-3D's MicroED engine.

    In a ``.txt`` input, a ``Type TopHat`` beam with ``Exposure`` in an
    experiment whose crystal has ``Subprogram EMED``. RADDOSE-3D's MicroED
    refuses any other beam type: only a top-hat beam carries an exposure.

    Attributes
    ----------
    energy : float
        Electron energy in keV.
    exposure : float
        Fluence, in electrons per square Angstrom.
    collimation : Collimation
        The beam's size and shape. Where it is larger than the crystal, the
        crystal limits the exposed area, axis by axis.
    """

    type: Literal["electron"] = "electron"
    energy: Annotated[float, Field(gt=0.0)]
    exposure: Annotated[float, Field(gt=0.0)]
    collimation: Collimation


XrayBeam = Annotated[
    GaussianBeam | TophatBeam | ExperimentalBeam, Field(discriminator="type")
]
"""A beam for the MX calculation."""

Beam = Annotated[
    GaussianBeam | TophatBeam | ExperimentalBeam | ElectronBeam,
    Field(discriminator="type"),
]
"""Any beam: an X-ray beam, or an electron beam for MicroED."""
