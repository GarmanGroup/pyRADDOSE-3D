"""Beam profiles: how much energy falls where.

Ports ``BeamGaussian.java`` and ``BeamTophat.java``. A beam answers one
question for the exposure loop -- given a point in the plane perpendicular to
the beam, what is the energy flux there -- and two supporting ones about its
own size.

Units
-----
Positions and beam dimensions in micrometres, energy in keV, flux in photons
per second, and the returned intensity in joules per second per square
micrometre.

Differentiability
-----------------
Intensity is differentiable in flux, energy and the beam widths. It is *not*
differentiable across a collimation edge, where it steps to zero; that is a
real discontinuity in the model rather than an artefact, and the gradient
either side of it is correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pyraddose_3d.backend._array import erf, promote

__all__ = ["GaussianBeam", "TophatBeam", "normal_cdf"]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""

#: Elementary charge in coulombs, so keV becomes joules.
ELEMENTARY_CHARGE = 1.602176565e-19
KEV_TO_JOULES = 1000 * ELEMENTARY_CHARGE

#: Full width at half maximum over standard deviation. The Java calls this
#: ``SIGMA_TO_FWHM`` but divides by it, so the name is backwards there.
FWHM_TO_SIGMA = 2 * math.sqrt(2 * math.log(2))

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_PI = math.sqrt(2.0 * math.pi)

#: Steps in the midpoint rule used for an elliptical aperture. Fixed at the
#: Java's value: the result is an approximation, so matching it means
#: matching the step count too.
_ELLIPSE_QUADRATURE_STEPS = 1000


def normal_cdf(x: Array, sigma: Array) -> Array:
    """Cumulative probability of a zero-mean normal distribution.

    Parameters
    ----------
    x : Array
        Position.
    sigma : Array
        Standard deviation.

    Returns
    -------
    Array
        ``P(X <= x)``.
    """
    return 0.5 * (1.0 + erf(x / (sigma * _SQRT_2)))


def _normal_pdf(x: Array, sigma: Array) -> Array:
    """Density of a zero-mean normal distribution."""
    xp, (x, sigma) = promote(x, sigma)
    return xp.exp(-(x**2) / (2.0 * sigma**2)) / (sigma * _SQRT_2_PI)


@dataclass(frozen=True)
class GaussianBeam:
    """A beam with a bivariate Gaussian profile.

    Attributes
    ----------
    flux : Array
        Photons per second, before any container attenuation.
    energy : Array
        Photon energy in keV.
    fwhm_x, fwhm_y : Array
        Full width at half maximum in micrometres.
    collimation_x, collimation_y : Array | None
        Aperture size in micrometres. ``None`` for an uncollimated beam,
        which is then normalised over the whole plane.
    circular : bool
        Whether the aperture is elliptical rather than rectangular.
    """

    flux: Array
    energy: Array
    fwhm_x: Array
    fwhm_y: Array
    collimation_x: Array | None = None
    collimation_y: Array | None = None
    circular: bool = False

    @property
    def sigma_x(self) -> Array:
        """Horizontal standard deviation in micrometres."""
        return self.fwhm_x / FWHM_TO_SIGMA

    @property
    def sigma_y(self) -> Array:
        """Vertical standard deviation in micrometres."""
        return self.fwhm_y / FWHM_TO_SIGMA

    @property
    def collimated_fraction(self) -> Array:
        """Fraction of the beam that passes the aperture.

        The Java calls this ``normFactor``. Dividing by it conserves flux:
        the photons blocked by the aperture are not lost from the total, they
        are concentrated into what gets through.

        Returns
        -------
        Array
            A number in (0, 1], or exactly 1 for an uncollimated beam.
        """
        if self.collimation_x is None or self.collimation_y is None:
            return 1.0

        half_x = self.collimation_x / 2
        half_y = self.collimation_y / 2
        if not self.circular:
            # Written as the Java's four CDF terms rather than the
            # algebraically equal product of two differences, so the
            # rounding matches.
            upper_x = normal_cdf(half_x, self.sigma_x)
            lower_x = normal_cdf(-half_x, self.sigma_x)
            upper_y = normal_cdf(half_y, self.sigma_y)
            lower_y = normal_cdf(-half_y, self.sigma_y)
            return (
                upper_x * upper_y
                - lower_x * upper_y
                - upper_x * lower_y
                + lower_x * lower_y
            )
        return self._elliptical_fraction(half_x, half_y)

    def _elliptical_fraction(self, half_x: Array, half_y: Array) -> Array:
        """Fraction passing an elliptical aperture, by midpoint quadrature.

        Exact in y via the CDF, approximate in x. An ellipse has no
        closed-form bivariate normal integral, so the Java integrates
        numerically and this must match it step for step.
        """
        # sigma comes in too: the aperture may be plain floats while the
        # beam width is a tensor, and the quadrature mixes the two.
        xp, (half_x, half_y, sigma_x, sigma_y) = promote(
            half_x, half_y, self.sigma_x, self.sigma_y
        )
        steps = _ELLIPSE_QUADRATURE_STEPS
        width = (2.0 * half_x) / steps
        # Midpoints 0.5, 1.5, ... are exact in any float type, so the
        # integer arange costs no precision once scaled by width.
        x = -half_x + (xp.arange(steps) + 0.5) * width
        # Half-height of the ellipse at each x, clamped so that rounding at
        # the extremes cannot put a negative under the root.
        y_bound = half_y * xp.sqrt(
            xp.clip(1.0 - (x * x) / (half_x * half_x), 0.0, None)
        )
        strip = _normal_pdf(x, sigma_x) * (
            normal_cdf(y_bound, sigma_y) - normal_cdf(-y_bound, sigma_y)
        )
        return xp.sum(strip) * width

    def _inside_aperture(
        self, x: Array, y: Array, half_x: Array, half_y: Array
    ) -> Array:
        """Whether a point passes the aperture.

        Takes the aperture half-widths rather than reading them from the
        beam, because it is only meaningful when there is an aperture and
        the caller has already established that.
        """
        xp, (x, y, half_x, half_y) = promote(x, y, half_x, half_y)
        if self.circular:
            return (x / half_x) ** 2 + (y / half_y) ** 2 <= 1
        return (xp.abs(x) <= half_x) & (xp.abs(y) <= half_y)

    def intensity(self, x: Array, y: Array, off_axis: Array = 0.0) -> Array:
        """Energy flux at a point in the beam's plane.

        Parameters
        ----------
        x, y : Array
            Position in micrometres, relative to the beam centre.
        off_axis : Array, optional
            Horizontal displacement of the sample from the beam axis, in
            micrometres. Subtracted from ``x``, matching the Java.

        Returns
        -------
        Array
            Joules per second per square micrometre. Zero outside the
            aperture.
        """
        # The beam's own parameters are promoted alongside the positions:
        # a tensor flux with float coordinates would otherwise leave the
        # profile in NumPy and the scale in torch.
        scale = KEV_TO_JOULES * self.energy * self.flux / self.collimated_fraction
        xp, (x, y, off_axis, sigma_x, sigma_y, scale) = promote(
            x, y, off_axis, self.sigma_x, self.sigma_y, scale
        )
        shifted = x - off_axis
        profile = _normal_pdf(shifted, sigma_x) * _normal_pdf(y, sigma_y)
        if self.collimation_x is None or self.collimation_y is None:
            return profile * scale
        inside = self._inside_aperture(
            shifted, y, self.collimation_x / 2, self.collimation_y / 2
        )
        return xp.where(inside, profile * scale, 0.0)

    @property
    def minimum_dimension(self) -> Array:
        """Smallest length scale in the beam, for choosing a voxel size.

        Returns
        -------
        Array
            The smaller of the two FWHMs, further limited by the aperture
            when there is one.
        """
        xp, (fwhm_x, fwhm_y) = promote(self.fwhm_x, self.fwhm_y)
        smallest_fwhm = xp.minimum(fwhm_x, fwhm_y)
        if self.collimation_x is None or self.collimation_y is None:
            return smallest_fwhm
        _, (coll_x, coll_y) = promote(self.collimation_x, self.collimation_y)
        return xp.minimum(xp.minimum(coll_x, coll_y), smallest_fwhm)


@dataclass(frozen=True)
class TophatBeam:
    """A beam of uniform intensity within its footprint and zero outside.

    Attributes
    ----------
    flux : Array
        Photons per second.
    energy : Array
        Photon energy in keV.
    width_x, width_y : Array
        Footprint in micrometres.
    circular : bool
        Whether the footprint is elliptical rather than rectangular.
    """

    flux: Array
    energy: Array
    width_x: Array
    width_y: Array
    circular: bool = False

    @property
    def area(self) -> Array:
        """Illuminated area in square micrometres."""
        if self.circular:
            return math.pi * (self.width_x / 2) * (self.width_y / 2)
        return self.width_x * self.width_y

    def intensity(self, x: Array, y: Array, off_axis: Array = 0.0) -> Array:
        """Energy flux at a point in the beam's plane.

        Parameters
        ----------
        x, y : Array
            Position in micrometres, relative to the beam centre.
        off_axis : Array, optional
            Horizontal displacement of the sample from the beam axis.

        Returns
        -------
        Array
            Joules per second per square micrometre, uniform inside the
            footprint and zero outside.
        """
        flux = KEV_TO_JOULES * self.flux * self.energy / self.area
        xp, (x, y, off_axis, half_x, half_y, flux) = promote(
            x, y, off_axis, self.width_x / 2, self.width_y / 2, flux
        )
        shifted = x - off_axis
        if self.circular:
            inside = (shifted / half_x) ** 2 + (y / half_y) ** 2 <= 1
        else:
            inside = (xp.abs(shifted) <= half_x) & (xp.abs(y) <= half_y)
        return xp.where(inside, flux, 0.0)

    @property
    def minimum_dimension(self) -> Array:
        """Smallest length scale in the beam, for choosing a voxel size."""
        xp, (width_x, width_y) = promote(self.width_x, self.width_y)
        return xp.minimum(width_x, width_y)
