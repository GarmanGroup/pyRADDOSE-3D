"""Beam profiles: how much energy falls where.

Ports ``BeamGaussian.java``, ``BeamTophat.java`` and the measured profiles of
``BeamExperimental.java`` and ``BeamExperimentalpgm.java``. A beam answers one
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
from pathlib import Path
from typing import Any

import numpy as np

from pyraddose_3d.backend._array import (
    array_namespace,
    astype,
    erf,
    is_torch,
    promote,
    to_namespace,
)

__all__ = [
    "ExperimentalBeam",
    "GaussianBeam",
    "TophatBeam",
    "normal_cdf",
    "pink_beam_energies",
    "read_pgm",
]

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


def read_pgm(path: str | Path) -> np.ndarray:
    """Read a beam image from a plain PGM file, as ``BeamExperimentalpgm``.

    The Java discards the first line, skips any lines that start with a
    ``#`` comment, reads the width, the height and the maximum value, and
    then the pixels row by row from the top. It stores them reversed in both
    directions and transposed: the result's first index runs along y, from
    the image's right-hand column, and its second along x, from its bottom
    row. That is the beam's frame, with the image's horizontal axis along
    the goniometer.

    Parameters
    ----------
    path : str | Path
        A plain (``P2``) PGM file: the pixel values written as text.

    Returns
    -------
    np.ndarray
        ``(width, height)`` relative intensities, as the Java's
        ``dataStructure``.

    Raises
    ------
    ValueError
        If the file is not a plain PGM, or has fewer pixels than it says.
        RADDOSE-3D does not check the magic number, and fails on a binary
        (``P5``) file part way through reading it.
    """
    text = Path(path).read_text("latin-1")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines[0].split() or lines[0].split()[0] != "P2":
        raise ValueError(
            f"{path} is not a plain (P2) PGM file, the only kind RADDOSE-3D reads"
        )
    # Scanner.hasNext("#.*") then nextLine(): each blank or comment line goes.
    start = 1
    while start < len(lines) and (
        not lines[start].split() or lines[start].split()[0].startswith("#")
    ):
        start += 1
    tokens = " ".join(lines[start:]).split()
    try:
        width, height = int(tokens[0]), int(tokens[1])
        int(tokens[2])  # the maximum value, unused
        values = [float(t) for t in tokens[3 : 3 + width * height]]
    except (IndexError, ValueError):
        raise ValueError(f"{path} does not hold a PGM header and pixels") from None
    if len(values) < width * height:
        raise ValueError(
            f"{path} has {len(values)} pixels where its header says {width} x {height}"
        )
    image = np.array(values, dtype=np.float64).reshape(height, width)
    return np.ascontiguousarray(image[::-1, ::-1].T)


def _numpy(value: Array) -> np.ndarray:
    """A value as a NumPy array, for indexing rather than arithmetic."""
    if is_torch(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class ExperimentalBeam:
    """A measured beam profile, interpolated bilinearly between its pixels.

    Ports ``BeamExperimental``. The image is given a border of zeros one
    pixel wide, normalised so that it holds the beam's energy flux, and read
    by bilinear interpolation between pixel centres. Outside the image less
    its outermost pixel the intensity is zero.

    An aperture clips the profile with its own sizes, and the flux -- the
    flux after collimation -- is spread over the pixels whose centres it
    lets through.

    Attributes
    ----------
    flux : Array
        Photons per second, after any container.
    energy : Array
        Photon energy in keV.
    profile : np.ndarray
        Relative intensities, ``[y, x]``, as :func:`read_pgm` returns them.
    pixel_x, pixel_y : float
        Pixel size in micrometres.
    collimation_x, collimation_y : float | None
        The aperture's full widths in micrometres, or ``None`` for none.
    circular : bool
        Whether the aperture is elliptical rather than rectangular.
    """

    flux: Array
    energy: Array
    profile: np.ndarray
    pixel_x: float
    pixel_y: float
    collimation_x: float | None = None
    collimation_y: float | None = None
    circular: bool = False

    def _inside_aperture(self, x: Any, y: Any) -> Any:
        """``insideAperture``: whether points pass the aperture."""
        if self.collimation_x is None or self.collimation_y is None:
            return np.ones(np.shape(x), dtype=bool) if np.ndim(x) else True
        half_x, half_y = self.collimation_x / 2, self.collimation_y / 2
        if self.circular:
            return (x * x) / (half_x * half_x) + (y * y) / (half_y * half_y) <= 1
        return (abs(x) <= half_x) & (abs(y) <= half_y)

    @property
    def width_x(self) -> float:
        """``beamXSize``: the image, its border included, in micrometres."""
        return float((self.profile.shape[1] + 2) * self.pixel_x)

    @property
    def width_y(self) -> float:
        """``beamYSize``."""
        return float((self.profile.shape[0] + 2) * self.pixel_y)

    def intensity(self, x: Array, y: Array, off_axis: Array = 0.0) -> Array:
        """Energy flux at a point in the beam's plane.

        Parameters
        ----------
        x, y : Array
            Position in micrometres, relative to the image's centre.
        off_axis : Array, optional
            Horizontal displacement of the sample from the beam axis.

        Returns
        -------
        Array
            Joules per second per square micrometre.
        """
        bordered = np.pad(self.profile, 1)
        # beamSum, accumulated in the Java's order over the pixels the
        # aperture lets through. Pixel [i, j] is centred at bordered index
        # (i + 1, j + 1).
        rows, columns = np.indices(self.profile.shape)
        centre_x = (columns + 1 + 0.5) * self.pixel_x - self.width_x / 2
        centre_y = (rows + 1 + 0.5) * self.pixel_y - self.width_y / 2
        admitted = self._inside_aperture(centre_x, centre_y)
        total = float(np.cumsum(np.where(admitted, self.profile, 0.0).ravel())[-1])
        if total <= 0:
            raise ValueError(
                "the experimental beam's collimation lets none of its image through"
            )
        xp, (x, y, off_axis, energy, flux) = promote(
            x, y, off_axis, self.energy, self.flux
        )
        shifted = x - off_axis
        reach_x = self.width_x / 2 - self.pixel_x
        reach_y = self.width_y / 2 - self.pixel_y
        inside = (xp.abs(shifted) <= reach_x) & (xp.abs(y) <= reach_y)
        if self.collimation_x is not None:
            inside = inside & to_namespace(
                xp, self._inside_aperture(_numpy(shifted), _numpy(y))
            )

        real_x = shifted + self.width_x / 2
        real_y = y + self.width_y / 2
        column = xp.floor(real_x / self.pixel_x - 0.5)
        row = xp.floor(real_y / self.pixel_y - 0.5)
        # Java narrows each fraction to a float before interpolating.
        fraction_x = _through_float32(xp, real_x / self.pixel_x - (column + 0.5))
        fraction_y = _through_float32(xp, real_y / self.pixel_y - (row + 0.5))

        # Points outside the image are masked below; clip them to any pixel.
        i = np.clip(_numpy(row).astype(np.int64), 0, bordered.shape[0] - 2)
        j = np.clip(_numpy(column).astype(np.int64), 0, bordered.shape[1] - 2)
        denominator = total * self.pixel_x * self.pixel_y

        def corner(di: int, dj: int) -> Array:
            value = to_namespace(xp, bordered[i + di, j + dj])
            return value * KEV_TO_JOULES * energy * flux / denominator

        value = (
            corner(0, 0) * (1 - fraction_x) * (1 - fraction_y)
            + corner(0, 1) * fraction_x * (1 - fraction_y)
            + corner(1, 0) * (1 - fraction_x) * fraction_y
            + corner(1, 1) * fraction_x * fraction_y
        )
        return xp.where(inside, value, 0.0)

    @property
    def minimum_dimension(self) -> float:
        """``beamMinumumDimension``: the image's smaller side."""
        return min(self.width_x, self.width_y)


def _through_float32(xp: Any, value: Array) -> Array:
    """``(float) value``, widened back: rounded to single precision."""
    return astype(astype(value, xp.float32), xp.float64)


#: ``NormalEnergyDistribution.FWHM_TO_SIGMA``.
_FWHM_TO_SIGMA = 1 / (2 * math.sqrt(2 * math.log(2)))
#: ``Crystal.energiesPerAngle``: how many energies a pink beam is sampled at.
PINK_BEAM_SAMPLES = 200


def pink_beam_energies(
    mean: Array, fwhm: float, samples: int = PINK_BEAM_SAMPLES
) -> Array:
    """The energies RADDOSE-3D samples a pink beam at.

    Ports ``SampleNormalEnergyDistribution``: a Gaussian of the given
    full width at half maximum, sampled at evenly spaced cumulative
    probabilities ``i / (n - 1)``, keeping those within 40 standard
    deviations of the mean. The two ends are infinite and always dropped,
    so the probabilities are ``i / (samples + 1)`` for ``i = 1 .. samples``.

    Parameters
    ----------
    mean : Array
        Mean photon energy, keV. A tensor carries a gradient through.
    fwhm : float
        Full width at half maximum, keV.
    samples : int
        How many energies.

    Returns
    -------
    Array
        ``(samples,)`` energies in keV, ascending.
    """
    from scipy.special import erfinv

    sigma = _FWHM_TO_SIGMA * fwhm
    # Grow the grid until exactly `samples` points survive the 40-sigma cut,
    # as the Java does; both infinite ends always fail it.
    length = samples
    while True:
        p = np.arange(length) / (length - 1)
        with np.errstate(divide="ignore"):
            z = math.sqrt(2) * erfinv(2 * p - 1)
        kept = np.abs(sigma * z) <= 40 * sigma
        if int(kept.sum()) == samples:
            break
        length += 1
    xp = array_namespace(mean)
    offsets = sigma * z[kept]
    if xp is not np:
        offsets = xp.asarray(offsets, dtype=mean.dtype)
    return mean + offsets
