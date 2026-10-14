"""Photons: where they enter, whether they interact, and what they free.

Ports the photon half of ``MC.startMonteCarloXFEL``, ``producePhotoElectron``
and ``produceCompton`` for photons in a crystal with no surrounding, every
photon of a run at once. The Java draws for one photon at a time; the draws
here are the same distributions, made for all photons together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr, ndtri

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.coefficients import material_coefficients
from pyraddose_3d.backend.elements import read_table

if TYPE_CHECKING:
    from pyraddose_3d.backend.composition import Composition

__all__ = [
    "BeamShape",
    "PhotonCoefficients",
    "Shells",
    "beam_positions",
    "compton_electrons",
    "cos_angle_to_x",
    "photon_coefficients",
    "shells",
]

_ELECTRON_MASS = 9.10938356e-31
_SPEED_OF_LIGHT = 299792458.0
#: ``2 sqrt(2 ln 2)``: a Gaussian's FWHM over its standard deviation.
_SIGMA_TO_FWHM = 2 * math.sqrt(2 * math.log(2))
#: ``MC.numberAngularEmissionBins``.
_EMISSION_BINS = 50
#: Shells in RADDOSE-3D's photoelectric model: K, L1-L3, M1-M5.
_SHELLS = 9


@dataclass(frozen=True)
class BeamShape:
    """The beam as ``getPhotonBeamXYPos`` and ``getExposedArea`` read it.

    Attributes
    ----------
    kind : {"gaussian", "tophat"}
        The profile.
    collimation : tuple[float, float]
        Its aperture in x and y, um. A Gaussian beam with none is collimated
        at three FWHM.
    circular : bool
        Whether the aperture is an ellipse rather than a rectangle.
    sigma : tuple[float, float]
        A Gaussian beam's standard deviations in x and y, um.
    """

    kind: str
    collimation: tuple[float, float]
    circular: bool
    sigma: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_model(cls, beam: object) -> BeamShape:
        """From a ``GaussianBeam`` or ``TophatBeam`` model."""
        collimation = getattr(beam, "collimation", None)
        circular = collimation is not None and collimation.shape == "circular"
        if getattr(beam, "type", None) == "gaussian":
            fwhm = beam.fwhm  # type: ignore[attr-defined]
            sigma = (fwhm[0] / _SIGMA_TO_FWHM, fwhm[1] / _SIGMA_TO_FWHM)
            if collimation is None:
                size = (sigma[0] * _SIGMA_TO_FWHM * 3, sigma[1] * _SIGMA_TO_FWHM * 3)
            else:
                size = (collimation.x, collimation.y)
            return cls("gaussian", size, circular, sigma)
        assert collimation is not None
        return cls("tophat", (collimation.x, collimation.y), circular)

    def exposed_area_nm2(self) -> float:
        """``getExposedArea``: the aperture's area, nm^2.

        An ellipse's is taken as pi times its full widths, four times its
        area; reproduced.
        """
        x, y = self.collimation[0] * 1000, self.collimation[1] * 1000
        return math.pi * x * y if self.circular else x * y

    def exposes(self, x_nm: np.ndarray, y_nm: np.ndarray) -> np.ndarray:
        """``testIfInsideExposedArea``: points within the aperture."""
        half_x, half_y = self.collimation[0] / 2, self.collimation[1] / 2
        if self.circular:
            return (np.power(x_nm / 1000, 2) / math.pow(half_x, 2)) + (
                np.power(y_nm / 1000, 2) / math.pow(half_y, 2)
            ) <= 1
        return (np.abs(x_nm / 1000) <= half_x) & (np.abs(y_nm / 1000) <= half_y)


def _plus_or_minus(rng: np.random.Generator, n: int) -> np.ndarray:
    """``PosOrNeg``."""
    return np.where(rng.random(n) < 0.5, 1.0, -1.0)


def beam_positions(
    rng: np.random.Generator, beam: BeamShape, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """``getPhotonBeamXYPos``: where each photon crosses the crystal's plane, nm."""
    cx, cy = beam.collimation
    if beam.kind == "tophat":
        x = 1000 * cx * (rng.random(n) - 0.5)
        u = rng.random(n)
        if beam.circular:
            # Bounded by the line from the centre to the aperture's corner,
            # not by the ellipse; reproduced.
            y = (
                1000
                * u
                * np.sqrt(
                    math.pow(cy / 2, 2) * (np.power(x / 1000, 2) / math.pow(cx / 2, 2))
                )
            )
            y = y * _plus_or_minus(rng, n)
        else:
            y = 1000 * ((u * cy) - (cy / 2))
        return x, y

    sx, sy = beam.sigma
    if not beam.circular:
        return (
            _truncated_normal(rng, sx, cx, n) * 1000,
            _truncated_normal(rng, sy, cy, n) * 1000,
        )
    if sx == sy:
        rng.random(n)  # RND1, drawn and never used
        theta = 2 * math.pi * rng.random(n)
        r = _truncated_normal(rng, sx, cx, n) * 1000
        return r * np.cos(theta), r * np.sin(theta)
    x = np.empty(n)
    y = np.empty(n)
    pending = np.arange(n)
    while len(pending):
        tx = rng.standard_normal(len(pending)) * sx * 1000
        ty = rng.standard_normal(len(pending)) * sy * 1000
        accept = (np.power(tx / 1000, 2) / math.pow(cx / 2, 2)) + (
            np.power(ty / 1000, 2) / math.pow(cy / 2, 2)
        ) <= 1
        x[pending[accept]] = tx[accept]
        y[pending[accept]] = ty[accept]
        pending = pending[~accept]
    return x, y


def _truncated_normal(
    rng: np.random.Generator, sigma: float, aperture: float, n: int
) -> np.ndarray:
    """A normal of width ``sigma`` within +-aperture/2, by its inverse CDF, um."""
    lower = float(ndtr(-aperture / 2 / sigma))
    upper = float(ndtr(aperture / 2 / sigma))
    return ndtri(lower + rng.random(n) * (upper - lower)) * sigma


@dataclass(frozen=True)
class Shells:
    """One element's shells, K to M5, as the photoelectric model reads them.

    Attributes
    ----------
    edges : numpy.ndarray
        The energy above which each shell can be ionised, keV: the K, L1,
        L2, L3 and M1-M5 absorption edges. NaN where the table has none.
    binding : numpy.ndarray
        ``getShellBindingEnergy``: what an ionisation of each costs, keV --
        the edges, but L1's tabulated binding energy, and 0 for a missing M1.
    ionisation : numpy.ndarray
        ``EdgeRatio``: each shell's share of the absorption above its edge,
        ``1 - 1/ratio``, and 1 where the ratio is 0.
    fluorescence_yield : numpy.ndarray
        The chance a vacancy in each is filled with a photon, not an Auger
        electron.
    """

    edges: np.ndarray
    binding: np.ndarray
    ionisation: np.ndarray
    fluorescence_yield: np.ndarray


def _value(text: str) -> float:
    return float(text) if text else math.nan


@cache
def shells(z: int) -> Shells:
    """The shells of element ``z``."""
    edges_row = next(r for r in read_table("absorption_edges.tsv") if int(r["z"]) == z)
    ratios = next(r for r in read_table("edge_ratios.tsv") if int(r["z"]) == z)
    yields = next(r for r in read_table("fluorescence.tsv") if int(r["z"]) == z)
    electron = next(
        r for r in read_table("electron_ionisation.tsv") if int(r["z"]) == z
    )
    names = ("k", "l1", "l2", "l3", "m1", "m2", "m3", "m4", "m5")
    edges = np.array([_value(edges_row[f"{n}_kev"]) for n in names])
    binding = edges.copy()
    binding[1] = _value(electron["l1_binding_kev"])
    if math.isnan(binding[4]):
        binding[4] = 0.0  # getM1Edge
    with np.errstate(divide="ignore"):
        ratio = np.array([_value(ratios[n]) for n in names])
        ionisation = 1 - 1 / ratio
    ionisation[ionisation == -np.inf] = 1.0
    fluorescence = np.array([_value(yields[f"yield_{n}"]) for n in names])
    return Shells(edges, binding, ionisation, fluorescence)


def _shell_cdf(z: int, energy: float) -> np.ndarray:
    """``getRelativeShellProbs``: running chances of each shell, K first.

    A shell beyond the photon's energy, or an L shell of an element lighter
    than magnesium, or an M shell of one lighter than tantalum, keeps 0: the
    draw that falls past every running chance takes the K shell.
    """
    s = shells(z)
    cdf = np.zeros(_SHELLS)
    running = 0.0
    taken = 0.0
    for shell in range(_SHELLS):
        allowed = energy > s.edges[shell]
        if shell in (1, 2, 3):
            allowed = allowed and z >= 12
        elif shell >= 4:
            allowed = allowed and z >= 73
        if not allowed:
            continue
        p = s.ionisation[shell] if shell == 0 else s.ionisation[shell] * (1 - taken)
        taken += p
        running += p
        cdf[shell] = running
    return cdf


@dataclass(frozen=True)
class PhotonCoefficients:
    """What ``updateCoefficients`` and its neighbours give at one energy.

    Attributes
    ----------
    absorption, compton, elastic : float
        The crystal's photoelectric, Compton and coherent coefficients, /um.
    photoelectric_cdf, compton_cdf : numpy.ndarray
        Each element's running share of the photoelectric and Compton
        coefficients, in atomic-number order.
    shell_cdf : numpy.ndarray
        ``(elements, 9)``: :func:`_shell_cdf` for each element.
    """

    absorption: float
    compton: float
    elastic: float
    photoelectric_cdf: np.ndarray
    compton_cdf: np.ndarray
    shell_cdf: np.ndarray

    @property
    def total_mean_free_path_nm(self) -> float:
        """``totalMFPL``: one over every coefficient, nm."""
        return (1 / (self.absorption + self.compton + self.elastic)) * 1000

    @property
    def elastic_chance(self) -> float:
        """``elasticProb``: coherent over photoelectric and Compton alone."""
        return self.elastic / (self.absorption + self.compton)

    @property
    def compton_chance(self) -> float:
        """``probCompton``."""
        photon = (1 / (self.absorption + self.compton)) * 1000
        return 1 - (photon / ((1 / self.absorption) * 1000))


def photon_coefficients(composition: Composition, energy: float) -> PhotonCoefficients:
    """The crystal's photon coefficients and element shares at ``energy`` keV."""
    totals = material_coefficients(composition, energy)
    counts = np.asarray(composition.counts, dtype=float)
    photo, compton = [], []
    for element, count in zip(composition.elements, counts, strict=True):
        cs = element.absorption_coefficients(energy)
        photo.append(count * float(cs.photoelectric))
        compton.append(count * float(cs.compton))
    photo_share = np.array(photo) / sum(photo)
    compton_share = np.array(compton) / sum(compton)
    return PhotonCoefficients(
        absorption=float(totals.absorption),
        compton=float(totals.inelastic),
        elastic=float(totals.elastic),
        photoelectric_cdf=np.cumsum(photo_share),
        compton_cdf=np.cumsum(compton_share),
        shell_cdf=np.stack(
            [_shell_cdf(e.atomic_number, energy) for e in composition.elements]
        ),
    )


def first_above(cdf: np.ndarray, uniform: np.ndarray, default: int = 0) -> np.ndarray:
    """The first index whose running chance exceeds each draw, as the Java walks.

    ``default`` where none does.
    """
    above = cdf[None, :] > uniform[:, None]
    found = above.any(axis=1)
    return np.where(found, above.argmax(axis=1), default)


def _emission_cdf() -> np.ndarray:
    """``populateAngularEmissionProbs``: the dipole distribution, beta = 2."""

    def height(angle: float) -> float:
        return (1 / (4 * math.pi)) * (
            1 + (2 * 0.5 * (3 * math.pow(math.cos(angle), 2) - 1))
        )

    total = 0.0
    last = 0.0
    for i in range(101):
        h = height((math.pi / 100) * i)
        if i > 0:
            total += ((last + h) / 2) * (math.pi / 100)
        last = h
    cdf = np.zeros(_EMISSION_BINS)
    running = 0.0
    last = 0.0
    for i in range(_EMISSION_BINS + 1):
        h = height((math.pi / _EMISSION_BINS) * i)
        if i > 0:
            running += (((last + h) / 2) * (math.pi / _EMISSION_BINS)) / total
            cdf[i - 1] = running
        last = h
    return cdf


_EMISSION = _emission_cdf()


def cos_angle_to_x(rng: np.random.Generator, n: int) -> np.ndarray:
    """``getCosAngleToX``: the cosine of the emission angle to the polarisation."""
    u = rng.random(n)
    above = _EMISSION[None, :] > u[:, None]
    found = above.any(axis=1)
    bin_ = above.argmax(axis=1)
    last = np.where(bin_ > 0, _EMISSION[np.maximum(bin_ - 1, 0)], 0.0)
    step = math.pi / _EMISSION_BINS
    along = (u - last) / (_EMISSION[bin_] - last)
    angle = np.where(found, bin_ * step + along * step, 0.0)
    return np.cos(angle)


def photoelectron_directions(
    rng: np.random.Generator,
    k_shell: np.ndarray,
    angle: np.ndarray,
    rotated_polarisation: bool,
) -> np.ndarray:
    """``producePhotoElectron``'s direction for each photoelectron, ``(N, 3)``.

    From the K shell, three times in four, along the polarisation's dipole
    pattern, rotated with the crystal; otherwise "at random", with the polar
    angle drawn uniformly from 0 to 2 pi. ``rotated_polarisation`` is whether
    the goniometer and the polarisation are at right angles.
    """
    n = len(k_shell)
    polarised = (rng.random(n) > 0.25) & k_shell
    direction = np.empty((n, 3))

    m = int(polarised.sum())
    if m:
        towards = cos_angle_to_x(rng, m)
        across = (
            _plus_or_minus(rng, m) * rng.random(m) * np.sqrt(1 - np.power(towards, 2))
        )
        if rotated_polarisation:
            x0, y = towards, across
        else:
            y, x0 = towards, across
        z0 = _plus_or_minus(rng, m) * np.sqrt(1 - np.power(x0, 2) - np.power(y, 2))
        this_angle = 2 * math.pi - angle[polarised]
        direction[polarised, 0] = x0 * np.cos(this_angle) + z0 * np.sin(this_angle)
        direction[polarised, 1] = y
        direction[polarised, 2] = -1 * x0 * np.sin(this_angle) + z0 * np.cos(this_angle)

    rest = ~polarised
    r = int(rest.sum())
    theta = rng.random(r) * 2 * math.pi
    phi = rng.random(r) * 2 * math.pi
    direction[rest, 0] = np.sin(theta) * np.cos(phi)
    direction[rest, 1] = np.sin(theta) * np.sin(phi)
    direction[rest, 2] = np.cos(theta)
    return direction


def compton_electrons(
    rng: np.random.Generator, photon_energy: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """``produceCompton``: each recoil electron's energy (keV) and direction.

    The photon's angle is drawn uniformly from 0 to pi, not by Klein and
    Nishina; reproduced.
    """
    photon_theta = math.pi * rng.random(n)
    mc_squared = _ELECTRON_MASS * math.pow(_SPEED_OF_LIGHT, 2)
    incident = photon_energy * KEV_TO_JOULES
    energy = (
        (math.pow(incident, 2) * (1 - np.cos(photon_theta)))
        / (mc_squared * (1 + ((incident / mc_squared) * (1 - np.cos(photon_theta)))))
    ) / KEV_TO_JOULES
    electron_phi = np.arctan(
        (1 / np.tan(photon_theta / 2)) / (1 + (incident / mc_squared))
    )
    z = np.cos(electron_phi)
    x = _plus_or_minus(rng, n) * rng.random(n) * np.sqrt(1 - np.power(z, 2))
    y = _plus_or_minus(rng, n) * np.sqrt(1 - np.power(x, 2) - np.power(z, 2))
    return energy, np.stack([x, y, z], axis=1)
