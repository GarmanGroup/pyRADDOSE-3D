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
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr, ndtri

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.coefficients import material_coefficients
from pyraddose_3d.backend.elements import read_table

if TYPE_CHECKING:
    from pyraddose_3d.backend.composition import Composition

__all__ = [
    "OUTER_SHELL",
    "BeamShape",
    "PhotonCoefficients",
    "Shells",
    "about_axis",
    "beam_positions",
    "choose",
    "compton_electrons",
    "cos_angle_to_x",
    "ionised_shells",
    "isotropic_directions",
    "klein_nishina_cosines",
    "photoelectron_directions",
    "photon_coefficients",
    "shell_cdf",
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

        An ellipse's is pi times its half-widths; RADDOSE-3D took its full
        widths, four times the area, until #45.
        """
        x, y = self.collimation[0] * 1000, self.collimation[1] * 1000
        return math.pi * (x / 2) * (y / 2) if self.circular else x * y

    def exposes(self, x_nm: np.ndarray, y_nm: np.ndarray) -> np.ndarray:
        """``testIfInsideExposedArea``: points within the aperture.

        A point is outside only if a comparison says so, as in the Java: so an
        undefined one, where an electron's last step was drawn from a
        negative mean free path, is inside.
        """
        half_x, half_y = self.collimation[0] / 2, self.collimation[1] / 2
        if self.circular:
            return ~(
                (np.power(x_nm / 1000, 2) / math.pow(half_x, 2))
                + (np.power(y_nm / 1000, 2) / math.pow(half_y, 2))
                > 1
            )
        return ~(np.abs(x_nm / 1000) > half_x) & ~(np.abs(y_nm / 1000) > half_y)


def beam_positions(
    rng: np.random.Generator, beam: BeamShape, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """``getPhotonBeamXYPos``: where each photon crosses the crystal's plane, nm.

    Uniform over a top-hat's aperture; a Gaussian's two normals each
    truncated to the aperture's extent. Within an elliptical aperture, a
    point is drawn again until it falls inside, as RADDOSE-3D draws it since
    #45: a circular top-hat filled a bow tie, and a circular Gaussian drew
    its radius from a one-dimensional normal.
    """
    cx, cy = beam.collimation
    x = np.empty(n)
    y = np.empty(n)
    pending = np.arange(n)
    while len(pending):
        m = len(pending)
        if beam.kind == "tophat":
            tx = 1000 * cx * (rng.random(m) - 0.5)
            ty = 1000 * ((rng.random(m) * cy) - (cy / 2))
        else:
            tx = _truncated_normal(rng, beam.sigma[0], cx, m) * 1000
            ty = _truncated_normal(rng, beam.sigma[1], cy, m) * 1000
        keep = beam.exposes(tx, ty) if beam.circular else np.ones(m, dtype=bool)
        x[pending[keep]] = tx[keep]
        y[pending[keep]] = ty[keep]
        pending = pending[~keep]
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


#: ``MC.OUTER_SHELL``: a photoelectric absorption by a shell the model leaves
#: out, an outer one. It has no binding energy and does not relax.
OUTER_SHELL = -1


def shell_cdf(z: int, energy: np.ndarray) -> np.ndarray:
    """``getRelativeShellProbs``: each shell's running chance, K first.

    ``(energies, 9)``. A shell beyond the photon's energy, an L shell of an
    element lighter than magnesium, or an M shell of one lighter than
    tantalum keeps 0. A draw past every running chance ionises an outer
    shell (:data:`OUTER_SHELL`).
    """
    s = shells(z)
    energy = np.atleast_1d(np.asarray(energy, dtype=float))
    cdf = np.zeros((len(energy), _SHELLS))
    running = np.zeros(len(energy))
    taken = np.zeros(len(energy))
    for shell in range(_SHELLS):
        allowed = energy > s.edges[shell]
        if shell in (1, 2, 3):
            allowed &= z >= 12
        elif shell >= 4:
            allowed &= z >= 73
        p = s.ionisation[shell] if shell == 0 else s.ionisation[shell] * (1 - taken)
        p = np.where(allowed, p, 0.0)
        taken += p
        running += p
        cdf[:, shell] = np.where(allowed, running, 0.0)
    return cdf


def ionised_shells(cdf: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """``MC.ionisedShell``: the first shell whose running chance passes.

    :data:`OUTER_SHELL` for a draw past them all.
    """
    above = cdf > uniform[:, None]
    return np.where(above.any(axis=1), above.argmax(axis=1), OUTER_SHELL)


@dataclass(frozen=True)
class PhotonCoefficients:
    """What ``updateCoefficients`` and its neighbours give, photon by photon.

    Attributes
    ----------
    absorption, compton, elastic : numpy.ndarray
        The material's photoelectric, Compton and coherent coefficients at
        each photon's energy, /um.
    photoelectric_cdf, compton_cdf : numpy.ndarray
        ``(photons, elements)``: each element's running share of the
        photoelectric and Compton coefficients, in atomic-number order.
    """

    absorption: np.ndarray
    compton: np.ndarray
    elastic: np.ndarray
    photoelectric_cdf: np.ndarray
    compton_cdf: np.ndarray

    @property
    def total_mean_free_path_nm(self) -> np.ndarray:
        """``totalMFPL``: one over every coefficient, nm."""
        return (1 / (self.absorption + self.compton + self.elastic)) * 1000

    @property
    def photon_mean_free_path_nm(self) -> np.ndarray:
        """``photonMFPL``: photoelectric and Compton alone, nm.

        The surrounding's is taken this way.
        """
        return (1 / (self.absorption + self.compton)) * 1000

    @property
    def elastic_chance(self) -> np.ndarray:
        """``MC.elasticChance``: coherent over all three, as since RADDOSE-3D #48."""
        return self.elastic / (self.absorption + self.compton + self.elastic)

    @property
    def compton_chance(self) -> np.ndarray:
        """``probCompton``."""
        return 1 - (self.photon_mean_free_path_nm / ((1 / self.absorption) * 1000))

    def take(self, index: np.ndarray) -> PhotonCoefficients:
        """The coefficients of the chosen photons."""
        return PhotonCoefficients(
            self.absorption[index],
            self.compton[index],
            self.elastic[index],
            self.photoelectric_cdf[index],
            self.compton_cdf[index],
        )


def photon_coefficients(
    composition: Composition, energy: np.ndarray
) -> PhotonCoefficients:
    """A material's photon coefficients and element shares at each energy, keV.

    Computed once for each distinct energy: a monochromatic beam's photons
    share one.
    """
    distinct, inverse = np.unique(np.asarray(energy, dtype=float), return_inverse=True)
    totals = material_coefficients(composition, distinct)
    counts = np.asarray(composition.counts, dtype=float)
    photo, compton = [], []
    for element, count in zip(composition.elements, counts, strict=True):
        cs = element.absorption_coefficients(distinct)
        photo.append(count * np.asarray(cs.photoelectric, dtype=float))
        compton.append(count * np.asarray(cs.compton, dtype=float))
    photo_terms = np.stack(photo, axis=1)
    compton_terms = np.stack(compton, axis=1)
    photo_cdf = np.cumsum(photo_terms / photo_terms.sum(axis=1, keepdims=True), axis=1)
    compton_cdf = np.cumsum(
        compton_terms / compton_terms.sum(axis=1, keepdims=True), axis=1
    )
    return PhotonCoefficients(
        absorption=np.asarray(totals.absorption, dtype=float)[inverse],
        compton=np.asarray(totals.inelastic, dtype=float)[inverse],
        elastic=np.asarray(totals.elastic, dtype=float)[inverse],
        photoelectric_cdf=photo_cdf[inverse],
        compton_cdf=compton_cdf[inverse],
    )


def choose(cdf: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """``getIonisedElement``: per row, the first element whose share passes.

    The last, should rounding leave none.
    """
    above = cdf > uniform[:, None]
    return np.where(above.any(axis=1), above.argmax(axis=1), cdf.shape[1] - 1)


def _emission_cdf() -> np.ndarray:
    """``populateAngularEmissionProbs``: the dipole distribution, beta = 2.

    Per unit angle, so times the angle's sine, and normalised over the bins
    it accumulates, as RADDOSE-3D has it since #47.
    """

    def height(angle: float) -> float:
        return (
            (1 / (4 * math.pi))
            * (1 + (2 * 0.5 * (3 * math.pow(math.cos(angle), 2) - 1)))
            * math.sin(angle)
        )

    step = math.pi / _EMISSION_BINS
    heights = [height(step * i) for i in range(_EMISSION_BINS + 1)]
    areas = [((a + b) / 2) * step for a, b in pairwise(heights)]
    total = 0.0
    for area in areas:
        total += area
    cdf = np.zeros(_EMISSION_BINS)
    running = 0.0
    for i, area in enumerate(areas):
        running += area / total
        cdf[i] = running
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


def isotropic_directions(rng: np.random.Generator, n: int) -> np.ndarray:
    """``MC.isotropicDirection``: ``(n, 3)`` unit vectors uniform over the sphere.

    The cosine of the polar angle is uniform on [-1, 1].
    """
    cos_theta = 1 - 2 * rng.random(n)
    phi = 2 * math.pi * rng.random(n)
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=1
    )


def about_axis(
    rng: np.random.Generator, cos_to_axis: np.ndarray, axis: int
) -> np.ndarray:
    """``MC.aboutAxis``: ``(N, 3)`` directions around the x (0) or y (1) axis.

    Each at its angle to the axis, uniformly around it.
    """
    around = 2 * math.pi * rng.random(len(cos_to_axis))
    sin = np.sqrt(1 - cos_to_axis * cos_to_axis)
    a, b = sin * np.cos(around), sin * np.sin(around)
    if axis == 0:
        return np.stack([cos_to_axis, a, b], axis=1)
    return np.stack([a, cos_to_axis, b], axis=1)


def photoelectron_directions(
    rng: np.random.Generator, k_shell: np.ndarray, rotated_polarisation: bool
) -> np.ndarray:
    """``producePhotoElectron``'s direction for each photoelectron, ``(N, 3)``.

    From the K shell, at an angle to the polarisation drawn from the dipole
    distribution, uniformly around it; otherwise uniformly over the sphere.
    The polarisation is the beam's, and does not turn with the crystal.
    ``rotated_polarisation`` is whether the goniometer and the polarisation
    are at right angles: the polarisation is along x if so, along y if not.
    As RADDOSE-3D has it since #47, and every K-shell photoelectron since
    #59: three in four did, but in XFEL.
    """
    n = len(k_shell)
    polarised = np.asarray(k_shell, dtype=bool)
    direction = np.empty((n, 3))
    m = int(polarised.sum())
    if m:
        direction[polarised] = about_axis(
            rng, cos_angle_to_x(rng, m), 0 if rotated_polarisation else 1
        )
    direction[~polarised] = isotropic_directions(rng, n - m)
    return direction


def klein_nishina_cosines(rng: np.random.Generator, k: np.ndarray) -> np.ndarray:
    """``MC.kleinNishinaCosine``: each Compton-scattered photon's cosine.

    Drawn from Klein and Nishina's cross-section by rejection.

    ``k`` is each photon's energy over the electron's rest energy.
    """
    mu = np.empty(len(k))
    pending = np.arange(len(k))
    while len(pending):
        trial = 1 - 2 * rng.random(len(pending))
        p = 1 / (1 + k[pending] * (1 - trial))
        f = p * p * (p + 1 / p - (1 - trial * trial)) / 2
        accept = rng.random(len(pending)) <= f
        mu[pending[accept]] = trial[accept]
        pending = pending[~accept]
    return mu


def compton_electrons(
    rng: np.random.Generator, photon_energy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``produceCompton``: each recoil electron's energy (keV) and direction.

    The photon's angle is drawn by Klein and Nishina, and the electron leaves
    at its angle to the beam uniformly around it, as RADDOSE-3D has it since
    #47.
    """
    photon_energy = np.asarray(photon_energy, dtype=float)
    n = len(photon_energy)
    mc_squared = _ELECTRON_MASS * math.pow(_SPEED_OF_LIGHT, 2)
    incident = photon_energy * KEV_TO_JOULES
    photon_theta = np.arccos(klein_nishina_cosines(rng, incident / mc_squared))
    energy = (
        (np.power(incident, 2) * (1 - np.cos(photon_theta)))
        / (mc_squared * (1 + ((incident / mc_squared) * (1 - np.cos(photon_theta)))))
    ) / KEV_TO_JOULES
    electron_phi = np.arctan(
        (1 / np.tan(photon_theta / 2)) / (1 + (incident / mc_squared))
    )
    around = 2 * math.pi * rng.random(n)
    x = np.sin(electron_phi) * np.cos(around)
    y = np.sin(electron_phi) * np.sin(around)
    return energy, np.stack([x, y, np.cos(electron_phi)], axis=1)
