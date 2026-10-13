"""Photoelectron and fluorescence escape, as RADDOSE-3D models them.

Ports the escape machinery of ``Crystal`` and ``CrystalPolyhedron``. When
a voxel absorbs a photon, not all of the energy stays there:

- **Photoelectron escape.** The photoelectron travels a few micrometres
  before it has deposited all its energy, so some of the voxel's dose lands
  in its neighbours, and from a voxel near the surface some leaves the
  crystal altogether. RADDOSE-3D picks one direction per voxel per image at
  random -- weighted towards the beam's polarisation -- and deposits the
  dose along it at distances drawn from a Gumbel distribution of path
  lengths, convolved with the energy-deposition profile along a track.
- **Fluorescence escape.** An atom ionised in an inner shell may relax by
  emitting a fluorescence photon, which can travel much further before it
  is reabsorbed. The rest of that energy is Auger electrons, which deposit
  locally.
- **The surrounding.** Photoelectrons produced in the liquid around the
  crystal can travel into it. RADDOSE-3D builds a second, coarser grid over
  a shell of surrounding and runs photoelectron escape from it.

Matching the Java's random draws
--------------------------------
All of this is deterministic except the directions, which are ``nextInt``
draws from a single ``java.util.Random``. The draws are taken here in
exactly the Java's order -- voxel by voxel in ``i, j, k`` order, the
fluorescence's direction before the photoelectron's, the crystal's voxels
before the surrounding's -- from :class:`~pyraddose_3d.backend.java_random.JavaRandom`
with the same seed, so the port reproduces RADDOSE-3D's escape results
rather than only their statistics. That holds while the same voxels are
lit in every image: a voxel exactly on a collimation edge, where the two
could disagree on whether the beam reaches it, would shift the stream, and
from then on the realisation would differ while staying statistically the
same.

Not differentiable: the directions are integer draws and the deposition
lands in whole voxels.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.backend.elements import Element, element_database, read_table

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyraddose_3d.backend.composition import Composition
    from pyraddose_3d.backend.java_random import JavaRandom

__all__ = [
    "PE_ANGLE_RESOLUTION",
    "Escape",
    "FluorescenceEscape",
    "PhotoelectronEscape",
    "Surrounding",
    "escape_factors",
    "fluorescence_escape",
    "photoelectron_escape",
    "round_half_up",
    "surrounding_composition",
    "surrounding_escape",
]

Array = Any

#: ``CrystalPolyhedron.PE_ANGLE_RESOLUTION``: tracks drawn per voxel per
#: image. RADDOSE-3D lets an environment variable change it; the default is 1.
PE_ANGLE_RESOLUTION = 1
#: Directions the photoelectron tracks are spread over, per hemisphere axis.
_PE_ANGLE_RES_LIMIT = 100
#: ``PE_ANGLE_LIMIT``: a full turn.
_PE_ANGLE_LIMIT = 1 * 2 * math.pi
#: ``FL_ANGLE_RESOLUTION`` and ``FL_ANGLE_RES_LIMIT``.
_FL_ANGLE_RES_LIMIT = 16
#: Default fluorescence distance bins, ``flDistBins``.
_FL_DISTANCE_BINS = 8
#: ``setUpPEPolarisation``: an s shell's asymmetry, and the beam's degree of
#: polarisation.
_BETA = 2.0
_DEGREE_OF_POLARISATION = 0.75
#: Columns of a row of :func:`escape_factors`, as ``getFluorescentEscapeFactors``.
FACTOR_COLUMNS = 27


# --- per-element shell data -------------------------------------------------


@dataclass(frozen=True)
class _Shells:
    """What escape needs of one element, from RADDOSE-3D's element table."""

    edges: tuple[float, ...]  # K, L1, L2, L3, M1, M2, M3, M4, M5 in keV
    ionisation: tuple[float, ...]  # same order, from the edge-jump ratios
    yields: tuple[float, float, float, float]  # K, L1, L2, L3
    average_k: float
    average_l: float


def _number(text: str) -> float:
    return float(text) if text else 0.0


@functools.cache
def _shells(atomic_number: int) -> _Shells:
    """``Element.EdgeRatio`` and the getters escape reads, for one element."""
    edges_row = next(
        r for r in read_table("absorption_edges.tsv") if int(r["z"]) == atomic_number
    )
    ratio_row = next(
        r for r in read_table("edge_ratios.tsv") if int(r["z"]) == atomic_number
    )
    fl_row = next(
        r for r in read_table("fluorescence.tsv") if int(r["z"]) == atomic_number
    )
    element = element_database().by_number(atomic_number)
    # K, L1, L2, L3 and M1 are the cross-section table's edges -- EDGE_K,
    # EDGE_L, L2, L3, EDGE_M -- and M2 to M5 the absorption-edge columns.
    edges = (
        element.edge_k,
        element.edge_l or 0.0,
        element.l2,
        element.l3,
        element.edge_m or 0.0,
        *(_number(edges_row[f"m{n}_kev"]) for n in (2, 3, 4, 5)),
    )
    ionisation = []
    for shell in ("k", "l1", "l2", "l3", "m1", "m2", "m3", "m4", "m5"):
        ratio = _number(ratio_row[shell])
        # 1 - 1/0 is -Infinity in Java; EdgeRatio then sets it to 1.
        ionisation.append(1.0 if ratio == 0 else 1 - 1 / ratio)
    return _Shells(
        edges=edges,
        ionisation=tuple(ionisation),
        yields=(
            _number(fl_row["yield_k"]),
            _number(fl_row["yield_l1"]),
            _number(fl_row["yield_l2"]),
            _number(fl_row["yield_l3"]),
        ),
        average_k=_number(fl_row["average_k_kev"]),
        average_l=_number(fl_row["average_l_kev"]),
    )


def _remaining(earlier: list[float]) -> float:
    """``1 - kFactorA - l1FactorA - ...``, subtracted in that order."""
    remaining = 1.0
    for factor in earlier:
        remaining -= factor
    return remaining


def _photoelectric(element: Element, count: float, energy: float) -> float:
    """``calculateCoefficientsElement(...).get(PHOTOELECTRIC)``, per cell volume."""
    sigma = float(element.absorption_coefficients(energy).photoelectric)
    return count * sigma


def escape_factors(
    elements: tuple[Element, ...],
    counts: np.ndarray,
    beam_energy: float,
    photoelectric_at: Callable[[float], float],
    *,
    surrounding: bool = False,
) -> np.ndarray:
    """``getFluorescentEscapeFactors``: one row of 27 per element.

    Parameters
    ----------
    elements : tuple[Element, ...]
        The elements present.
    counts : np.ndarray
        Atoms of each per cell.
    beam_energy : float
        keV.
    photoelectric_at : Callable[[float], float]
        The crystal's photoelectric coefficient at an energy, in inverse
        micrometres -- for how far a fluorescence photon travels.
    surrounding : bool
        ``getCryoFluorescentEscapeFactors``, which leaves the L shells'
        yields and coefficients at zero.

    Returns
    -------
    np.ndarray
        ``(elements, 27)``: the element's share of photoabsorption, then for
        K, L1, L2 and L3 the edge energy, ionisation probability,
        fluorescence yield and the crystal's photoelectric coefficient at the
        fluorescence energy, then for M1 to M5 the edge and ionisation
        probability. A shell the beam cannot ionise is zero throughout.
    """
    factors = np.zeros((len(elements), FACTOR_COLUMNS))
    per_element = [
        _photoelectric(e, float(c), beam_energy)
        for e, c in zip(elements, counts, strict=True)
    ]
    total = 0.0
    for value in per_element:
        total += value
    for row, element in enumerate(elements):
        data = _shells(element.atomic_number)
        k_edge, l1_edge, l2_edge, l3_edge = data.edges[:4]
        factor_a = [0.0] * 9
        if beam_energy > k_edge:
            factor_a[0] = data.ionisation[0]
            factors[row, 1:3] = (k_edge, factor_a[0])
            # A shell with no tabulated emission energy has no fluorescence
            # to place, so it relaxes locally: hydrogen's K shell, and the L
            # shells of Mg to Co, whose distances were NaN.
            if data.average_k > 0:
                factors[row, 3:5] = (data.yields[0], photoelectric_at(data.average_k))
        light = element.atomic_number < 12
        for shell, (edge, column) in enumerate(
            ((l1_edge, 5), (l2_edge, 9), (l3_edge, 13)), start=1
        ):
            if beam_energy > edge and not light:
                factor_a[shell] = data.ionisation[shell] * _remaining(factor_a[:shell])
                factors[row, column] = edge
                factors[row, column + 1] = factor_a[shell]
                if not surrounding and data.average_l > 0:
                    factors[row, column + 2] = data.yields[shell]
                    factors[row, column + 3] = photoelectric_at(data.average_l)
        for shell in range(4, 9):
            edge = data.edges[shell]
            if beam_energy > edge and element.atomic_number >= 73:
                factor_a[shell] = data.ionisation[shell] * _remaining(factor_a[:shell])
                column = 17 + 2 * (shell - 4)
                factors[row, column] = edge
                factors[row, column + 1] = factor_a[shell]
        factors[row, 0] = per_element[row] / total
    return factors


def _binding_energy(factors: np.ndarray) -> float:
    """``calculatePEEnergySubtraction``: mean binding energy lost, keV."""
    shells = [
        (1, 2),
        (5, 6),
        (9, 10),
        (13, 14),
        *((17 + 2 * m, 18 + 2 * m) for m in range(5)),
    ]
    totals = []
    for energy_column, probability_column in shells:
        total = 0.0
        for row in factors:
            total += row[0] * row[energy_column] * row[probability_column]
        totals.append(total)
    result = 0.0
    for total in totals:
        result += total
    return result


# --- the photoelectron's path ------------------------------------------------


def _gumbel_location(density: float, energy: float) -> tuple[float, float]:
    if energy <= 20:
        return (
            0.0105 * density**2 - 0.0351 * density + 0.0387,
            0.0245 * density**2 - 0.0943 * density + 0.1171,
        )
    if energy <= 50:
        return (
            0.009 * density**2 - 0.0293 * density + 0.0318,
            0.0459 * density**2 - 0.1942 * density + 0.2582,
        )
    return (
        0.0072 * density**2 - 0.024 * density + 0.0262,
        0.16925 * density**2 - 0.05562 * density + 0.6075,
    )


def _gumbel_scale(density: float, energy: float) -> tuple[float, float]:
    if energy <= 20:
        return (
            0.0029 * density**2 - 0.0081 * density + 0.0076,
            0 * density**2 - 0.0085 * density + 0.01751,
        )
    if energy <= 50:
        return (
            0.0018 * density**2 - 0.0053 * density + 0.0051,
            0 * density**2 - 0.0238 * density + 0.0536,
        )
    return (
        0 * density**2 - 0.0006 * density + 0.0017,
        0 * density**2 - 0.0569 * density + 0.1068,
    )


def _gumbel_pdf(x: float, location: float, scale: float) -> float:
    z = (x - location) / scale
    return math.exp(z - math.exp(z)) / scale


def _java_pow(x: float, n: int) -> float:
    """``Math.pow(x, n)`` for a small integer power."""
    return math.pow(x, n)


def _max_distance(energy: float, location: float, scale: float) -> float:
    """``getMaxPEDistance``: where the path-length density falls to 0.1%."""
    target = _gumbel_pdf(location, location, scale) * 0.001
    height = _gumbel_pdf(location, location, scale)
    step = energy / 100
    if energy < 10:
        step /= 10
    distance = location
    while height >= target:
        distance += step
        height = _gumbel_pdf(distance, location, scale)
        if distance >= location * 2:
            break
    return distance + 1


@functools.cache
def _energy_table(name: str) -> list[tuple[float, tuple[float, ...]]]:
    text = resources.files("pyraddose_3d.data").joinpath(name).read_text("ascii")
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        rows.append((float(fields[0]), tuple(float(f) for f in fields[1:8])))
    return rows


def _energy_coefficients(name: str, energy: float) -> list[float]:
    """``ReadEnergyCSV.openCSV``: the nearest row's polynomial, highest power first."""
    used = 1.01 if energy <= 1.0 else 99.99 if energy >= 100 else energy
    coefficients = [0.0] * 7
    previous = None
    for row_energy, values in _energy_table(name):
        if row_energy > _round_long(used * 100.0) / 100.0:
            assert previous is not None
            interpolation = (used - previous[0]) / (row_energy - previous[0])
            chosen = previous[1] if interpolation <= 0.5 else values
            for i in range(7):
                coefficients[6 - i] = chosen[i]
            break
        previous = (row_energy, values)
    return coefficients


def _polynomial(c: list[float], x: float) -> float:
    return (
        c[6] * _java_pow(x, 6)
        + c[5] * _java_pow(x, 5)
        + c[4] * _java_pow(x, 4)
        + c[3] * _java_pow(x, 3)
        + c[2] * _java_pow(x, 2)
        + c[1] * x
        + c[0]
    )


def _energy_deposited(distances: np.ndarray, bins: int, energy: float) -> list[float]:
    """``calculateEnergyDistn``: deposition profile along a track of ``bins``."""
    low = _energy_coefficients("pe_energy_low.csv", energy)
    medium = _energy_coefficients("pe_energy_medium.csv", energy)
    high = _energy_coefficients("pe_energy_high.csv", energy)
    out = []
    for i in range(bins):
        with np.errstate(invalid="ignore"):
            x = float(distances[i] / distances[bins - 1])
        if x < 2 / energy:
            out.append(0.0 if x == 0 else _polynomial(low, x))
        elif x > 1 - (2 / energy):
            out.append(_polynomial(high, x))
        else:
            out.append(_polynomial(medium, x))
    return out


def _dose_by_distance(
    distances: np.ndarray, location: float, scale: float, energy: float
) -> np.ndarray:
    """``calcProportionVoxDoseDepositedByDist``: the share of dose at each distance."""
    bins = len(distances)
    path = [_gumbel_pdf(float(d), location, scale) for d in distances]
    total_energy = [[0.0] * bins for _ in range(bins)]
    for i in range(bins):
        profile = _energy_deposited(distances, bins - i, energy)
        for j in range(len(profile) - 1, 0, -1):
            total_energy[i][j] = profile[j]

    path_integral = 0.0
    energy_integral = [0.0] * bins
    for l in range(bins - 1):  # noqa: E741 - the Java's name
        width = float(distances[l + 1] - distances[l])
        path_integral += width * ((path[l + 1] + path[l]) / 2)
        for i in range(bins):
            if total_energy[i][l + 1] != 0:
                energy_integral[i] += width * (
                    (total_energy[i][l + 1] + total_energy[i][l]) / 2
                )

    widths = [0.0] * bins
    heights = [0.0] * bins
    count = -1
    for l in range(bins - 1, 0, -1):  # noqa: E741
        count += 1
        widths[count] = float(distances[l] - distances[l - 1])
        heights[count] = (path[l] + path[l - 1]) / 2

    proportion = [0.0] * bins
    for l in range(bins - 1, 0, -1):  # noqa: E741
        for i in range(bins):
            height = 0.0
            if total_energy[i][l] != 0:
                height = (total_energy[i][l] + total_energy[i][l - 1]) / 2
            if energy_integral[i] != 0:
                proportion[l] += (widths[i] * heights[i] / path_integral) * (
                    widths[i] * height / energy_integral[i]
                )
    return np.array(proportion)


def _round_long(x: float) -> int:
    """``Math.round(double)``: half up, to a long."""
    return math.floor(x) + (1 if x - math.floor(x) >= 0.5 else 0)


def _polarisation_weights(factors: np.ndarray) -> list[float]:
    """``setUpPEPolarisation``: how many copies of each direction to draw from."""

    def height(phi: float) -> float:
        return (1 / (4 * math.pi)) * (
            1 + (_BETA * 0.5 * (3 * math.pow(math.cos(phi), 2) - 1))
        )

    weight = 0.0
    for row in factors:
        weight += row[0] * row[2]
    weights = []
    for i in range(_PE_ANGLE_RES_LIMIT // 2 + 1):
        angle = i * (_PE_ANGLE_LIMIT / _PE_ANGLE_RES_LIMIT)
        point = height(angle) / height(0)
        total = point * weight * _DEGREE_OF_POLARISATION
        weights.append(
            _round_long(1000 * total)
            + (1000 * (1 - (weight * _DEGREE_OF_POLARISATION)))
        )
    return weights


@dataclass(frozen=True)
class _Tracks:
    """The directions of ``findVoxelsReachedByPE``, before any rotation."""

    directions: np.ndarray  # (tracks, 3): x, y, z unit vectors
    bias: np.ndarray  # the track index for each draw value


def _tracks(weights: list[float], vertical_goniometer: bool) -> _Tracks:
    """``findVoxelsReachedByPE``'s directions and draw table.

    Floating-point loop counters and Java's integer conversions are kept
    exactly, because they decide how many directions there are and which
    track each random draw maps to.
    """
    numbers = 0
    for w in weights:
        numbers = int(numbers + w)  # int += double narrows each time
    numbers *= _PE_ANGLE_RES_LIMIT
    numbers = int(numbers * 1.5)

    step = 2 * math.pi / _PE_ANGLE_RES_LIMIT
    counter = -1
    copies = []
    directions = []
    theta = 0 * math.pi
    while theta < 2 * math.pi:
        phi = 0.0
        while phi <= _PE_ANGLE_LIMIT / 2:
            replicate = (theta == 0 or theta == (_PE_ANGLE_LIMIT / 2)) and phi != 0
            if not replicate:
                counter += 1
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                dot = x if vertical_goniometer else y
                magnitude = math.sqrt(x**2 + y**2 + z**2)
                angle_to_axis = math.acos(dot / magnitude)
                place = int(
                    np.rint((angle_to_axis * _PE_ANGLE_RES_LIMIT) / _PE_ANGLE_LIMIT)
                )
                # `while (running < weight) { add; running++ }` with an int
                # counter against a double weight: ceil(weight) copies.
                weight = weights[place]
                copies.append(math.ceil(weight) if weight > 0 else 0)
                directions.append((x, y, z))
            phi += step
        theta += step

    filled = np.repeat(np.arange(len(copies), dtype=np.int64), copies)
    if len(filled) > numbers:
        raise IndexError("trackNumberBias overflows its array, as it would in Java")
    big = np.zeros(numbers, dtype=np.int64)
    big[: len(filled)] = filled
    # The Java keeps everything up to the first zero after the first
    # non-zero entry -- the end of what was filled -- or nothing at all if
    # the array has no such zero.
    nonzero = np.flatnonzero(big > 0)
    actual = 0
    if len(nonzero):
        trailing = np.flatnonzero(big[nonzero[0] :] == 0)
        if len(trailing):
            actual = int(nonzero[0] + trailing[0])
    return _Tracks(np.array(directions), big[:actual].copy())


def _rotated_offsets(
    tracks: _Tracks, distances: np.ndarray, pixels_per_micron: float, angle: float
) -> np.ndarray:
    """``relativeVoxXYZ``: track positions in voxels, ``(bins, tracks, 3)``."""
    turns = int(angle / (2 * math.pi))
    this_angle = 2 * math.pi - (angle - (turns * 2 * math.pi))
    x, y, z = tracks.directions.T
    x_rot = x * math.cos(this_angle) + z * math.sin(this_angle)
    z_rot = -1 * x * math.sin(this_angle) + z * math.cos(this_angle)
    r = distances[:, None] * pixels_per_micron
    return np.stack([r * x_rot, r * y, r * z_rot], axis=-1)


def round_half_up(x: np.ndarray) -> np.ndarray:
    """``StrictMath.round``: to the nearest integer, halves up.

    Exactly, including 0.49999999999999994, which ``floor(x + 0.5)`` gets
    wrong because the addition rounds up -- and NaN, which Java rounds to 0.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.where(np.isnan(x), 0.0, x)
    down = np.floor(finite)
    return (down + (finite - down >= 0.5)).astype(np.int64)


@dataclass(frozen=True)
class PhotoelectronEscape:
    """``setPEparamsForCurrentBeam`` and what it computes.

    Attributes
    ----------
    binding_energy : float
        ``EnergyToSubtractFromPE``, keV.
    distances : np.ndarray
        ``PE_DISTANCES_TRAVELLED``, micrometres.
    proportion : np.ndarray
        ``propnDoseDepositedAtDist``: share of the dose at each distance.
    tracks : _Tracks
        Directions and the draw table.
    """

    binding_energy: float
    distances: np.ndarray
    proportion: np.ndarray
    tracks: _Tracks
    location: float
    scale: float
    pe_energy: float


def photoelectron_escape(
    factors: np.ndarray,
    beam_energy: float,
    density: float,
    pixels_per_micron: float,
    resolution: int | None,
    vertical_goniometer: bool,
) -> tuple[PhotoelectronEscape, int]:
    """Set up the crystal's photoelectron escape. Returns it and ``peDistBins``."""
    binding = _binding_energy(factors)
    pe_energy = beam_energy - binding
    location_coefficients = _gumbel_location(density, pe_energy)
    scale_coefficients = _gumbel_scale(density, pe_energy)
    location = (
        location_coefficients[0] * math.pow(pe_energy, 2)
        + location_coefficients[1] * pe_energy
    )
    scale = (
        scale_coefficients[0] * math.pow(pe_energy, 2)
        + scale_coefficients[1] * pe_energy
    )

    max_distance = math.ceil(_max_distance(pe_energy, location, scale))
    if resolution is not None and resolution >= 2:
        bins = resolution
    else:
        pixel = 1 / pixels_per_micron
        if pixel >= max_distance:
            bins = 2
        else:
            bins = 1 + math.ceil(max_distance / pixel)
            bins = max(bins, max_distance)
            bins += 1
            bins += 50
    interval = max_distance / (bins - 1)
    distances = np.array([i * interval for i in range(bins)])
    weights = _polarisation_weights(factors)
    escape = PhotoelectronEscape(
        binding_energy=binding,
        distances=distances,
        proportion=_dose_by_distance(distances, location, scale, pe_energy),
        tracks=_tracks(weights, vertical_goniometer),
        location=location,
        scale=scale,
        pe_energy=pe_energy,
    )
    return escape, bins


# --- fluorescence -------------------------------------------------------------


@dataclass(frozen=True)
class FluorescenceEscape:
    """``setFLparamsForCurrentBeam`` and the energies around it.

    Attributes
    ----------
    release : float
        ``calcFluorescence``: fluorescence energy per absorbed photon, J.
    proportion : np.ndarray
        ``(elements, 4)``: each element and shell's share of ``release``.
    distances : np.ndarray
        ``(elements, 4, bins)`` in micrometres.
    distribution : np.ndarray
        ``(elements, 4, bins)``: where the fluorescence is reabsorbed, the
        last bin being what escapes.
    directions : np.ndarray
        ``(numberOfTracksFL, 3)`` unit vectors, one drawn per voxel.
    """

    release: float
    proportion: np.ndarray
    distances: np.ndarray
    distribution: np.ndarray
    directions: np.ndarray

    @property
    def tracks(self) -> int:
        """``numberOfTracksFL``: the bound of each voxel's draw."""
        return len(self.directions)


def fluorescence_escape(
    factors: np.ndarray, crystal_size: np.ndarray, resolution: int | None
) -> FluorescenceEscape:
    """Set up fluorescence escape for the crystal."""
    per_event = np.zeros((len(factors), 4))
    release = 0.0
    for i, row in enumerate(factors):
        mu_ratio = row[0]
        k1 = row[1] * row[2] * row[3]
        l1 = row[5] * row[6] * row[7]
        l2 = row[9] * row[10] * row[11]
        l3 = row[13] * row[14] * row[15]
        per_event[i] = (
            k1 * mu_ratio * KEV_TO_JOULES,
            l1 * mu_ratio * KEV_TO_JOULES,
            l2 * mu_ratio * KEV_TO_JOULES,
            l3 * mu_ratio * KEV_TO_JOULES,
        )
        release = release + ((k1 + l1 + l2 + l3) * mu_ratio)
    release = release * KEV_TO_JOULES
    with np.errstate(invalid="ignore", divide="ignore"):
        proportion = per_event / release

    bins = (
        resolution if resolution is not None and resolution >= 2 else _FL_DISTANCE_BINS
    )
    resolution_steps = bins - 1
    distances = np.zeros((len(factors), 4, bins))
    distribution = np.zeros((len(factors), 4, bins))
    diagonal = math.pow(
        math.pow(crystal_size[0], 2)
        + math.pow(crystal_size[1], 2)
        + math.pow(crystal_size[2], 2),
        0.5,
    )
    for i, row in enumerate(factors):
        for j in range(4):
            if not proportion[i, j] > 0:
                continue
            mu = row[4 * j + 4]
            with np.errstate(divide="ignore"):
                max_fl = -1 * (math.log(0.05) / mu) if mu != 0 else math.inf
            max_fl = min(max_fl, diagonal) if max_fl > diagonal else max_fl
            for q in range(resolution_steps + 1):
                distances[i, j, q] = (max_fl / resolution_steps) * q
            running = 0.0
            for l in range(bins - 1, -1, -1):  # noqa: E741
                reached = math.exp(-mu * distances[i, j, l])
                distribution[i, j, l] = reached if l == bins - 1 else reached - running
                running += distribution[i, j, l]

    return FluorescenceEscape(
        release, proportion, distances, distribution, _fluorescence_directions()
    )


def _fluorescence_directions() -> np.ndarray:
    """``findVoxelsReachedByFL``'s directions: a Fibonacci lattice.

    Fluorescence is emitted with equal probability in every direction, so
    the directions are spread evenly over the sphere, each standing for the
    same solid angle. Formed as the Java forms them, the azimuth reduced to
    ``[0, 2 pi)`` before its cosine and sine are taken.

    Returns
    -------
    np.ndarray
        ``(256, 3)`` unit vectors.
    """
    count = _FL_ANGLE_RES_LIMIT * _FL_ANGLE_RES_LIMIT
    golden_angle = math.pi * (3 - math.sqrt(5))
    directions = np.empty((count, 3))
    for track in range(count):
        z = 1 - (2.0 * track + 1) / count
        radius = math.sqrt(1 - z * z)
        phi = math.fmod(track * golden_angle, 2 * math.pi)
        directions[track] = (radius * math.cos(phi), radius * math.sin(phi), z)
    return directions


# --- the surrounding ------------------------------------------------------------


@dataclass(frozen=True)
class Surrounding:
    """The surrounding's grid and photoelectrons.

    Attributes
    ----------
    pixels_per_micron : float
        ``cryoPPM``.
    extra_voxels : int
        How far the grid extends past the crystal on each side, in its voxels.
    shape : tuple[int, int, int]
        ``cryoCrystSizeVoxels``.
    origin : np.ndarray
        Coordinates of voxel ``(0, 0, 0)``, micrometres.
    """

    pixels_per_micron: float
    extra_voxels: int
    shape: tuple[int, int, int]
    origin: np.ndarray
    binding_energy: float
    distances: np.ndarray
    proportion: np.ndarray
    tracks: _Tracks
    composition_density: float

    def coordinates(self) -> np.ndarray:
        """``getCryoCrystCoord`` for every voxel, ``(N, 3)`` in i, j, k order."""
        axes = [
            self.origin[a] + np.arange(self.shape[a]) / self.pixels_per_micron
            for a in range(3)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        return np.stack([g.reshape(-1) for g in grid], axis=-1)


def surrounding_composition(
    cell_volume: float,
    heavy_concentrations: dict[str, float],
    density_based: bool,
    elements: dict[str, float],
    density: float,
) -> tuple[Composition, float]:
    """``addCryoConcentrations`` and ``calculateCryoDensity``.

    The surrounding fills a cell the size of the crystal's unit cell: water,
    less whatever its solutes displace, unless it is ``density_based`` --
    then ``elements`` at ``density`` g/ml instead. Unlike the crystal's
    solvent, a negative water count is not clamped.

    Returns
    -------
    tuple[Composition, float]
        The surrounding's cell, and its density in g/cm^3.
    """
    from pyraddose_3d.backend.composition import (
        _MASS_TO_CELL_VOLUME,
        _UNITS_PER_MILLI,
        AVOGADRO,
        WATER_CONCENTRATION,
        Composition,
    )

    database = element_database()
    concentration: dict[int, float] = {}
    for symbol, value in heavy_concentrations.items():
        number = database.by_symbol(symbol).atomic_number
        concentration[number] = concentration.get(number, 0.0) + value
    if density_based and elements:
        grams_per_mole = 0.0
        for symbol, count in elements.items():
            grams_per_mole += database.by_symbol(symbol).atomic_weight * count
        oil = (density / grams_per_mole) * 1e6
        for symbol, count in elements.items():
            number = database.by_symbol(symbol).atomic_number
            concentration[number] = concentration.get(number, 0.0) + oil * count

    occurrence: dict[int, float] = {}
    non_water = 0.0
    for number, value in concentration.items():
        atoms = value * AVOGADRO * cell_volume * 1e-3 * 1e-27
        occurrence[number] = occurrence.get(number, 0.0) + atoms
        non_water += atoms
    if not density_based:
        water = (
            WATER_CONCENTRATION * AVOGADRO / _UNITS_PER_MILLI * cell_volume
            * (1 / _MASS_TO_CELL_VOLUME) - non_water
        )  # fmt: skip
        occurrence[1] = occurrence.get(1, 0.0) + water * 2
        occurrence[8] = occurrence.get(8, 0.0) + water

    numbers = sorted(occurrence)
    present = tuple(database.by_number(n) for n in numbers)
    counts = np.array([occurrence[n] for n in numbers])
    mass = 0.0
    for element, count in zip(present, counts, strict=True):
        mass += float(count) * element.atomic_weight_in_grams
    surrounding_density = mass * _MASS_TO_CELL_VOLUME / (cell_volume * _UNITS_PER_MILLI)
    return Composition(present, counts, cell_volume), surrounding_density


def _surrounding_ppm(
    max_distance: int, crystal_size: np.ndarray, beam_minimum: float, crystal_ppm: float
) -> float:
    """``setCryoPPM``."""
    smallest = min(crystal_size[2], min(crystal_size[0], crystal_size[1]))
    multiply = 20
    if smallest >= beam_minimum:
        multiply = int(multiply * 1.5)
    ideal = ((1 / max_distance) * 5) + ((1 / max_distance) * multiply * (1 / smallest))
    if ideal >= crystal_ppm:
        return float(math.ceil(ideal / crystal_ppm) * crystal_ppm)
    return float(crystal_ppm / int(crystal_ppm / ideal))


def surrounding_escape(
    factors: np.ndarray,
    beam_energy: float,
    surrounding_density: float,
    crystal_density: float,
    bins: int,
    mesh_min: np.ndarray,
    mesh_max: np.ndarray,
    crystal_ppm: float,
    beam_minimum: float,
    vertical_goniometer: bool,
) -> Surrounding:
    """``setCryoPEparamsForCurrentBeam`` and ``produceCryoSolutionCrystal``."""
    binding = _binding_energy(factors)
    pe_energy = beam_energy - binding
    loc_c = _gumbel_location(surrounding_density, pe_energy)
    scale_c = _gumbel_scale(surrounding_density, pe_energy)
    location = loc_c[0] * math.pow(pe_energy, 2) + loc_c[1] * pe_energy
    scale = scale_c[0] * math.pow(pe_energy, 2) + scale_c[1] * pe_energy
    max_distance = math.ceil(_max_distance(pe_energy, location, scale))

    crystal_size = mesh_max - mesh_min
    average_side = (crystal_size[0] + crystal_size[1] + crystal_size[2]) / 3
    share = (
        0.5
        if average_side >= max_distance
        else max_distance / (max_distance + average_side)
    )

    ppm = _surrounding_ppm(max_distance, crystal_size, beam_minimum, crystal_ppm)
    extra = int(max_distance / (1 / ppm))
    dims = crystal_size + ((extra / ppm) * 2)
    shape = tuple(_round_long(d * ppm) + 1 for d in dims)
    origin = mesh_min - (extra / ppm)

    interval = max_distance / (bins - 1)
    distances = np.array([i * interval for i in range(bins)])
    # Re-fitted to a density between the surrounding's and the crystal's,
    # but the maximum distance is not recomputed.
    mixed = (surrounding_density * share) + (crystal_density * (1 - share))
    loc_c = _gumbel_location(mixed, pe_energy)
    scale_c = _gumbel_scale(mixed, pe_energy)
    location = loc_c[0] * math.pow(pe_energy, 2) + loc_c[1] * pe_energy
    scale = scale_c[0] * math.pow(pe_energy, 2) + scale_c[1] * pe_energy

    return Surrounding(
        pixels_per_micron=ppm,
        extra_voxels=extra,
        shape=(shape[0], shape[1], shape[2]),
        origin=np.asarray(origin, dtype=np.float64),
        binding_energy=binding,
        distances=distances,
        proportion=_dose_by_distance(distances, location, scale, pe_energy),
        tracks=_tracks(_polarisation_weights(factors), vertical_goniometer),
        composition_density=surrounding_density,
    )


class Escape:
    """Photoelectron and fluorescence escape for one run.

    Built once, by :func:`pyraddose_3d.solve.simulate`, then called by
    :func:`~pyraddose_3d.backend.exposure.expose` for each angle and energy
    with the voxels that step lit, in place of simply adding their dose.

    Parameters
    ----------
    random : JavaRandom
        The run's random numbers, shared across wedges as in the Java.
    shape : tuple[int, int, int]
        The crystal's voxel grid.
    occupied : np.ndarray
        ``(N,)`` boolean, which voxels are inside the crystal.
    pixels_per_micron : float
        The crystal's resolution.
    density : float
        The crystal's density, g/cm^3.
    beam_energy : float
        The beam's (mean) energy, keV.
    photoelectron, fluorescence : optional
        Each kind of escape that is on.
    surrounding : Surrounding | None
        With photoelectron escape, the medium around the crystal.
    surrounding_absorption : np.ndarray | None
        The surrounding's photoelectric coefficient at each sampled energy.
    """

    def __init__(
        self,
        random: JavaRandom,
        shape: tuple[int, int, int],
        occupied: np.ndarray,
        pixels_per_micron: float,
        density: float,
        beam_energy: float,
        photoelectron: PhotoelectronEscape | None,
        fluorescence: FluorescenceEscape | None,
        surrounding: Surrounding | None = None,
        surrounding_absorption: np.ndarray | None = None,
    ) -> None:
        self.random = random
        self.shape = shape
        self.occupied = np.asarray(occupied, dtype=bool)
        self.ppm = pixels_per_micron
        self.density = density
        self.beam_energy = beam_energy
        self.photoelectron = photoelectron
        self.fluorescence = fluorescence
        self.surrounding = surrounding
        self.surrounding_absorption = surrounding_absorption
        nx, ny, nz = shape
        grid = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        self.ijk = np.stack([g.reshape(-1) for g in grid], axis=-1)
        self._surrounding_geometry: tuple[Any, tuple[np.ndarray, ...]] | None = None
        if surrounding is not None:
            self._prepare_surrounding()

    # -- the crystal's grid ---------------------------------------------------

    def _inside(self, ijk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``isCrystalAt`` for integer indices: flat index and whether inside."""
        nx, ny, nz = self.shape
        i, j, k = ijk[..., 0], ijk[..., 1], ijk[..., 2]
        bounded = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
        flat = np.where(bounded, (i * ny + j) * nz + k, 0)
        return flat, bounded & self.occupied[flat]

    def _deposit(
        self, dose: np.ndarray, targets: np.ndarray, amounts: np.ndarray
    ) -> None:
        flat, inside = self._inside(targets)
        np.add.at(dose, flat[inside], amounts[inside])

    def expose_step(
        self,
        *,
        dose: np.ndarray,
        lit: np.ndarray,
        photoelectric_dose: np.ndarray,
        photons: np.ndarray,
        fluence_to_dose: float,
        angle: float,
        rotated: Any,
        shift: np.ndarray,
        beam: Any,
        off_axis: float,
        attenuation: float,
        exposure_seconds: float,
        angle_count: int,
        sample: int,
        photon_energy: float,
    ) -> np.ndarray:
        """Add one angle and energy's photoelectric dose, with escape.

        ``Crystal.exposeAngle``'s voxel loop from the photoelectric dose on,
        and its loop over the surrounding. The Compton dose, elastic yield
        and the diffraction-weighted dose are the caller's.

        Returns
        -------
        np.ndarray
            The new ``(N,)`` dose.
        """
        dose = np.array(dose, dtype=np.float64, copy=True)
        index = np.flatnonzero(lit & (photoelectric_dose > 0))
        if len(index):
            self._crystal_step(
                dose, index, photoelectric_dose, photons, fluence_to_dose, angle,
                photon_energy,
            )  # fmt: skip
        if self.surrounding is not None and self.photoelectron is not None:
            self._surrounding_step(
                dose, angle, rotated, shift, beam, off_axis, attenuation,
                exposure_seconds, angle_count, sample, photon_energy,
            )  # fmt: skip
        return dose

    def _crystal_step(
        self,
        dose: np.ndarray,
        index: np.ndarray,
        photoelectric_dose: np.ndarray,
        photons: np.ndarray,
        fluence_to_dose: float,
        angle: float,
        photon_energy: float,
    ) -> None:
        """The lit crystal voxels, in ``i, j, k`` order.

        The photoelectron leaves with the photon's energy less the binding
        energy, which the atom releases as it relaxes: as fluorescence,
        tracked when fluorescence escape is on, and the rest locally.
        """
        pe, fl = self.photoelectron, self.fluorescence
        image_dose = photoelectric_dose[index]
        photon_count = photons[index]
        ijk = self.ijk[index]

        zeros = np.zeros(len(index))
        fl_release, relaxation, dose_pe = zeros, zeros, zeros
        if fl is not None:
            fl_release = fluence_to_dose * (fl.release * photon_count)
        if pe is not None:
            relaxation = (pe.binding_energy / photon_energy) * image_dose
            dose_pe = image_dose - relaxation

        # The draws, in the Java's order: voxel by voxel, the fluorescence's
        # direction when there is fluorescence to release, then the
        # photoelectron's.
        slots = (1 if fl is not None else 0) + (PE_ANGLE_RESOLUTION**2 if pe else 0)
        draws = np.zeros((len(index), 0), dtype=np.int64)
        if slots:
            bounds = np.full((len(index), slots), -1, dtype=np.int64)
            column = 0
            if fl is not None:
                bounds[:, 0] = np.where(fl_release > 0, fl.tracks, -1)
                column = 1
            if pe is not None:
                bounds[:, column:] = len(pe.tracks.bias)
            order = bounds.reshape(-1)
            draws = np.full(order.shape, -1, dtype=np.int64)
            wanted = order > 0
            draws[wanted] = self.random.next_ints(order[wanted])
            draws = draws.reshape(bounds.shape)

        if fl is not None:
            self._fluorescence(dose, ijk, fl_release, draws[:, 0])
            if pe is None:
                np.add.at(dose, index, image_dose - fl_release)
        if pe is not None:
            if fl is not None:
                local = relaxation - fl_release
                np.add.at(dose, index, np.where(local > 0, local, 0.0))
            offsets = _rotated_offsets(pe.tracks, pe.distances, self.ppm, angle)
            bias = pe.tracks.bias
            first = 1 if fl is not None else 0
            for q in range(PE_ANGLE_RESOLUTION**2):
                tracks = bias[draws[:, first + q]]
                targets = round_half_up(
                    ijk[:, None, :] + offsets[:, tracks, :].transpose(1, 0, 2)
                )
                amounts = (
                    dose_pe[:, None]
                    * pe.proportion[None, :]
                    / math.pow(PE_ANGLE_RESOLUTION, 2)
                )
                self._deposit(dose, targets, amounts)
            if fl is None:
                np.add.at(dose, index, np.where(relaxation > 0, relaxation, 0.0))
        if pe is None and fl is None:
            np.add.at(dose, index, image_dose)

    def _fluorescence(
        self,
        dose: np.ndarray,
        ijk: np.ndarray,
        release: np.ndarray,
        draws: np.ndarray,
    ) -> None:
        """``addDoseAfterFL``: along the direction drawn for each lit voxel."""
        fl = self.fluorescence
        assert fl is not None
        releasing = release > 0
        ijk, release = ijk[releasing].astype(np.float64), release[releasing]
        directions = fl.directions[draws[releasing]]
        for n in range(fl.proportion.shape[0]):
            for shell in range(4):
                share = fl.proportion[n, shell]
                if share == 0:
                    continue
                for m in range(fl.distances.shape[2]):
                    amount = release * share * fl.distribution[n, shell, m]
                    r = fl.distances[n, shell, m] * self.ppm
                    targets = ijk + r * directions
                    self._deposit(dose, round_half_up(targets), amount)

    # -- the surrounding --------------------------------------------------------

    def _prepare_surrounding(self) -> None:
        surrounding = self.surrounding
        assert surrounding is not None
        coordinates = surrounding.coordinates()
        nx, ny, nz = surrounding.shape
        grid = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        index = np.stack([g.reshape(-1) for g in grid], axis=-1)
        ratio = self.ppm / surrounding.pixels_per_micron
        in_crystal_grid = (index - surrounding.extra_voxels) * ratio
        _, inside = self._inside(round_half_up(in_crystal_grid))
        # Only voxels outside the crystal are surrounding.
        self._surrounding_coordinates = coordinates[~inside]
        self._surrounding_in_crystal = in_crystal_grid[~inside]

    def _surrounding_step(
        self,
        dose: np.ndarray,
        angle: float,
        rotated: Any,
        shift: np.ndarray,
        beam: Any,
        off_axis: float,
        attenuation: float,
        exposure_seconds: float,
        angle_count: int,
        sample: int,
        photon_energy: float,
    ) -> None:
        """``exposeAngle``'s loop over the surrounding."""
        from pyraddose_3d.backend.exposure import _place
        from pyraddose_3d.backend.geometry import find_path_in_front

        surrounding = self.surrounding
        pe = self.photoelectron
        assert surrounding is not None and pe is not None
        assert self.surrounding_absorption is not None
        # Where the surrounding is, how much beam reaches it and through how
        # much crystal depend on the angle alone, so a pink beam's energies
        # share them.
        key = (angle, tuple(np.asarray(shift, dtype=np.float64).tolist()), off_axis)
        if self._surrounding_geometry is None or self._surrounding_geometry[0] != key:
            placed = _place(self._surrounding_coordinates, np.asarray(shift), angle)
            intensity = np.asarray(beam.intensity(placed[:, 0], placed[:, 1], off_axis))
            lit = intensity > 0
            placed, intensity = placed[lit], intensity[lit]
            # How much crystal the beam crossed to get here.
            depth = find_path_in_front(rotated, placed)
            offsets = _rotated_offsets(
                pe.tracks, surrounding.distances, self.ppm, angle
            )
            self._surrounding_geometry = (
                key,
                (intensity, self._surrounding_in_crystal[lit], depth, offsets),
            )
        intensity, in_crystal_grid, depth, offsets = self._surrounding_geometry[1]
        # Not divided between a pink beam's energies -- reproduced.
        attenuation_factor = (
            math.pow(surrounding.pixels_per_micron, -2) * exposure_seconds / angle_count
        )
        fluence = intensity * attenuation_factor * np.exp(depth * -attenuation)
        absorbed = float(self.surrounding_absorption[sample])
        energy_per_fluence = 1 - math.exp(-1 * absorbed / surrounding.pixels_per_micron)
        energy = energy_per_fluence * fluence
        keep = energy > 0
        energy, in_crystal_grid = energy[keep], in_crystal_grid[keep]

        # The photoelectron leaves with the photon's energy less the binding
        # energy, whether or not fluorescence escape is on.
        energy_pe = energy - (surrounding.binding_energy / photon_energy) * energy

        bias = surrounding.tracks.bias
        draws = self.random.next_ints(
            np.full(len(energy) * PE_ANGLE_RESOLUTION**2, len(bias), dtype=np.int64)
        ).reshape(len(energy), PE_ANGLE_RESOLUTION**2)
        energy_to_dose = 1e-15 * (math.pow(self.ppm, -3) * self.density)
        for q in range(PE_ANGLE_RESOLUTION**2):
            tracks = bias[draws[:, q]]
            voxel, bin_index = self._bins_reaching_crystal(
                in_crystal_grid, offsets, tracks
            )
            targets = round_half_up(
                in_crystal_grid[voxel] + offsets[bin_index, tracks[voxel], :]
            )
            amounts = (
                energy_pe[voxel] * surrounding.proportion[bin_index]
                / math.pow(PE_ANGLE_RESOLUTION, 2)
            )  # fmt: skip
            self._deposit(dose, targets, (amounts / energy_to_dose) * 1e-06)

    def _bins_reaching_crystal(
        self, starts: np.ndarray, offsets: np.ndarray, tracks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The (voxel, distance bin) pairs whose target can be in the crystal's grid.

        A track's position is linear in the bin, so the bins that can round
        into the grid -- a coordinate in ``[-0.5, n - 0.5)`` on every axis --
        form one interval per track, found by division and widened by a bin
        either side. Only those are evaluated, with the Java's own
        arithmetic; the rest cannot deposit anything. Most of a surrounding
        track lies outside the crystal, so this is most of the work saved.
        """
        bins = offsets.shape[0]
        last = offsets[-1, tracks, :]  # (voxels, 3): position change over the track
        low = np.zeros(len(starts))
        high = np.full(len(starts), float(bins - 1))
        for axis, size in enumerate(self.shape):
            start, change = starts[:, axis], last[:, axis] / (bins - 1)
            lo_edge, hi_edge = -0.5 - start, size - 0.5 - start
            with np.errstate(divide="ignore", invalid="ignore"):
                a, b = lo_edge / change, hi_edge / change
            entering = np.where(change > 0, a, np.where(change < 0, b, -np.inf))
            leaving = np.where(change > 0, b, np.where(change < 0, a, np.inf))
            # A track that does not move along this axis is in range for all
            # of it, or for none.
            still_inside = (change == 0) & (lo_edge <= 0) & (hi_edge > 0)
            entering = np.where((change == 0) & ~still_inside, np.inf, entering)
            low = np.maximum(low, np.floor(entering) - 1)
            high = np.minimum(high, np.ceil(leaving) + 1)
        low = np.clip(low, 0, bins - 1)
        high = np.clip(high, -1, bins - 1)
        counts = np.maximum(high - low + 1, 0).astype(np.int64)
        voxel = np.repeat(np.arange(len(starts)), counts)
        first = np.repeat(low.astype(np.int64), counts)
        position = np.arange(counts.sum()) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        return voxel, first + position
