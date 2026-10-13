r"""RADDOSE-3D's output files, written as the Java writes them.

A script that reads RADDOSE-3D's output reads the port's unchanged: the same
files under the same names, the same columns in the same order, and numbers
printed by the same rules -- see :mod:`pyraddose_3d._java_format` for why
those are not Python's. With the default prefix the files are:

``output-Summary.txt``
    The human-readable summary. RADDOSE-3D also prints it to the console.
``output-Summary.csv``
    The same per-wedge numbers as a table.
``output-DoseState.csv``
    ``x,y,z,dose,fluence,elastic`` for every voxel inside the crystal, at
    the end of the run.
``output-DoseState.R``
    An R script that draws dose contours of the final state.
``output-RDE.csv``, ``output-DWDs.csv``
    One row per image: relative diffraction efficiency, and the
    diffraction-weighted dose with its resolution-dependent RDE.
``output-VoxelDose.csv``, ``output-VoxelFluences.csv``
    Every voxel's dose and fluence in every image of the last wedge. These
    need per-image fields, so they are written only for a simulation run
    with ``record_images=True``.

Each file mirrors one Java class, ``Output<Name>``, and the notes on each
function say where it copies a quirk rather than fixing it. Lines always end
in ``\n``: the Java writes its platform separator in some places and ``\n``
in others, so on Windows its files have mixed endings.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d._java_format import java_double, java_e, java_f, java_float
from pyraddose_3d.backend._array import to_namespace
from pyraddose_3d.backend.summary import resolution_rde

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pyraddose_3d.models.beam import Beam
    from pyraddose_3d.models.crystal import Crystal
    from pyraddose_3d.models.wedge import Wedge
    from pyraddose_3d.solve import Simulation

__all__ = [
    "dose_state_csv",
    "dose_state_r",
    "dwds_csv",
    "final_dose_state",
    "image_series",
    "rde_csv",
    "summary_csv",
    "summary_text",
    "voxel_dose_csv",
    "voxel_fluences_csv",
    "voxel_images",
    "write_outputs",
]

#: ``OutputSummaryText``'s dose histogram: 9 bins from 0.1 to 30 MGy, plus
#: one below and one above.
_HISTOGRAM_MIN = 0.1
_HISTOGRAM_MAX = 30.0
_HISTOGRAM_BINS = 9

#: Java's ``Float.MIN_VALUE``, the smallest positive single: the per-voxel
#: writers print anything at or below it as a bare ``0``.
_FLOAT_MIN_VALUE = float(np.nextafter(np.float32(0), np.float32(1)))

#: Microns to millimetres, ``CrystalCylinder.CONVERSION_MICRONS_TO_MILLIMETERS``.
_MICRONS_TO_MILLIMETRES = 1e-3


def _numpy(values: Any) -> np.ndarray:
    return np.asarray(to_namespace(np, values))


def _float(value: Any) -> float:
    return float(_numpy(value))


# --- what the Java writes ---------------------------------------------------


def final_dose_state(simulation: Simulation) -> dict[str, np.ndarray]:
    """The crystal at the end of the run, as ``DoseState.csv`` lists it.

    Parameters
    ----------
    simulation : Simulation
        From :func:`pyraddose_3d.solve.simulate`.

    Returns
    -------
    dict[str, np.ndarray]
        ``coordinates`` ``(N, 3)`` in micrometres, and ``dose``, ``fluence``
        and ``elastic`` ``(N,)``, for the N voxels inside the crystal in the
        Java's ``i, j, k`` order.

    Notes
    -----
    The Java accumulates fluence and elastic yield on the crystal across
    every wedge, as it does dose, whereas the port's exposure resets them per
    wedge -- so they are summed here.
    """
    occupied = _numpy(simulation.occupied).astype(bool)
    exposures = simulation.exposures
    return {
        "coordinates": _numpy(simulation.grid.coordinates())[occupied],
        "dose": _numpy(exposures[-1].dose)[occupied],
        "fluence": np.sum([_numpy(e.fluence) for e in exposures], axis=0)[occupied],
        "elastic": np.sum([_numpy(e.elastic) for e in exposures], axis=0)[occupied],
    }


def _require_images(simulation: Simulation) -> None:
    for exposure in simulation.exposures:
        if exposure.images is None:
            raise ValueError(
                "this needs per-image fields; run simulate() with record_images=True"
            )


def voxel_images(simulation: Simulation) -> dict[str, np.ndarray]:
    """Per-image dose and fluence over the last wedge, as the Java lists them.

    Parameters
    ----------
    simulation : Simulation
        From :func:`pyraddose_3d.solve.simulate` with ``record_images=True``.

    Returns
    -------
    dict[str, np.ndarray]
        ``coordinates`` ``(N, 3)``, and ``dose`` and ``fluence`` ``(A, N)``
        for the A images of the last wedge.

    Notes
    -----
    Only the last wedge, because ``OutputVoxelDose`` keeps the last wedge it
    was shown. A voxel's dose in an image is zero unless it was exposed in
    that image: the Java builds a fresh array per image and fills only the
    voxels it observes, so an unexposed voxel reads zero, not its dose so far.
    """
    _require_images(simulation)
    occupied = _numpy(simulation.occupied).astype(bool)
    last = simulation.exposures[-1].images
    assert last is not None
    fluence = _numpy(last.fluence)[:, occupied]
    dose = _numpy(last.dose)[:, occupied]
    return {
        "coordinates": _numpy(simulation.grid.coordinates())[occupied],
        "dose": np.where(fluence > 0, dose, 0.0),
        "fluence": fluence,
    }


def image_series(simulation: Simulation) -> dict[str, np.ndarray]:
    """One row per image of every wedge, as ``DWDs.csv`` and ``RDE.csv``.

    Parameters
    ----------
    simulation : Simulation
        From :func:`pyraddose_3d.solve.simulate` with ``record_images=True``.

    Returns
    -------
    dict[str, np.ndarray]
        ``wedge`` and ``image`` (both from 1), ``angle`` in degrees, ``dwd``
        in MGy, ``volume`` exposed in cubic micrometres, ``rde`` ``(A, 5)`` in
        the Java's order ``[max_res, 1A, 2A, 3A, 4A]``, and for ``RDE.csv``
        ``rde_angle``, ``weighted`` and ``minimum``.

    Notes
    -----
    ``weighted`` is the fluence-weighted mean of the decay model's
    efficiency over exposed voxels, and ``minimum`` its running minimum --
    reset per wedge, not per image. ``RDE.csv`` writes an image with no
    exposed voxels as zeros throughout, its angle included.
    """
    _require_images(simulation)
    voxel_volume = float(simulation.grid.volume)

    columns: dict[str, list[Any]] = {
        key: []
        for key in (
            "wedge", "image", "angle", "dwd", "volume", "rde",
            "rde_angle", "weighted", "minimum",
        )
    }  # fmt: skip
    for number, (exposure, angles, wedge) in enumerate(
        zip(
            simulation.exposures,
            simulation.angles,
            simulation.experiment.wedges,
            strict=True,
        ),
        start=1,
    ):
        images = exposure.images
        assert images is not None
        weighted = _numpy(images.weighted_decay)
        fluence_total = _numpy(images.fluence_total)
        minimum = _numpy(images.minimum_decay)
        observations = _numpy(images.observations)
        dwd = _numpy(exposure.image_dwd)
        columns["rde"].extend(resolution_rde(dwd, wedge.max_resolution))

        running_min = 1.0
        for index, image_angle in enumerate(_numpy(angles)):
            # The Java's arithmetic, angle * (180 / PI), not math.degrees.
            degrees = float(image_angle) * (180 / math.pi)
            columns["wedge"].append(number)
            columns["image"].append(index + 1)
            columns["angle"].append(degrees)
            columns["dwd"].append(float(dwd[index]))
            # Lit voxels counted once per observation: for a pink beam that
            # is once per sampled energy, so RADDOSE-3D reports a volume that
            # many times too large. Reproduced.
            columns["volume"].append(int(observations[index]) * voxel_volume)
            if fluence_total[index] > 0:
                running_min = min(running_min, float(minimum[index]))
                columns["rde_angle"].append(degrees)
                columns["weighted"].append(
                    float(weighted[index] / fluence_total[index])
                )
                columns["minimum"].append(running_min)
            else:
                columns["rde_angle"].append(0.0)
                columns["weighted"].append(0.0)
                columns["minimum"].append(0.0)

    series = {key: np.asarray(value) for key, value in columns.items()}
    series["rde"] = series["rde"].reshape(-1, 5)
    return series


# --- the per-voxel and per-image files ----------------------------------------


def _floats(values: Iterable[float]) -> list[str]:
    """``Float.toString`` of each value, after the Java's ``(float)`` cast."""
    return [java_float(v) for v in values]


def _clamped(value: float) -> str:
    """A per-voxel value as the DoseState writers print it.

    Cast to single, and printed as a bare ``0`` -- not ``0.0`` -- if that is
    at or below ``Float.MIN_VALUE``, which is how ``OutputFinalDoseStateCSV``
    and ``OutputFinalDoseStateR`` spell an untouched voxel.
    """
    single = float(np.float32(value))
    return "0" if single <= _FLOAT_MIN_VALUE else java_float(single)


def dose_state_csv(simulation: Simulation) -> str:
    """``OutputFinalDoseStateCSV``: the final state of every crystal voxel.

    No header. One line per voxel inside the crystal, ``x,y,z,dose,fluence,
    elastic``, each printed as a Java ``float``.
    """
    state = final_dose_state(simulation)
    lines = []
    for position, dose, fluence, elastic in zip(
        state["coordinates"],
        state["dose"],
        state["fluence"],
        state["elastic"],
        strict=True,
    ):
        lines.append(
            ",".join(
                [
                    *_floats(position),
                    _clamped(dose),
                    _clamped(fluence),
                    _clamped(elastic),
                ]
            )
            + "\n"
        )
    return "".join(lines)


def dose_state_r(
    simulation: Simulation, generated: datetime.datetime | None = None
) -> str:
    """``OutputFinalDoseStateR``: an R script drawing the final dose.

    Parameters
    ----------
    simulation : Simulation
        From :func:`pyraddose_3d.solve.simulate`.
    generated : datetime.datetime, optional
        Time stamped in the header. Defaults to now, as in the Java.

    Returns
    -------
    str
        The script. It needs R's ``rgl`` and ``misc3d`` packages, and holds
        the whole voxel grid -- voxels outside the crystal as ``0`` -- in R's
        column-major order.
    """
    generated = generated or datetime.datetime.now()
    shape = simulation.grid.shape
    occupied = _numpy(simulation.occupied).astype(bool)
    dose = np.where(occupied, _numpy(simulation.exposures[-1].dose), 0.0)
    dose = dose.reshape(shape)

    nx, ny, nz = shape
    out = [
        "# Crystal dose state visualization using R\n",
        "# http://www.r-project.org/\n#\n",
        f"# Code generated {generated:%Y-%m-%d %H:%M:%S}\n",
        f"# Crystal size: {nx} x {ny} x {nz} voxels\n",
        "#\n\n",
        "contourlevels <- c(0.1, 20, 30) # MGy\n",
        "contourcolours <- c('lightblue', 'darkblue', 'red')\n",
        "contouropacity <- c(0.2, 0.5, 1)\n\n",
        'require("rgl")\n',
        'require("misc3d")\n\n',
        "# Three dimensional dose array (MGy)\n",
        f"dose <- array(0, c({nx}, {ny}, {nz}))\n",
    ]
    for k in range(nz):
        # i varies fastest, then j: R's column-major order.
        values = dose[:, :, k].ravel(order="F")
        out.append(f"dose[,,{k + 1}]<-c(" + ",".join(map(_clamped, values)) + ")\n")
    out += [
        "contour3d(dose, level=contourlevels, color=contourcolours, "
        "alpha=contouropacity)\n",
        "# axes3d()\n",
        f"wire3d(translate3d(scale3d(cube3d(),{nx}/2,{ny}/2,{nz}/2),"
        f"{nx}/2,{ny}/2,{nz}/2),col = 'grey')\n",
    ]
    views = ((0, 0), (0, 45), (45, 45), (90, 45), (90, 90))
    for number, (theta, phi) in enumerate(views):
        # The Java's spacing, which differs for the first snapshot only.
        gap = "" if number == 0 else " "
        out.append(f"rgl.viewpoint( theta = {theta}, phi = {phi})\n")
        out.append(
            f'rgl.snapshot({gap}"plot_{theta}_{phi}.png", fmt = "png", top = TRUE)\n'
        )
        if number < len(views) - 1:
            out.append("Sys.sleep(1)\n")
    out.append('print("Plots Saved")\n')
    return "".join(out)


def _voxel_series(simulation: Simulation, field: str) -> str:
    """``OutputVoxelDose`` / ``OutputVoxelFluences``.

    A first row of ``x_y_z`` labels, then one row per image of the last
    wedge, every value -- the last included -- followed by a comma.
    """
    series = voxel_images(simulation)
    labels = "".join("_".join(_floats(p)) + "," for p in series["coordinates"])
    rows = ["".join(v + "," for v in _floats(image)) for image in series[field]]
    return labels + "\n" + "".join(row + "\n" for row in rows)


def voxel_dose_csv(simulation: Simulation) -> str:
    """``OutputVoxelDose``: every voxel's dose in every image of the last wedge."""
    return _voxel_series(simulation, "dose")


def voxel_fluences_csv(simulation: Simulation) -> str:
    """``OutputVoxelFluences``: every voxel's fluence, likewise."""
    return _voxel_series(simulation, "fluence")


def dwds_csv(simulation: Simulation) -> str:
    """``OutputDWDs``: the diffraction-weighted dose after each image.

    The RDE columns are written 1, 2, 3 and 4 Angstrom first and the
    wedge's maximum resolution last, though the Java computes them the other
    way round.
    """
    series = image_series(simulation)
    lines = [
        "Wedge, RADDOSE Image Number, DWD Angle, DWD, Vol, "
        "1A RDE, 2A RDE, 3A RDE, 4A RDE, max res RDE\n"
    ]
    for wedge, image, angle, dwd, volume, rde in zip(
        series["wedge"],
        series["image"],
        series["angle"],
        series["dwd"],
        series["volume"],
        series["rde"],
        strict=True,
    ):
        fields = [str(wedge), str(image), *map(java_double, (angle, dwd, volume))]
        fields += [java_double(rde[column]) for column in (1, 2, 3, 4, 0)]
        lines.append(",".join(fields) + "\n")
    return "".join(lines)


def rde_csv(simulation: Simulation) -> str:
    """``OutputRDECSV``: the relative diffraction efficiency of each image."""
    series = image_series(simulation)
    lines = ["Wedge, Image Number, Angle, Avg RDE, min RDE\n"]
    for wedge, image, angle, weighted, minimum in zip(
        series["wedge"],
        series["image"],
        series["rde_angle"],
        series["weighted"],
        series["minimum"],
        strict=True,
    ):
        values = map(java_double, (angle, weighted, minimum))
        lines.append(",".join([str(wedge), str(image), *values]) + "\n")
    return "".join(lines)


# --- the summaries ------------------------------------------------------------

_SUMMARY_CSV_HEADER = (
    "Wedge Number, Average DWD, Last DWD,Elastic Yield (wedge), "
    "Diffraction Efficiency, AD-WC, AD-ExpRegion, Max Dose, Dose Threshold, "
    "Abs En Threshold, TAD, Dose Contrast, Used Volume, Wedge Absorbed Energy, "
    "Dose Inefficiency, Dose Inefficiency PE\n"
)


def summary_csv(simulation: Simulation) -> str:
    """``OutputSummaryCSV``: the per-wedge summary as a table.

    The header's missing space after "Last DWD," is the Java's.
    """
    lines = [_SUMMARY_CSV_HEADER]
    for summary in simulation.summaries:
        values = summary.as_dict()
        fields = [str(summary.wedge_number)]
        for key, value in list(values.items())[1:]:
            exponential = key in (
                "elastic_yield",
                "diffraction_efficiency",
                "wedge_absorbed_energy",
            )
            fields.append(java_e(value) if exponential else java_f(value))
        lines.append(", ".join(fields) + "\n")
    return "".join(lines)


def _rotation_line(crystal: Crystal) -> str:
    """The second line of a polyhedral crystal's description, if rotated."""
    # Through radians and back, as the Java stores them.
    plane = math.radians(crystal.angle_p)
    loop = math.radians(crystal.angle_l)
    if loop == 0 and plane == 0:
        return ""
    return (
        f"\nRotated by {java_f(math.degrees(plane), 1)} deg in the plane of the "
        f"loop and the loop is bent by {java_f(math.degrees(loop), 1)} relative "
        "to the rotation axis at phi = 0."
    )


def crystal_description(simulation: Simulation) -> str:
    """``Crystal.crystalInfo``: shape, size and voxel size.

    The size is the bounding box of the crystal *after* its P and L
    rotations, so a 20 x 16 um cuboid rotated 45 degrees reports 25 x 25.
    A sphere's diameter is the extent of its polyhedral approximation,
    slightly under the diameter asked for.
    """
    crystal = simulation.experiment.crystal
    vertices = _numpy(simulation.mesh.vertices)
    size = vertices.max(axis=0) - vertices.min(axis=0)
    voxel = java_f(1 / simulation.grid.pixels_per_micron, 2)

    if crystal.type == "cuboid":
        dims = ", ".join(java_f(s, 0) for s in size)
        return (
            f"Cuboid (Polyhedron) crystal of size [{dims}] um [x, y, z] at a "
            f"resolution of {voxel} microns per voxel edge." + _rotation_line(crystal)
        )
    if crystal.type == "cylinder":
        diameter = java_f(size[0] * _MICRONS_TO_MILLIMETRES, 2)
        height = java_f(size[1] * _MICRONS_TO_MILLIMETRES, 2)
        return (
            f"Cylinder (Polyhedron) crystal of diameter {diameter} mm and height "
            f"{height} mm at a resolution of {voxel} microns per voxel edge."
            + _rotation_line(crystal)
        )
    if crystal.type == "spherical":
        return (
            f"Spherical crystal of diameter {java_f(size[0], 0)} um, "
            f"at a resolution of {voxel} um per voxel edge."
        )
    dims = ", ".join(java_f(s, 2) for s in size)
    return (
        f"Polyhedron crystal of bounding size [{dims}] um [x, y, z] at a "
        f"resolution of {voxel} microns per voxel edge." + _rotation_line(crystal)
    )


def decay_description(crystal: Crystal) -> str:
    """The decay model's ``toString``."""
    if crystal.decay_model == "simple":
        return "Simple DDM."
    if crystal.decay_model == "linear":
        return (
            "Linear Dose Decay Model used. Diffraction intensity decreases "
            "linearly with dose according to ddmFactor = 1 - dose/(2*43) "
            "(Owen et al. 2006)"
        )
    if crystal.decay_model == "bfactor":
        return "Bfactor DDM."
    # DDMLeal reads (gamma, b0, beta) and uses zeros if any is missing.
    gamma, b_zero, beta = crystal.decay_parameters or (0.0, 0.0, 0.0)
    rows = (
        ("Gamma", gamma, "MGy^{-1}"),
        ("B0", b_zero, "Angstroms^2"),
        ("Beta", beta, "Angstroms^2 MGy^{-1}"),
    )
    return (
        "Dose Decay Model from Leal et al. 2012 is being used.\n"
        "Using decay parameter values:\n"
        + "\n".join(
            f"{name:>5}{'=':>2}{java_f(value, 3):>7} {unit}"
            for name, value, unit in rows
        )
    )


def beam_description(beam: Beam) -> str:
    """The beam's ``getDescription``, newline included.

    The flux is the one asked for, before any container attenuates it.
    ``BeamTophat`` concatenates raw doubles, so it reads ``2.0E12`` where
    the Gaussian reads ``2.0e+12``.
    """
    if beam.type == "tophat":
        assert beam.collimation is not None
        return (
            f"Top hat beam, {java_double(beam.collimation.x)} by "
            f"{java_double(beam.collimation.y)} (x by y) um with "
            f"{java_double(beam.flux)} photons per second at "
            f"{java_double(beam.energy)}keV.\n"
        )
    collimation = ""
    if beam.collimation is not None:
        collimation = (
            f"{java_f(beam.collimation.x, 1)}x{java_f(beam.collimation.y, 1)} um "
        )
    return (
        f"Gaussian beam, {collimation}with {java_f(beam.fwhm[0], 2)} by "
        f"{java_f(beam.fwhm[1], 2)} FWHM (x by y) and {java_e(beam.flux, 1)} "
        f"photons per second at {java_f(beam.energy, 2)} keV.\n"
    )


def wedge_description(wedge: Wedge) -> str:
    """``Wedge.wedgeProperties``: exposure, angles and any translation.

    Angles go through radians and back, and translations through degrees and
    back, because the Java stores them that way.
    """
    start = math.degrees(math.radians(wedge.start))
    end = math.degrees(math.radians(wedge.end))
    text = (
        f"Collecting data for a total of {java_f(wedge.exposure_time, 1)}s from "
        f"phi = {java_f(start, 1)} to {java_f(end, 1)} deg.\n"
    )
    offset = wedge.start_offset or (0.0, 0.0, 0.0)
    translation = wedge.translate_per_degree or (0.0, 0.0, 0.0)
    if any(v != 0 for v in (*offset, *translation)):
        per_degree = [math.radians(math.degrees(v)) for v in translation]
        text += "Start is offset by [{}] um [x,y,z].\n".format(
            ", ".join(java_f(v) for v in offset)
        ) + "Helical scanning is at [{}] um/deg in [x,y,z]\n".format(
            ", ".join(java_f(v) for v in per_degree)
        )
    return text


def coefficients_description(simulation: Simulation) -> str:
    """The composition's ``toString``: coefficients at the beam energy."""
    coefficients = simulation.coefficients
    absorption, attenuation, elastic, inelastic = (
        java_e(_float(value), 2)
        for value in (
            coefficients.absorption,
            coefficients.attenuation,
            coefficients.elastic,
            coefficients.inelastic,
        )
    )
    density = java_f(_float(simulation.density), 2)
    if simulation.experiment.crystal.composition.method == "average":
        return (
            "Dummy crystal coefficients used.\n"
            f"Absorption coefficient: {absorption} /um.\n"
            f"Attenuation coefficient: {attenuation} /um.\n"
            f"Elastic coefficient: {elastic} /um.\n"
            f"Density: {density} g/ml.\n"
        )
    # "Photelectric" is the Java's spelling.
    return (
        "\n"
        "Crystal coefficients calculated with RADDOSE-3D. \n"
        f"Photelectric Coefficient: {absorption} /um.\n"
        f"Inelastic Coefficient: {inelastic} /um.\n"
        f"Elastic Coefficient: {elastic} /um.\n"
        f"Attenuation Coefficient: {attenuation} /um.\n"
        f"Density: {density} g/ml.\n"
    )


def dose_histogram(doses: np.ndarray) -> list[float]:
    """``Histogram``'s normalised counts of the doses above zero.

    Eleven bins: below 0.1 MGy, nine equal bins up to 30, and 30 upwards.
    The bin index is truncated toward zero, as the Java's ``(int)`` cast.
    """
    step = (_HISTOGRAM_MAX - _HISTOGRAM_MIN) / _HISTOGRAM_BINS
    counts = [0] * (_HISTOGRAM_BINS + 2)
    observed = [float(d) for d in doses if d > 0]
    for dose in observed:
        if dose < _HISTOGRAM_MIN:
            counts[0] += 1
        elif dose >= _HISTOGRAM_MAX:
            counts[-1] += 1
        else:
            counts[1 + int((dose - _HISTOGRAM_MIN) / step)] += 1
    total = len(observed)
    return [count / total if total else math.nan for count in counts]


def _histogram_lines(doses: np.ndarray) -> str:
    step = (_HISTOGRAM_MAX - _HISTOGRAM_MIN) / _HISTOGRAM_BINS
    breaks = [-math.inf] + [
        _HISTOGRAM_MIN + (i - 1) * step for i in range(1, _HISTOGRAM_BINS + 2)
    ]
    fraction = dose_histogram(doses)
    bins = len(fraction)

    def pct(value: float) -> str:
        return f"{java_f(value * 100, 1):>4}"

    def edge(value: float) -> str:
        return f"{java_f(value, 1):>4}"

    lines = [
        "Final Dose Histogram:\n",
        f"Bin  1,  0.0 to {edge(breaks[1])} MGy: {pct(fraction[0])} % \n",
    ]
    for i in range(1, bins - 1):
        lines.append(
            f"Bin {i + 1:2d}, {edge(breaks[i])} to {edge(breaks[i + 1])} MGy: "
            f"{pct(fraction[i])} % \n"
        )
    lines.append(
        f"Bin {bins:2d}, {edge(breaks[bins - 1])} MGy upwards: "
        f"{pct(fraction[bins - 1])} %\n"
    )
    return "".join(lines)


def summary_text(simulation: Simulation) -> str:
    """``OutputSummaryText``: what RADDOSE-3D prints and saves as Summary.txt.

    The 95% threshold label is longer than the 42-character column it is
    padded to, and the Java lets it overflow; so does this.
    """
    experiment = simulation.experiment
    occupied = _numpy(simulation.occupied).astype(bool)

    def row(label: str, value: str) -> str:
        return f"{label:<42}: {value}\n"

    out = [
        crystal_description(simulation) + "\n",
        decay_description(experiment.crystal) + "\n",
        beam_description(experiment.beam),
    ]
    for wedge, summary, exposure in zip(
        experiment.wedges, simulation.summaries, simulation.exposures, strict=True
    ):
        values = summary.as_dict()
        threshold = values["abs_en_threshold"]
        out.append(f"Wedge {summary.wedge_number}:\n")
        out.append(
            wedge_description(wedge) + coefficients_description(simulation) + "\n"
        )
        out += [
            row(
                "Average Diffraction Weighted Dose",
                f"{java_f(values['average_dwd'])} MGy",
            ),
            row("Last Diffraction Weighted Dose", f"{java_f(values['last_dwd'])} MGy"),
            row("Elastic Yield", f"{java_e(values['elastic_yield'], 2)} photons"),
            row(
                "Diffraction Efficiency (Elastic Yield/DWD)",
                f"{java_e(values['diffraction_efficiency'], 2)} photons/MGy",
            ),
            row("Average Dose (Whole Crystal)", f"{java_f(values['ad_wc'])} MGy"),
            row(
                "Average Dose (Exposed Region)",
                f"{java_f(values['ad_exp_region'])} MGy",
            ),
            row("Max Dose", f"{java_f(values['max_dose'])} MGy"),
            row(
                f"Average Dose ({java_f(threshold, 1)} % of total absorbed energy "
                f"threshold ({java_f(values['dose_threshold'], 2)} MGy))",
                f"{java_f(values['tad'])} MGy",
            ),
            row(
                "Dose Contrast (Max/Threshold Av.)", java_f(values["dose_contrast"], 2)
            ),
            row("Used Volume", f"{java_f(values['used_volume'], 1)}%"),
            row(
                "Absorbed Energy (this Wedge)",
                f"{java_e(values['wedge_absorbed_energy'], 2)} J.",
            ),
            row(
                "Dose Inefficiency (Max Dose/mJ Absorbed)",
                f"{java_f(values['dose_inefficiency'], 1)} 1/g",
            ),
            row(
                "Dose Inefficiency PE (Max Dose/mJ Deposited)",
                f"{java_f(values['dose_inefficiency_pe'], 1)} 1/g",
            ),
        ]
        out.append(_histogram_lines(_numpy(exposure.dose)[occupied]))
    return "".join(out)


# --- all of them ------------------------------------------------------------------


def write_outputs(simulation: Simulation, prefix: str = "output-") -> list[Path]:
    """Write every file RADDOSE-3D writes by default.

    Parameters
    ----------
    simulation : Simulation
        From :func:`pyraddose_3d.solve.simulate`. Run it with
        ``record_images=True`` for the per-image files; without, they are
        skipped.
    prefix : str
        Prepended to each file name, as RADDOSE-3D's ``--prefix``; it may
        include a directory, which must exist.

    Returns
    -------
    list[Path]
        The files written.
    """
    writers: dict[str, Callable[[Simulation], str]] = {
        "Summary.csv": summary_csv,
        "Summary.txt": summary_text,
        "DoseState.csv": dose_state_csv,
        "DoseState.R": dose_state_r,
    }
    if all(e.images is not None for e in simulation.exposures):
        writers |= {
            "RDE.csv": rde_csv,
            "DWDs.csv": dwds_csv,
            "VoxelDose.csv": voxel_dose_csv,
            "VoxelFluences.csv": voxel_fluences_csv,
        }
    written = []
    for name, writer in writers.items():
        path = Path(f"{prefix}{name}")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(writer(simulation))
        written.append(path)
    return written
