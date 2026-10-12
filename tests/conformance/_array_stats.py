"""Reduce RADDOSE-3D's per-voxel and per-image outputs to comparable statistics.

The Java writes five files the sixteen summary numbers are averaged from --
the final dose state, the dose and fluence of every voxel after every image,
and two per-image series. They run to megabytes, so the corpus commits
statistics of them instead of the files.

Both sides of the comparison go through the reducers here: ``generate.py``
feeds them the Java's CSVs, the test feeds them the port's arrays. A
reduction implemented twice would drift, and the drift would look exactly
like a port error.

What each statistic is for
--------------------------
The summary numbers are all averages, sums and extremes over the crystal, so
they cannot see a dose that is right in total but in the wrong place, or
right at the end but wrong along the way. The statistics here are chosen to
see exactly those:

``DoseState``
    The final field. The dose-weighted centroid and spread are the only
    quantities in the corpus that depend on *where* the dose is: mirror the
    whole field and every summary number is unchanged, but the centroid
    moves.
``VoxelDose`` / ``VoxelFluences``
    Per image, the sum, maximum and number of exposed voxels. Catches an
    error in *when* dose was deposited that the final total averages away.
``DWDs`` / ``RDE``
    Per-image series written at full precision, compared element by element.

Earlier versions of the corpus flattened every numeric field of each file
into one list before taking statistics, so ``DoseState`` reported a minimum
of -10 (a coordinate) and a maximum of 7.3e6 (a fluence). Each column is now
reduced on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.outputs import final_dose_state, image_series, voxel_images

if TYPE_CHECKING:
    from pyraddose_3d.solve import Simulation

__all__ = [
    "DEFERRED",
    "compare",
    "dose_state",
    "from_simulation",
    "image_rows",
    "per_image_series",
    "tolerance_for",
]

#: Quantities the Java writes that the port does not produce yet, with why.
#: The comparison fails if one of these turns up on the port side, so a
#: deferral cannot outlive the gap it names. Empty since the output writers
#: landed; kept so the next gap has somewhere to go.
DEFERRED: dict[tuple[str, str], str] = {}

_QUANTILES = {"p05": 0.05, "median": 0.5, "p95": 0.95}


def _distribution(values: np.ndarray) -> dict[str, float]:
    """Sum, mean, extremes and three quantiles of one column."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = len(ordered)
    stats = {
        "sum": float(ordered.sum()),
        "mean": float(ordered.mean()),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }
    for name, fraction in _QUANTILES.items():
        stats[name] = float(ordered[min(int(fraction * count), count - 1)])
    return stats


def dose_state(
    coordinates: np.ndarray, dose: np.ndarray, fluence: np.ndarray, elastic: np.ndarray
) -> dict[str, Any]:
    """Reduce the final state of every voxel inside the crystal.

    Parameters
    ----------
    coordinates : np.ndarray
        ``(M, 3)`` voxel centres in micrometres.
    dose, fluence, elastic : np.ndarray
        ``(M,)`` cumulative over every wedge.

    Returns
    -------
    dict[str, Any]
        Per-column distributions, and the dose-weighted centroid and spread
        along each axis in micrometres.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    dose = np.asarray(dose, dtype=np.float64)
    weights = dose / dose.sum()
    centroid = (coordinates * weights[:, None]).sum(axis=0)
    spread = np.sqrt(((coordinates - centroid) ** 2 * weights[:, None]).sum(axis=0))
    return {
        "voxels": len(dose),
        "dose": _distribution(dose),
        "fluence": _distribution(fluence),
        "elastic": _distribution(elastic),
        "dose_centroid": centroid.tolist(),
        "dose_spread": spread.tolist(),
    }


def image_rows(rows: np.ndarray) -> dict[str, Any]:
    """Reduce an ``(images, voxels)`` table, one statistic per image.

    Parameters
    ----------
    rows : np.ndarray
        ``(A, M)``; a voxel that was not exposed in an image is zero in that
        row, as the Java writes it.

    Returns
    -------
    dict[str, Any]
        The table's shape, and per image the sum, maximum and number of
        non-zero entries.
    """
    rows = np.asarray(rows, dtype=np.float64)
    return {
        "images": int(rows.shape[0]),
        "voxels": int(rows.shape[1]),
        "sum": rows.sum(axis=1).tolist(),
        "max": rows.max(axis=1).tolist(),
        "nonzero": (rows != 0).sum(axis=1).tolist(),
    }


def per_image_series(**columns: Any) -> dict[str, list[float]]:
    """Package per-image columns as plain lists, compared element by element."""
    return {name: [float(v) for v in values] for name, values in columns.items()}


# --- the port's side --------------------------------------------------------


def _as_java_float(values: Any) -> np.ndarray:
    """Round to single precision, as the Java does before writing.

    ``DoseState``, ``VoxelDose`` and ``VoxelFluences`` are written as Java
    ``float``. That costs nothing at the tolerances used here, but it decides
    which values come out as exactly zero, and the per-image non-zero count
    depends on that.
    """
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def from_simulation(simulation: Simulation) -> dict[str, Any]:
    """Reduce a port run to the statistics the Java's files reduce to.

    Parameters
    ----------
    simulation : Simulation
        From :func:`pyraddose_3d.solve.simulate` with ``record_images=True``.

    Returns
    -------
    dict[str, Any]
        Keyed as the corpus's ``array_stats.json``, minus :data:`DEFERRED`.

    Notes
    -----
    Which values each Java writer chooses, and how it masks them, is defined
    once, in the tables :mod:`pyraddose_3d.outputs` writes its files from --
    getting that wrong would look like a physics error. This reduces those
    tables without going through text; ``test_output_files.py`` checks that the
    written files reduce to the same thing.
    """
    state = final_dose_state(simulation)
    stats: dict[str, Any] = {
        "DoseState": dose_state(
            _as_java_float(state["coordinates"]),
            _as_java_float(state["dose"]),
            _as_java_float(state["fluence"]),
            _as_java_float(state["elastic"]),
        )
    }

    voxels = voxel_images(simulation)
    stats["VoxelDose"] = image_rows(_as_java_float(voxels["dose"]))
    stats["VoxelFluences"] = image_rows(_as_java_float(voxels["fluence"]))

    series = image_series(simulation)
    rde = series["rde"]
    stats["DWDs"] = per_image_series(
        wedge=series["wedge"],
        angle=series["angle"],
        dwd=series["dwd"],
        volume=series["volume"],
        rde_1A=rde[:, 1],
        rde_2A=rde[:, 2],
        rde_3A=rde[:, 3],
        rde_4A=rde[:, 4],
        rde_max_res=rde[:, 0],
    )
    stats["RDE"] = per_image_series(
        wedge=series["wedge"],
        angle=series["rde_angle"],
        weighted=series["weighted"],
        minimum=series["minimum"],
    )
    return stats


# --- comparison -------------------------------------------------------------

#: Statistics compared exactly: counts of voxels, images and exposed voxels,
#: and the wedge each image belongs to.
_COUNTS = frozenset({"voxels", "images", "nonzero", "wedge"})

#: DoseState statistics that average over the crystal, as opposed to ones
#: that land on a single voxel -- which is usually a voxel on a face.
_TOTALS = frozenset({"sum", "mean", "max"})


def tolerance_for(
    path: tuple[str, ...], tolerances: dict[str, float]
) -> tuple[str, float]:
    """How a leaf is compared, and how closely.

    Parameters
    ----------
    path : tuple[str, ...]
        Location of the leaf, e.g. ``("DoseState", "dose", "p05")``.
    tolerances : dict[str, float]
        ``array_tolerances`` from the manifest.

    Returns
    -------
    tuple[str, float]
        ``"relative"`` or ``"absolute"``, and the limit.

    Raises
    ------
    KeyError
        If the manifest has no tolerance for this statistic. Deliberate: a
        new statistic must be given a measured tolerance, not inherit one.
    """
    entry, statistic = path[0], path[-1]
    if statistic in _COUNTS:
        return "absolute", tolerances["counts"]
    if statistic == "angle":
        return "absolute", tolerances["angle_degrees"]
    if entry == "DoseState":
        if statistic in ("dose_centroid", "dose_spread"):
            return "absolute", tolerances["DoseState/position_um"]
        family = "totals" if statistic in _TOTALS else "extremes"
        return "relative", tolerances[f"DoseState/{family}"]
    if statistic.startswith("rde_"):
        # A fraction in [0, 1] that reaches exp(-40) on absorbing crystals,
        # where a relative comparison would weigh noise in the last bits.
        return "absolute", tolerances["DWDs/rde"]
    return "relative", tolerances[f"{entry}/{statistic}"]


def _deviation(kind: str, expected: float, actual: float) -> float:
    """Deviation in the units its tolerance is stated in."""
    if kind == "relative":
        scale = max(abs(expected), abs(actual))
        return 0.0 if scale == 0 else abs(actual - expected) / scale
    return abs(actual - expected)


def _leaves(tree: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Flatten nested dicts to (path, value) pairs; lists stay whole."""
    if isinstance(tree, dict):
        out = []
        for key, value in tree.items():
            out.extend(_leaves(value, (*path, key)))
        return out
    return [(path, tree)]


def compare(
    expected: dict[str, Any], actual: dict[str, Any], tolerances: dict[str, float]
) -> tuple[list[str], list[tuple[str, float, float]]]:
    """Compare two reductions leaf by leaf.

    Parameters
    ----------
    expected, actual : dict[str, Any]
        From the corpus and from :func:`from_simulation`.
    tolerances : dict[str, float]
        ``array_tolerances`` from the manifest; see :func:`tolerance_for`.

    Returns
    -------
    tuple[list[str], list[tuple[str, float, float]]]
        Failure messages, and every compared leaf as
        ``(path, worst deviation, limit)`` so the achieved agreement can be
        reported rather than just pass or fail.
    """
    failures: list[str] = []
    achieved: list[tuple[str, float, float]] = []
    actual_leaves = dict(_leaves(actual))

    for path, want in _leaves(expected):
        label = "/".join(path)
        key = (path[0], path[-1])
        if path not in actual_leaves:
            if key not in DEFERRED:
                failures.append(f"{label}: the port does not produce it")
            continue
        if key in DEFERRED:
            failures.append(
                f"{label}: the port now produces this, so take it out of DEFERRED"
            )
            continue

        got = actual_leaves[path]
        kind, limit = tolerance_for(path, tolerances)
        wants = want if isinstance(want, list) else [want]
        gots = got if isinstance(got, list) else [got]
        if len(wants) != len(gots):
            failures.append(f"{label}: {len(wants)} entries expected, got {len(gots)}")
            continue

        deviations = [_deviation(kind, e, a) for e, a in zip(wants, gots, strict=True)]
        worst = max(deviations, default=0.0)
        if worst > limit:
            at = deviations.index(worst)
            where = f"[{at}]" if isinstance(want, list) else ""
            failures.append(
                f"{label}{where}: expected {wants[at]!r}, got {gots[at]!r} "
                f"({kind} {worst:.3e} > {limit:.1e})"
            )
        achieved.append((label, worst, limit))

    return failures, achieved
