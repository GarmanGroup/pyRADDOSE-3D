"""The sixteen numbers RADDOSE-3D reports for a wedge.

Ports ``ExposureSummary`` and the column set of ``OutputSummaryCSV``. Where
:mod:`pyraddose_3d.backend.exposure` computes a dose per voxel, this reduces
that field to the quantities a user actually reads.

Most are straightforward reductions. Three are not, and are worth knowing
before reading the code:

**Diffraction-weighted dose** is not the average dose. It weights each voxel
by how much it *diffracted* -- its fluence, times its relative diffraction
efficiency -- so a voxel sitting in the beam's tail counts for little however
much dose it took. It is accumulated per image by the exposure loop, because
it depends on the dose history, and only averaged here.

**The dose threshold** is an iso-surface, not a percentile of voxels. It is
the dose above which a given fraction of the *total absorbed dose* lies. With
the default of 0.95, 5% of the dose is in voxels below the threshold.

**Dose inefficiency** is peak dose per unit energy absorbed: how much damage
the worst-off part of the crystal takes for each joule put in. Lower is
better. It exists to compare strategies, so its absolute scale matters less
than the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend._array import array_namespace

if TYPE_CHECKING:
    from pyraddose_3d.backend.exposure import ExposureResult

__all__ = ["WedgeSummary", "summarise"]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""

#: Fraction of the total dose that must lie above the reported threshold.
DEFAULT_ABSORBED_ENERGY_THRESHOLD = 0.95

#: Joules per megagray, per kilogram. Dose in MGy times mass in kg times
#: this is energy in joules.
_MEGAGRAY_TO_JOULES_PER_KG = 1e6


@dataclass(frozen=True)
class WedgeSummary:
    """The summary line RADDOSE-3D writes for one wedge.

    Field names follow the CSV columns, lower-cased. Doses are in MGy,
    energies in joules, volumes as percentages.
    """

    wedge_number: int
    average_dwd: Array
    last_dwd: Array
    elastic_yield: Array
    diffraction_efficiency: Array
    ad_wc: Array
    ad_exp_region: Array
    max_dose: Array
    dose_threshold: Array
    abs_en_threshold: float
    tad: Array
    dose_contrast: Array
    used_volume: Array
    wedge_absorbed_energy: Array
    dose_inefficiency: Array
    dose_inefficiency_pe: Array

    def as_dict(self) -> dict[str, float]:
        """The summary as plain floats, in CSV column order.

        Returns
        -------
        dict[str, float]
            Suitable for writing to JSON or comparing against a reference.
        """
        return {
            "wedge_number": float(self.wedge_number),
            "average_dwd": float(self.average_dwd),
            "last_dwd": float(self.last_dwd),
            "elastic_yield": float(self.elastic_yield),
            "diffraction_efficiency": float(self.diffraction_efficiency),
            "ad_wc": float(self.ad_wc),
            "ad_exp_region": float(self.ad_exp_region),
            "max_dose": float(self.max_dose),
            "dose_threshold": float(self.dose_threshold),
            "abs_en_threshold": float(self.abs_en_threshold),
            "tad": float(self.tad),
            "dose_contrast": float(self.dose_contrast),
            "used_volume": float(self.used_volume),
            "wedge_absorbed_energy": float(self.wedge_absorbed_energy),
            "dose_inefficiency": float(self.dose_inefficiency),
            "dose_inefficiency_pe": float(self.dose_inefficiency_pe),
        }


def _threshold_dose(
    doses: Array, total_dose: Array, quantile: float
) -> tuple[Array, Array]:
    """Find the iso-surface enclosing a given fraction of the total dose.

    Walks the voxel doses in ascending order, accumulating until the dose
    left above reaches ``quantile`` of the total. The dose at which that
    happens is the threshold, and everything strictly after it is counted.

    Parameters
    ----------
    doses : Array
        ``(M,)`` doses of the voxels that received any, in any order.
    total_dose : Array
        Their sum.
    quantile : float
        Fraction of the total dose to lie above the threshold.

    Returns
    -------
    tuple[Array, Array]
        The threshold dose and the number of voxels above it.

    Notes
    -----
    Reproduces ``getAbsDoseThreshold``, which walks a
    ``NavigableMap<Double, Integer>`` -- distinct doses in ascending order,
    each with a count. Voxels sharing a dose are therefore all on the same
    side of the threshold, which matters because a coarse voxel grid
    produces many exact ties.

    The map is seeded with a single zero-dose entry so that the maximum is
    defined even for an exposure that missed the crystal entirely. That
    entry never affects the threshold, since it contributes no dose.
    """
    xp = array_namespace(doses)
    order = xp.argsort(doses)
    ordered = doses[order]

    # Distinct doses, ascending, with how many voxels share each.
    unique, counts = np.unique(np.asarray(ordered), return_counts=True)
    contribution = unique * counts
    cumulative = np.cumsum(contribution)

    cutoff = (1 - quantile) * float(total_dose)
    # The first distinct dose at which the running total reaches the cutoff.
    reached = np.searchsorted(cumulative, cutoff, side="left")
    if reached >= len(unique):
        return float("inf"), 0

    threshold = unique[reached]
    above = int(counts[reached + 1 :].sum())
    return threshold, above


def summarise(
    result: ExposureResult,
    voxel_volume: float,
    density: Array,
    wedge_number: int = 1,
    quantile: float = DEFAULT_ABSORBED_ENERGY_THRESHOLD,
) -> WedgeSummary:
    """Reduce an exposure to the sixteen reported numbers.

    Parameters
    ----------
    result : ExposureResult
        Output of :func:`pyraddose_3d.backend.exposure.expose`.
    voxel_volume : float
        Volume of one voxel in cubic micrometres.
    density : Array
        Crystal density in g/cm^3.
    wedge_number : int, optional
        Which wedge this is, counting from one.
    quantile : float, optional
        Fraction of the total dose that must lie above the reported
        threshold. RADDOSE-3D's default is 0.95.

    Returns
    -------
    WedgeSummary
        The summary line.
    """
    xp = array_namespace(result.dose)

    occupied = result.occupied
    doses = xp.where(occupied, result.dose, 0.0)
    exposed = occupied & (result.dose > 0)

    occupied_count = xp.sum(occupied)
    exposed_count = xp.sum(exposed)
    total_dose = xp.sum(xp.where(exposed, result.dose, 0.0))
    max_dose = xp.max(doses)

    # A voxel's mass: its volume in um^3, as cm^3, times density in g/cm^3,
    # as kg.
    voxel_mass_kg = voxel_volume * 1e-12 * density * 1e-3
    total_energy = total_dose * _MEGAGRAY_TO_JOULES_PER_KG * voxel_mass_kg

    absorbed = xp.sum(xp.where(occupied, result.absorbed_energy, 0.0))
    elastic = xp.sum(xp.where(occupied, result.elastic, 0.0))

    exposed_doses = np.asarray(xp.where(exposed, result.dose, 0.0))
    exposed_doses = exposed_doses[exposed_doses > 0]
    threshold, above = _threshold_dose(exposed_doses, total_dose, quantile)
    average_above = quantile * total_dose / above if above else float("inf")

    average_dwd = xp.sum(result.image_dwd) / len(result.image_dwd)
    last_dwd = result.image_dwd[-1] if len(result.image_dwd) else 0.0

    return WedgeSummary(
        wedge_number=wedge_number,
        average_dwd=average_dwd,
        last_dwd=last_dwd,
        elastic_yield=elastic,
        diffraction_efficiency=elastic / average_dwd,
        ad_wc=total_dose / occupied_count,
        ad_exp_region=total_dose / exposed_count,
        max_dose=max_dose,
        dose_threshold=threshold,
        abs_en_threshold=quantile * 100,
        tad=average_above,
        dose_contrast=max_dose / average_above,
        used_volume=100.0 * exposed_count / occupied_count,
        wedge_absorbed_energy=absorbed,
        dose_inefficiency=(max_dose * 1e6) / (1000 * absorbed),
        dose_inefficiency_pe=(max_dose * 1e6) / (1000 * total_energy),
    )
