"""Tests for the sixteen numbers RADDOSE-3D reports for a wedge.

These are what a user actually reads, so a mistake here is more visible than
one in the dose field itself -- and easier to make, because several of the
quantities are weighted or thresholded in ways their names do not suggest.
"""

import csv
import math
import pathlib

import numpy as np
import pytest
from test_exposure import CASE_IDS, _run

from pyraddose_3d.backend.summary import (
    DEFAULT_ABSORBED_ENERGY_THRESHOLD,
    summarise,
)

REFERENCE = pathlib.Path(__file__).parent / "data" / "exposure.tsv"

#: The CSV columns, in the order ``OutputSummaryCSV`` writes them, minus the
#: wedge number that leads the line.
FIELDS = [
    "average_dwd", "last_dwd", "elastic_yield", "diffraction_efficiency",
    "ad_wc", "ad_exp_region", "max_dose", "dose_threshold",
    "abs_en_threshold", "tad", "dose_contrast", "used_volume",
    "wedge_absorbed_energy", "dose_inefficiency", "dose_inefficiency_pe",
]


def _reference_reports() -> dict[str, dict[str, float]]:
    reports = {}
    with REFERENCE.open(encoding="ascii") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row[0] == "REPORT":
                reports[row[1]] = dict(
                    zip(FIELDS, [float(v) for v in row[2:17]], strict=True)
                )
    return reports


REPORTS = _reference_reports()


def _summarise(case_id: str):
    case, grid, coordinates, _occupied, result = _run(case_id)
    return summarise(result, grid.volume, case["density"]), coordinates, result


# --- conformance against the Java ---------------------------------------


def test_reference_reports_every_case() -> None:
    assert set(REPORTS) == set(CASE_IDS)
    for report in REPORTS.values():
        assert set(report) == set(FIELDS)


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_summary_matches_java(case_id: str) -> None:
    """All sixteen fields, at 1e-4.

    Four of the five cases come out at 1e-13 or better. ``rotation-gauss``
    reaches 1.8e-5, which is the same on-face voxel instability measured in
    ``test_exposure.py`` -- not a different formula, the same doses being
    summarised.
    """
    summary, _, _ = _summarise(case_id)
    got = summary.as_dict()
    want = REPORTS[case_id]

    for field in FIELDS:
        assert got[field] == pytest.approx(want[field], rel=1e-4), field


@pytest.mark.conformance
@pytest.mark.parametrize("case_id", ["still-tophat", "collimated", "offset"])
def test_well_conditioned_cases_match_closely(case_id: str) -> None:
    """The cases without the on-face problem are held to 1e-11.

    A single loose tolerance across all five would let a real error in the
    threshold or weighting hide behind the one ill-conditioned case.
    """
    summary, _, _ = _summarise(case_id)
    got = summary.as_dict()
    want = REPORTS[case_id]

    for field in FIELDS:
        assert got[field] == pytest.approx(want[field], rel=1e-11), field


# --- the quantities that are not what their names suggest ---------------


def test_diffraction_weighted_dose_is_not_the_average_dose() -> None:
    """DWD weights each voxel by how much it diffracted, not by volume.

    In a Gaussian beam the well-lit centre dominates, so the DWD sits above
    the plain average over the crystal. Confusing the two would look
    plausible and be wrong by tens of percent.
    """
    summary, _, _ = _summarise("rotation-gauss")

    assert summary.average_dwd != pytest.approx(summary.ad_wc, rel=1e-6)
    assert float(summary.average_dwd) < float(summary.ad_wc)


def test_the_dose_threshold_encloses_the_right_fraction_of_the_dose() -> None:
    """95% of the *dose* lies above the threshold, not 95% of the voxels.

    Checked by summing the dose above it directly, on a Gaussian exposure
    where the dose varies smoothly enough for the threshold to land near
    its ideal position.
    """
    summary, _, result = _summarise("rotation-gauss")

    dose = np.asarray(result.dose)[np.asarray(result.occupied)]
    dose = dose[dose > 0]
    above = dose[dose > float(summary.dose_threshold)].sum()

    assert above / dose.sum() == pytest.approx(
        DEFAULT_ABSORBED_ENERGY_THRESHOLD, abs=0.01
    )
    # The threshold sits inside the range rather than at either end, which
    # is what makes it an iso-surface rather than a degenerate answer.
    assert dose.min() < float(summary.dose_threshold) < dose.max()


def test_a_quantised_dose_distribution_overshoots_the_threshold() -> None:
    """The threshold can only land on a dose that some voxel actually has.

    A uniform beam gives a handful of distinct doses -- ten, across three
    thousand voxels, for the tophat case -- and the lowest of them already
    carries more than the 5% that is meant to fall below the threshold. The
    threshold therefore lands on the minimum and *all* the dose is above it.

    That is the algorithm behaving correctly on a coarse distribution, not a
    failure, but it means the reported threshold is not always the
    iso-surface it is described as. Worth knowing before reading one.
    """
    summary, _, result = _summarise("still-tophat")

    dose = np.asarray(result.dose)[np.asarray(result.occupied)]
    dose = dose[dose > 0]

    assert len(np.unique(dose)) < 20, "this case should be coarsely quantised"
    assert float(summary.dose_threshold) == pytest.approx(dose.min())
    assert dose[dose >= float(summary.dose_threshold)].sum() == pytest.approx(
        dose.sum()
    )


def test_threshold_average_dose_lies_between_the_threshold_and_the_peak() -> None:
    summary, _, _ = _summarise("rotation-gauss")

    assert float(summary.dose_threshold) <= float(summary.tad)
    assert float(summary.tad) <= float(summary.max_dose)


def test_dose_contrast_is_peak_over_threshold_average() -> None:
    summary, _, _ = _summarise("rotation-gauss")

    assert float(summary.dose_contrast) == pytest.approx(
        float(summary.max_dose) / float(summary.tad), rel=1e-12
    )
    assert float(summary.dose_contrast) >= 1.0


def test_a_uniform_beam_has_less_dose_contrast_than_a_gaussian() -> None:
    """The quantity exists to say how evenly a crystal was dosed, so it had
    better order two obviously different exposures correctly.
    """
    tophat, _, _ = _summarise("still-tophat")
    gaussian, _, _ = _summarise("rotation-gauss")

    assert float(tophat.dose_contrast) < float(gaussian.dose_contrast)


def test_used_volume_is_a_percentage_of_the_crystal() -> None:
    summary, _, result = _summarise("collimated")

    occupied = int(np.sum(np.asarray(result.occupied)))
    exposed = int(
        np.sum(np.asarray(result.occupied) & (np.asarray(result.dose) > 0))
    )
    assert float(summary.used_volume) == pytest.approx(
        100.0 * exposed / occupied, rel=1e-12
    )
    assert 0 < float(summary.used_volume) <= 100.0


def test_average_over_the_exposed_region_exceeds_the_whole_crystal() -> None:
    """Unless every voxel was exposed, in which case they are equal. Dividing
    the same total by a smaller count cannot give less.
    """
    for case_id in CASE_IDS:
        summary, _, _ = _summarise(case_id)
        assert float(summary.ad_exp_region) >= float(summary.ad_wc)


def test_dose_inefficiency_falls_when_the_beam_is_better_matched() -> None:
    """Peak dose per joule absorbed. A collimated beam concentrates dose in
    the crystal, so it should look *worse* by this measure than one whose
    tails spill past it -- the number rewards spreading dose evenly, not
    delivering more of it.
    """
    collimated, _, _ = _summarise("collimated")
    uncollimated, _, _ = _summarise("rotation-gauss")

    assert float(collimated.dose_inefficiency) > 0
    assert float(uncollimated.dose_inefficiency) > 0
    assert float(collimated.dose_inefficiency) != pytest.approx(
        float(uncollimated.dose_inefficiency), rel=1e-6
    )


def test_the_threshold_percentage_is_reported_as_given() -> None:
    """``abs_en_threshold`` is an input echoed back, not a result. It is in
    the CSV so a reader knows which quantile the two columns beside it used.
    """
    summary, _, _ = _summarise("still-tophat")
    assert float(summary.abs_en_threshold) == 95.0


def test_a_different_quantile_moves_the_threshold() -> None:
    case, grid, _, _, result = _run("rotation-gauss")
    loose = summarise(result, grid.volume, case["density"], quantile=0.5)
    tight = summarise(result, grid.volume, case["density"], quantile=0.99)

    assert float(tight.dose_threshold) < float(loose.dose_threshold)
    assert float(loose.abs_en_threshold) == 50.0
    assert float(tight.abs_en_threshold) == 99.0


# --- shape of the output ------------------------------------------------


def test_the_summary_serialises_in_column_order() -> None:
    """``as_dict`` is what a JSON reference is written from, so its order and
    its key set are part of the contract.
    """
    summary, _, _ = _summarise("still-tophat")
    keys = list(summary.as_dict())

    assert keys == ["wedge_number", *FIELDS]
    assert all(isinstance(v, float) for v in summary.as_dict().values())


def test_every_reported_number_is_finite() -> None:
    for case_id in CASE_IDS:
        summary, _, _ = _summarise(case_id)
        for field, value in summary.as_dict().items():
            assert math.isfinite(value), f"{case_id}: {field} is {value}"
