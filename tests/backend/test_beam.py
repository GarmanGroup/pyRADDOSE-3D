"""Tests for beam profiles.

The conformance fixture probes points chosen to sit either side of every
edge in the model: just inside and just outside a rectangular aperture, on
and off the axis of an elliptical one, and well outside both.
"""

import csv
import math
import pathlib

import numpy as np
import pytest

from pyraddose_3d.backend.beam import (
    FWHM_TO_SIGMA,
    KEV_TO_JOULES,
    ExperimentalBeam,
    GaussianBeam,
    TophatBeam,
    normal_cdf,
    read_pgm,
)

REFERENCE = pathlib.Path(__file__).parent / "data" / "beams.tsv"

FLUX = 2.0e12
ENERGY = 12.1

BEAMS: dict[str, GaussianBeam | TophatBeam] = {
    "gauss-rect": GaussianBeam(FLUX, ENERGY, 80.0, 60.0, 100.0, 100.0),
    "gauss-uncollimated": GaussianBeam(FLUX, ENERGY, 80.0, 60.0),
    "gauss-ellipse": GaussianBeam(
        FLUX, ENERGY, 80.0, 60.0, 100.0, 100.0, circular=True
    ),
    "gauss-narrow-coll": GaussianBeam(FLUX, ENERGY, 80.0, 60.0, 40.0, 30.0),
    "tophat-rect": TophatBeam(FLUX, ENERGY, 100.0, 100.0),
    "tophat-ellipse": TophatBeam(FLUX, ENERGY, 100.0, 80.0, circular=True),
}


def _reference_rows(tag: str) -> list[list[str]]:
    with REFERENCE.open(encoding="ascii") as handle:
        return [row for row in csv.reader(handle, delimiter="\t") if row[0] == tag]


# --- conformance against the Java ---------------------------------------


@pytest.mark.conformance
def test_intensity_matches_java() -> None:
    """Asserted at 1e-12. Two thirds of the points come out bit-identical;
    the rest differ only in the elliptical aperture, where the Java sums a
    1000-step quadrature in a different order.
    """
    rows = _reference_rows("I")
    assert len(rows) == 90, "reference fixture is truncated"

    worst, worst_at = 0.0, ""
    for row in rows:
        beam_id = row[1]
        x, y, off_axis, want = (float(v) for v in row[2:6])
        got = float(BEAMS[beam_id].intensity(x, y, off_axis))
        if got == want:
            continue
        rel = abs(got - want) / abs(want) if want else abs(got)
        if rel > worst:
            worst, worst_at = rel, f"{beam_id} at ({x}, {y}) off_axis={off_axis}"
    assert worst < 1e-12, f"worst relative deviation {worst:.3e} at {worst_at}"


@pytest.mark.conformance
def test_minimum_dimension_matches_java() -> None:
    rows = _reference_rows("MINDIM")
    assert len(rows) == 6
    for row in rows:
        assert float(BEAMS[row[1]].minimum_dimension) == float(row[2])


@pytest.mark.conformance
def test_fixture_parameters_are_the_ones_used_here() -> None:
    """Guards against the fixture being regenerated with different inputs
    while these beams keep the old ones -- which would silently compare
    two different calculations.
    """
    row = _reference_rows("PARAMS")[0]
    assert float(row[2]) == FLUX
    assert float(row[4]) == ENERGY


# --- collimation --------------------------------------------------------


def test_uncollimated_beam_is_normalised_over_the_plane() -> None:
    assert BEAMS["gauss-uncollimated"].collimated_fraction == 1.0


def test_collimated_fraction_has_a_closed_form() -> None:
    """For a rectangular aperture the four-term CDF expression collapses to
    a product of two error functions. Checking against that is independent
    evidence, not a restatement of the implementation.
    """
    beam = BEAMS["gauss-rect"]
    expected = math.erf(
        beam.collimation_x / (2 * math.sqrt(2) * beam.sigma_x)
    ) * math.erf(beam.collimation_y / (2 * math.sqrt(2) * beam.sigma_y))
    assert float(beam.collimated_fraction) == pytest.approx(expected, rel=1e-12)


def test_an_ellipse_passes_less_than_its_bounding_rectangle() -> None:
    rectangle = float(BEAMS["gauss-rect"].collimated_fraction)
    ellipse = float(BEAMS["gauss-ellipse"].collimated_fraction)
    assert ellipse < rectangle


def test_narrower_collimation_passes_less() -> None:
    assert float(BEAMS["gauss-narrow-coll"].collimated_fraction) < float(
        BEAMS["gauss-rect"].collimated_fraction
    )


def test_collimation_conserves_flux_at_the_centre() -> None:
    """Dividing by the collimated fraction means blocked photons are
    redistributed, not discarded: a tighter aperture raises the on-axis
    intensity. Worth pinning, because dropping the division would look
    physically sensible and be wrong.
    """
    wide = float(BEAMS["gauss-rect"].intensity(0.0, 0.0))
    narrow = float(BEAMS["gauss-narrow-coll"].intensity(0.0, 0.0))
    assert narrow > wide


def test_intensity_is_zero_outside_a_rectangular_aperture() -> None:
    beam = BEAMS["gauss-rect"]
    assert float(beam.intensity(49.0, 0.0)) > 0
    assert float(beam.intensity(51.0, 0.0)) == 0.0
    assert float(beam.intensity(0.0, 51.0)) == 0.0


def test_intensity_is_zero_outside_an_elliptical_aperture() -> None:
    """(40, 40) is inside the bounding square but outside the inscribed
    ellipse, which is the case a rectangular test would miss.
    """
    beam = BEAMS["gauss-ellipse"]
    assert float(beam.intensity(49.0, 0.0)) > 0
    assert float(beam.intensity(40.0, 40.0)) == 0.0


def test_off_axis_shifts_the_profile_not_the_aperture() -> None:
    """``off_axis`` is subtracted from x before both the profile and the
    aperture test, so the whole beam moves together.
    """
    beam = BEAMS["gauss-rect"]
    assert float(beam.intensity(15.0, 0.0, 15.0)) == pytest.approx(
        float(beam.intensity(0.0, 0.0, 0.0))
    )


# --- tophat -------------------------------------------------------------


def test_tophat_is_uniform_inside_its_footprint() -> None:
    beam = BEAMS["tophat-rect"]
    centre = float(beam.intensity(0.0, 0.0))
    assert float(beam.intensity(49.0, 49.0)) == centre
    assert float(beam.intensity(51.0, 0.0)) == 0.0


def test_tophat_intensity_is_flux_over_area() -> None:
    beam = BEAMS["tophat-rect"]
    expected = KEV_TO_JOULES * FLUX * ENERGY / (100.0 * 100.0)
    assert float(beam.intensity(0.0, 0.0)) == pytest.approx(expected, rel=1e-14)


def test_elliptical_tophat_uses_the_ellipse_area() -> None:
    beam = BEAMS["tophat-ellipse"]
    assert float(beam.area) == pytest.approx(math.pi * 50.0 * 40.0)
    assert float(beam.intensity(0.0, 45.0)) == 0.0


# --- profile shape ------------------------------------------------------


def test_fwhm_is_where_intensity_halves() -> None:
    """The defining property of the full width at half maximum, checked on
    an uncollimated beam so the aperture cannot interfere.
    """
    beam = BEAMS["gauss-uncollimated"]
    centre = float(beam.intensity(0.0, 0.0))
    at_half_width = float(beam.intensity(beam.fwhm_x / 2, 0.0))
    assert at_half_width == pytest.approx(centre / 2, rel=1e-12)


def test_fwhm_to_sigma_factor() -> None:
    assert FWHM_TO_SIGMA == pytest.approx(2.3548200450309493)


def test_profile_integrates_to_the_total_flux() -> None:
    """An uncollimated Gaussian must put all of its energy somewhere.

    Integrated numerically over a grid wide enough to contain it.
    """
    beam = BEAMS["gauss-uncollimated"]
    step = 0.5
    axis = np.arange(-300.0, 300.0, step) + step / 2
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    total = float(np.sum(beam.intensity(grid_x, grid_y)) * step * step)
    assert total == pytest.approx(KEV_TO_JOULES * FLUX * ENERGY, rel=1e-9)


def test_normal_cdf_matches_the_standard_normal() -> None:
    for x, want in ((0.0, 0.5), (1.0, 0.8413447460685429), (-1.0, 0.1586552539314571)):
        assert float(normal_cdf(x, 1.0)) == pytest.approx(want, rel=1e-12)


# --- differentiability --------------------------------------------------


def test_gradient_flows_to_flux_and_widths() -> None:
    torch = pytest.importorskip("torch")
    flux = torch.tensor(FLUX, dtype=torch.float64, requires_grad=True)
    fwhm_x = torch.tensor(80.0, dtype=torch.float64, requires_grad=True)
    beam = GaussianBeam(flux, ENERGY, fwhm_x, 60.0, 100.0, 100.0)
    beam.intensity(0.0, 0.0).backward()

    assert flux.grad.item() > 0, "more photons must mean more intensity"
    assert fwhm_x.grad.item() < 0, "a broader beam must be dimmer on axis"


def test_gradient_is_numerically_correct_for_a_collimated_beam() -> None:
    torch = pytest.importorskip("torch")

    def on_axis(flux: object, fwhm_x: object, fwhm_y: object) -> object:
        return GaussianBeam(flux, ENERGY, fwhm_x, fwhm_y, 100.0, 100.0).intensity(
            10.0, 5.0
        )

    args = (
        torch.tensor(FLUX, dtype=torch.float64, requires_grad=True),
        torch.tensor(80.0, dtype=torch.float64, requires_grad=True),
        torch.tensor(60.0, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(on_axis, args, eps=1e-6, atol=1e-8)


def test_elliptical_quadrature_is_differentiable() -> None:
    """The midpoint sum is a finite sum, so it carries a gradient -- which is
    not obvious, and is the reason it was not written as a loop.
    """
    torch = pytest.importorskip("torch")
    fwhm_x = torch.tensor(80.0, dtype=torch.float64, requires_grad=True)
    beam = GaussianBeam(FLUX, ENERGY, fwhm_x, 60.0, 100.0, 100.0, circular=True)
    beam.collimated_fraction.backward()
    assert fwhm_x.grad is not None
    assert torch.isfinite(fwhm_x.grad)
    # A broader beam loses more of itself to a fixed aperture.
    assert fwhm_x.grad.item() < 0


def test_both_backends_agree() -> None:
    torch = pytest.importorskip("torch")
    points = np.array([0.0, 10.0, 25.0, 49.0, 51.0])
    for beam_id in BEAMS:
        beam = BEAMS[beam_id]
        got_np = beam.intensity(points, points * 0.5)
        got_torch = beam.intensity(
            torch.tensor(points, dtype=torch.float64),
            torch.tensor(points * 0.5, dtype=torch.float64),
        )
        np.testing.assert_allclose(
            got_np, got_torch.numpy(), rtol=1e-13, err_msg=beam_id
        )


def test_intensity_is_vectorised() -> None:
    beam = BEAMS["gauss-rect"]
    grid = np.linspace(-60.0, 60.0, 17)
    got = beam.intensity(grid, np.zeros_like(grid))
    assert got.shape == grid.shape
    for i, x in enumerate(grid):
        assert got[i] == pytest.approx(float(beam.intensity(float(x), 0.0)))


# --- experimental beams -------------------------------------------------
#
# RADDOSE-3D's BeamExperimentalTest checks nothing: it never applies a
# container, so the beam's attenuated flux stays 0 and every intensity is
# 0, and its assertions are one-sided -- intensity minus expected below a
# tolerance -- which 0 always satisfies. These check the values.

#: ``BeamExperimentalTest``'s profile that changes along both axes,
#: indexed [y, x]: 4 by 4 pixels of 5 um.
BOTH = np.add.outer(np.arange(4.0), np.arange(4.0)) + 2
PIXEL = 5.0


def _experimental() -> ExperimentalBeam:
    return ExperimentalBeam(1e12, 10.0, BOTH, PIXEL, PIXEL)


def _per_pixel(value: float) -> float:
    """A pixel's value as an energy flux, as generateBeamArray scales it."""
    return value * KEV_TO_JOULES * 10.0 * 1e12 / (BOTH.sum() * PIXEL * PIXEL)


def test_an_experimental_beam_holds_its_pixels_at_their_centres() -> None:
    """Pixel [i, j] is centred at x = 5j - 7.5, y = 5i - 7.5, the image
    being centred on the beam axis with a border of zeros one pixel wide.
    """
    i, j = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    got = _experimental().intensity(5.0 * j - 7.5, 5.0 * i - 7.5)
    assert got.tolist() == [[_per_pixel(v) for v in row] for row in BOTH]


def test_an_experimental_beam_interpolates_bilinearly() -> None:
    beam = _experimental()
    assert beam.intensity(0.0, -7.5) == pytest.approx(_per_pixel(3.5), rel=1e-15)
    assert beam.intensity(0.0, 0.0) == pytest.approx(_per_pixel(5.0), rel=1e-15)
    assert beam.intensity(-6.25, -6.25) == pytest.approx(_per_pixel(2.5), rel=1e-7)


def test_an_experimental_beam_carries_its_whole_flux() -> None:
    """Summed over the pixel centres, it is the beam's power."""
    i, j = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    power = _experimental().intensity(5.0 * j - 7.5, 5.0 * i - 7.5).sum()
    assert power * PIXEL * PIXEL == pytest.approx(KEV_TO_JOULES * 10.0 * 1e12)


def test_an_experimental_beam_fades_into_its_border() -> None:
    """Out to one pixel short of the image's edge, then nothing."""
    beam = _experimental()
    # Halfway between the last pixel's centre and the zero border's.
    assert beam.intensity(10.0, 7.5) == pytest.approx(_per_pixel(BOTH[3, 3]) / 2)
    assert beam.intensity(10.001, 7.5) == 0.0
    assert beam.intensity(0.0, -10.001) == 0.0
    assert beam.minimum_dimension == 30.0


def test_an_aperture_clips_the_profile_and_carries_the_flux() -> None:
    """As RADDOSE-3D's ``anApertureClipsTheProfileAndCarriesTheFlux``.

    A 10 um square lets through the four central pixels, of 4, 5, 5 and 6,
    and they carry the whole flux -- the flux after collimation.
    """
    beam = ExperimentalBeam(1e12, 10.0, BOTH, PIXEL, PIXEL, 10.0, 10.0)
    per_unit = KEV_TO_JOULES * 10.0 * 1e12 / ((4 + 5 + 5 + 6) * PIXEL * PIXEL)
    assert beam.intensity(-2.5, -2.5) == pytest.approx(4 * per_unit, rel=1e-15)
    assert beam.intensity(2.5, 2.5) == pytest.approx(6 * per_unit, rel=1e-15)
    assert beam.intensity(5.001, 0.0) == 0.0
    assert beam.intensity(0.0, -7.5) == 0.0


def test_a_circular_aperture_is_the_ellipse_its_sizes_give() -> None:
    beam = ExperimentalBeam(1e12, 10.0, BOTH, PIXEL, PIXEL, 16.0, 16.0, circular=True)
    assert beam.intensity(7.5, 0.0) > 0
    assert beam.intensity(7.5, 7.5) == 0.0


def test_an_aperture_admitting_nothing_is_refused() -> None:
    beam = ExperimentalBeam(1e12, 10.0, BOTH, PIXEL, PIXEL, 1.0, 1.0)
    with pytest.raises(ValueError, match="lets none of its image through"):
        beam.intensity(0.0, 0.0)


def test_an_experimental_beam_carries_a_gradient_in_flux() -> None:
    torch = pytest.importorskip("torch")
    flux = torch.tensor(1e12, dtype=torch.float64, requires_grad=True)
    beam = ExperimentalBeam(flux, 10.0, BOTH, PIXEL, PIXEL)
    intensity = beam.intensity(np.array([0.0, 2.5]), np.array([0.0, -2.5]))
    intensity.sum().backward()
    assert flux.grad.item() == pytest.approx(intensity.sum().item() / 1e12)


def test_a_pgm_is_read_into_the_beams_frame(tmp_path: pathlib.Path) -> None:
    """Reversed both ways and transposed, as BeamExperimentalpgm stores it."""
    path = tmp_path / "beam.pgm"
    path.write_text("P2\n# one\n\n# two\n3 2\n9\n1 2 3\n4 5 6\n", "ascii")
    assert read_pgm(path).tolist() == [[6, 3], [5, 2], [4, 1]]


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        ("P5\n3 2\n9\n", "not a plain"),
        ("P2\n3 2\n9\n1 2 3 4 5\n", "5 pixels where its header says 3 x 2"),
        ("P2\n3\n", "does not hold a PGM header"),
    ],
)
def test_a_pgm_that_cannot_be_read_is_refused(
    tmp_path: pathlib.Path, text: str, complaint: str
) -> None:
    path = tmp_path / "beam.pgm"
    path.write_text(text, "latin-1")
    with pytest.raises(ValueError, match=complaint):
        read_pgm(path)
