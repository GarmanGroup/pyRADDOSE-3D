"""Optimising the energy: the sensor model, the edges, and the search."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyraddose_3d.backend.container import elemental_mass_attenuation
from pyraddose_3d.backend.detector import sensor_absorption, sensor_edges
from pyraddose_3d.backend.elements import element_database
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.optimise import (
    absorption_edges,
    diffraction_efficiency,
    edge_free_intervals,
    maximise_energy,
    maximise_on_interval,
)
from pyraddose_3d.solve import solve


def _experiment(heavy_atoms: dict[str, float], **crystal: object) -> Experiment:
    return Experiment.model_validate(
        {
            "crystal": {
                "type": "cuboid",
                "dimensions": [40.0, 30.0, 20.0],
                "pixels_per_micron": 0.5,
                "composition": {
                    "method": "unit_cell",
                    "unit_cell": [78.02, 78.02, 78.02],
                    "num_monomers": 24,
                    "num_residues": 51,
                    "protein_heavy_atoms": heavy_atoms,
                    "solvent_heavy_conc": {"P": 425},
                },
                **crystal,
            },
            "beam": {
                "type": "gaussian",
                "flux": 2e12,
                "energy": 12.1,
                "fwhm": [15, 15],
            },
            "wedges": [{"start": 0.0, "end": 90.0, "exposure_time": 30.0}],
        }
    )


INSULIN = {"Zn": 0.333, "S": 6}
SEMET = {"Zn": 0.333, "S": 6, "Se": 2}

# --- the sensor --------------------------------------------------------------


def test_silicon_stops_soft_photons_and_passes_hard_ones() -> None:
    assert sensor_absorption(8.0) > 0.99
    assert sensor_absorption(20.0) < 0.4
    assert sensor_absorption(20.0, "Si", 1000.0) > sensor_absorption(20.0)
    assert sensor_absorption(20.0, "CdTe", 750.0) > 0.99


@pytest.mark.parametrize("energy", [6.0, 8.0, 10.0])
def test_silicon_photoabsorption_agrees_with_nist(energy: float) -> None:
    """An independent check, at energies where photoabsorption is all of it.

    NIST tabulates total attenuation; below 10 keV scattering is under 3% of
    it in silicon. Compared at NIST's own rows, so nothing is interpolated.
    """
    thickness_cm = 0.0450
    nist = -math.expm1(
        -float(elemental_mass_attenuation({"Si": 1.0}, energy)) * 2.329 * thickness_cm
    )
    assert float(sensor_absorption(energy)) == pytest.approx(nist, rel=0.03)


def test_sensor_absorption_is_differentiable_and_vectorised() -> None:
    torch = pytest.importorskip("torch")
    energy = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
    sensor_absorption(energy).backward()
    step = 1e-5
    difference = (
        float(sensor_absorption(15.0 + step)) - float(sensor_absorption(15.0 - step))
    ) / (2 * step)
    assert energy.grad.item() == pytest.approx(difference, rel=1e-7)
    many = sensor_absorption(np.array([8.0, 12.0, 20.0]))
    assert many.shape == (3,)
    assert np.all(np.diff(many) < 0)


def test_cdte_edges_are_its_k_edges() -> None:
    assert {26.711, 31.813} <= set(sensor_edges("CdTe"))


# --- the edges ---------------------------------------------------------------


def test_an_element_knows_its_edges() -> None:
    database = element_database()
    assert 12.658 in database.by_symbol("Se").edges
    assert 9.659 in database.by_symbol("Zn").edges


def test_a_crystals_edges_are_its_elements() -> None:
    assert absorption_edges(_experiment(INSULIN), 5.0, 40.0) == [9.659]
    assert absorption_edges(_experiment(SEMET), 5.0, 40.0) == [9.659, 12.658]


def test_a_containers_edges_count_too() -> None:
    foil = {"type": "elemental", "elements": {"Pb": 1}, "thickness": 5, "density": 11.3}
    edges = absorption_edges(_experiment(INSULIN, container=foil), 5.0, 40.0)
    assert {13.0352, 15.2} <= set(edges)


def test_intervals_stop_short_of_each_edge() -> None:
    pieces = edge_free_intervals(5.0, 40.0, [12.658, 50.0, 9.659])
    assert [round(a, 4) for piece in pieces for a in piece] == [
        5.0, 9.659, 9.659, 12.658, 12.658, 40.0,
    ]  # fmt: skip
    assert pieces[0][1] < 9.659 < pieces[1][0]
    assert pieces[0][0] == 5.0
    assert pieces[-1][1] == 40.0


# --- the search ----------------------------------------------------------------


def test_bisection_finds_an_interior_maximum() -> None:
    pytest.importorskip("torch")
    search = maximise_on_interval(
        lambda x: -((x - 3.3) ** 2), 1.0, 10.0, tolerance=1e-3
    )
    assert search.best[0] == pytest.approx(3.3, abs=1e-3)
    assert search.evaluations <= 2 + math.ceil(math.log2(9.0 / 1e-3))


def test_a_maximum_at_an_end_costs_one_or_two_runs() -> None:
    pytest.importorskip("torch")
    rising = maximise_on_interval(lambda x: x**2, 1.0, 10.0)
    assert (rising.best[0], rising.evaluations) == (10.0, 2)
    falling = maximise_on_interval(lambda x: -x, 1.0, 10.0)
    assert (falling.best[0], falling.evaluations) == (1.0, 1)


def _detected(summaries: list, energy: object) -> object:
    return summaries[0].diffraction_efficiency * sensor_absorption(energy)


def test_maximise_energy_finds_a_true_maximum() -> None:
    """Checked in NumPy, a tenth of a keV either side, independently."""
    pytest.importorskip("torch")
    experiment = _experiment(INSULIN)
    (search,) = maximise_energy(experiment, _detected, 10.0, 14.0)
    energy, _ = search.best
    assert 11.0 < energy < 12.5

    def at(e: float) -> float:
        summary = solve(experiment, {"beam.energy": e})[0]
        return float(summary.diffraction_efficiency * sensor_absorption(e))

    assert at(energy) > at(energy - 0.1)
    assert at(energy) > at(energy + 0.1)


def test_maximise_energy_does_not_cross_an_edge() -> None:
    """Below the Se edge, efficiency rises to the edge and then drops by a third.

    The gradient points up on both sides, so a search that crossed the edge
    would carry on upward into worse territory.
    """
    pytest.importorskip("torch")
    experiment = _experiment(SEMET)
    searches = maximise_energy(
        experiment, lambda summaries, _: summaries[0].diffraction_efficiency, 11.0, 14.0
    )
    assert len(searches) == 2
    below, above = searches
    assert below.best[0] == pytest.approx(12.658, abs=1e-4)
    # Above the edge the search climbs to the top of its piece, which is
    # still a third worse than just below the edge.
    assert above.best[0] == pytest.approx(14.0)
    assert below.best[1] > 1.3 * above.best[1]


# --- the objective ---------------------------------------------------------------


def test_one_wedge_has_its_own_diffraction_efficiency() -> None:
    summaries = solve(_experiment(INSULIN))
    assert diffraction_efficiency(summaries) == summaries[0].diffraction_efficiency


def test_several_wedges_pool_their_yield_and_weight_their_dose() -> None:
    experiment = _experiment(INSULIN)
    two = experiment.model_copy(
        update={"wedges": [*experiment.wedges, experiment.wedges[0]]}
    )
    first, second = solve(two)
    total = float(first.elastic_yield + second.elastic_yield)
    dose = (
        float(
            first.elastic_yield * first.average_dwd
            + second.elastic_yield * second.average_dwd
        )
        / total
    )
    assert float(diffraction_efficiency([first, second])) == pytest.approx(
        total / dose, rel=1e-12
    )
