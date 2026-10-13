"""Escape modelling's pieces, and how a run with escape behaves.

Whether escape reproduces RADDOSE-3D is the conformance corpus's question:
eight cases, compared value by value with the same seed. These cover what
the corpus does not: the arithmetic the Java's draws depend on, and the
refusals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyraddose_3d.backend.beam import pink_beam_energies
from pyraddose_3d.backend.escape import round_half_up
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.solve import solve

CASES = Path(__file__).parent.parent / "conformance" / "cases"


def test_rounding_is_javas() -> None:
    """Half up; 0.49999999999999994 down; NaN to zero, as StrictMath.round."""
    values = np.array([0.5, 1.5, -0.5, -1.5, 2.4, -2.6, 0.49999999999999994, np.nan])
    assert round_half_up(values).tolist() == [1, 2, 0, -1, 2, -3, 0, 0]


#: ``SampleNormalEnergyDistribution(12.1, 0.5, 200)``, first, second, the two
#: either side of the mean, second last and last. Printed by
#: ``tests/backend/data/se/raddo/raddose3D/DumpPink.java``.
PINK = {
    0: 11.552706912960844,
    1: 11.605648290769896,
    99: 12.09867602746605,
    100: 12.10132397253395,
    198: 12.594351709230105,
    199: 12.647293087039156,
}


def test_pink_beam_energies_are_the_javas() -> None:
    """To an ulp: the Java inverts the normal CDF with Apache's erfInv."""
    energies = pink_beam_energies(12.1, 0.5)
    assert len(energies) == 200
    for index, value in PINK.items():
        assert energies[index] == pytest.approx(value, rel=2e-16, abs=0)


def test_pink_beam_energies_carry_a_gradient() -> None:
    torch = pytest.importorskip("torch")
    mean = torch.tensor(12.1, dtype=torch.float64, requires_grad=True)
    pink_beam_energies(mean, 0.5).sum().backward()
    assert mean.grad.item() == pytest.approx(200.0)


@pytest.fixture(scope="module")
def escaping() -> Experiment:
    return Experiment.from_yaml(CASES / "100-escape-pe" / "config.yaml")


def test_the_same_seed_gives_the_same_dose(escaping: Experiment) -> None:
    first = [s.as_dict() for s in solve(escaping)]
    assert first == [s.as_dict() for s in solve(escaping)]
    other = escaping.model_copy(update={"seed": 1})
    assert [s.as_dict() for s in solve(other)] != first


def test_without_a_seed_one_is_chosen_and_recorded(escaping: Experiment) -> None:
    """As RADDOSE-3D seeds from the clock and prints what it used."""
    from pyraddose_3d.solve import simulate

    unseeded = escaping.model_copy(update={"seed": None})
    simulation = simulate(unseeded)
    assert simulation.seed is not None
    again = simulate(escaping.model_copy(update={"seed": simulation.seed}))
    assert [s.as_dict() for s in again.summaries] == [
        s.as_dict() for s in simulation.summaries
    ]


def test_escape_has_no_gradient(escaping: Experiment) -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(NotImplementedError, match="no gradient"):
        solve(escaping, {"beam.flux": torch.tensor(2e12, dtype=torch.float64)})


def test_photoelectron_escape_needs_elements(escaping: Experiment) -> None:
    """RADDOSE-3D fails with a NullPointerException here."""
    config = escaping.model_dump()
    config["crystal"]["composition"] = {"method": "average"}
    with pytest.raises(ValueError, match="needs the crystal's elements"):
        solve(Experiment.model_validate(config))


def test_fluorescence_without_elements_is_skipped_with_a_warning(
    escaping: Experiment,
) -> None:
    """As in RADDOSE-3D, which warns and applies none."""
    config = escaping.model_dump()
    config["crystal"]["composition"] = {"method": "average"}
    config["crystal"]["escape"] = {"fluorescence": True}
    fluorescing = Experiment.model_validate(config)
    config["crystal"]["escape"] = {}
    plain = Experiment.model_validate(config)
    with pytest.warns(UserWarning, match="none is modelled"):
        result = solve(fluorescing)
    assert [s.as_dict() for s in result] == [s.as_dict() for s in solve(plain)]
