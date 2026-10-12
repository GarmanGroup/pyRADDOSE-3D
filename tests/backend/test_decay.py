"""The Leal et al. decay model against RADDOSE-3D's own."""

from __future__ import annotations

import numpy as np
import pytest

from pyraddose_3d.backend.decay import leal_decay

DOSES = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

#: ``DDMLeal.calcDecay`` at ``DOSES``, printed by
#: ``tests/backend/data/DumpLeal.java`` against RADDOSE-3D 5.0.
JAVA = {
    (0.3, 10.0, 0.03): [
        1.0, 0.9765350749529179, 0.911660293426176, 0.6942173421745056,
        0.10410194841996147, 1.2040677067532557e-4, 2.2091664348239722e-16,
    ],
    (0.1, 5.0, 0.5): [
        1.0, 0.9649520291634105, 0.9273851859369268, 0.8461278792100324,
        0.5807895024906737, 0.2179537134548854, 0.007599515078081675,
    ],
    (0.0, 0.0, 0.0): [1.0] * 7,
}  # fmt: skip


@pytest.mark.parametrize("parameters", list(JAVA))
def test_matches_the_java(parameters: tuple[float, float, float]) -> None:
    np.testing.assert_allclose(
        leal_decay(*parameters)(DOSES), JAVA[parameters], rtol=1e-13
    )


def test_takes_a_scalar_and_keeps_the_shape() -> None:
    decay = leal_decay(0.3, 10.0, 0.03)
    assert float(decay(2.0)) == pytest.approx(JAVA[(0.3, 10.0, 0.03)][3], rel=1e-13)
    assert decay(np.ones((4, 5))).shape == (4, 5)


def test_a_large_crystal_is_summed_in_blocks() -> None:
    doses = np.linspace(0.0, 20.0, 40_000)
    decay = leal_decay(0.1, 5.0, 0.5)
    np.testing.assert_allclose(
        decay(doses)[::4000], [float(decay(d)) for d in doses[::4000]], rtol=1e-14
    )


def test_is_differentiable_in_dose() -> None:
    torch = pytest.importorskip("torch")
    decay = leal_decay(0.3, 10.0, 0.03)
    dose = torch.tensor([1.0, 5.0], dtype=torch.float64, requires_grad=True)
    decay(dose).sum().backward()
    step = 1e-6
    finite = (
        decay(np.array([1.0, 5.0]) + step) - decay(np.array([1.0, 5.0]) - step)
    ) / (2 * step)
    np.testing.assert_allclose(dose.grad.numpy(), finite, rtol=1e-6)
