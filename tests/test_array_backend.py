"""Tests for the NumPy/PyTorch dispatch layer.

These exist because the whole two-backend strategy rests on a claim that is
easy to assert and easy to get wrong: that the same source, run on either
library, gives the same numbers, and that under torch those numbers still
carry a correct gradient.
"""

import math

import numpy as np
import pytest

from pyraddose_3d.backend._array import (
    array_namespace,
    erf,
    is_torch,
    promote,
    to_namespace,
)

torch = pytest.importorskip("torch", reason="torch is an optional extra")

SQRT2 = math.sqrt(2.0)
# 2 * sqrt(2 * ln 2), the FWHM-to-sigma factor used throughout RADDOSE-3D.
FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def collimated_fraction(fwhm_x, fwhm_y, coll_x, coll_y):
    """Fraction of a 2D Gaussian beam passing a rectangular aperture.

    Written once, against no particular array library. This is the shape
    every backend module is expected to take, and it is the only place in
    the MX path that needs :func:`~pyraddose_3d.backend._array.erf`.
    """
    sigma_x = fwhm_x / FWHM_TO_SIGMA
    sigma_y = fwhm_y / FWHM_TO_SIGMA
    return erf(coll_x / (2.0 * SQRT2 * sigma_x)) * erf(coll_y / (2.0 * SQRT2 * sigma_y))


def test_namespace_dispatches_on_operand_type() -> None:
    assert array_namespace(np.float64(1.0)) is np
    assert array_namespace(torch.tensor(1.0)) is torch


def test_namespace_promotes_like_torch() -> None:
    """A tensor anywhere makes the expression a tensor expression.

    torch accepts NumPy operands; NumPy does not reliably accept tensors.
    """
    assert array_namespace(np.float64(1.0), torch.tensor(1.0)) is torch


def test_is_torch_does_not_need_torch_imported() -> None:
    """Identification is by module name, so it works without the import."""
    assert not is_torch(np.float64(1.0))
    assert not is_torch(1.0)
    assert is_torch(torch.tensor(1.0))


def test_erf_matches_between_backends() -> None:
    values = [-3.0, -0.5, 0.0, 1e-8, 0.5, 3.0]
    got_np = erf(np.asarray(values, dtype=np.float64))
    got_torch = erf(torch.tensor(values, dtype=torch.float64))
    np.testing.assert_allclose(got_np, got_torch.numpy(), rtol=0, atol=1e-15)


def test_erf_matches_stdlib() -> None:
    """Guards the scipy path against a wrong import or a bad approximation."""
    values = [-3.0, -0.5, 0.0, 0.5, 3.0]
    expected = [math.erf(v) for v in values]
    np.testing.assert_allclose(
        erf(np.asarray(values, dtype=np.float64)), expected, rtol=0, atol=1e-15
    )


def test_same_source_gives_same_answer_on_both_backends() -> None:
    """The core claim: one implementation, two libraries, one number."""
    args = (80.0, 60.0, 100.0, 100.0)
    got_np = collimated_fraction(*(np.float64(v) for v in args))
    got_torch = collimated_fraction(
        *(torch.tensor(v, dtype=torch.float64) for v in args)
    )
    # Tighter than the 1e-5 conformance target by ten orders of magnitude:
    # the two backends should agree to machine precision, not merely to
    # within tolerance. If this loosens, something has genuinely diverged.
    assert abs(float(got_np) - got_torch.item()) / float(got_np) < 1e-14


def test_gradient_is_correct_not_merely_present() -> None:
    """Differentiability is the reason torch is an option at all."""
    args = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for v in (80.0, 60.0, 100.0, 100.0)
    )
    assert torch.autograd.gradcheck(collimated_fraction, args, eps=1e-6, atol=1e-8)


def test_widening_the_aperture_admits_more_beam() -> None:
    """A sign check on the gradient, in case gradcheck ever agrees with a
    consistently wrong derivative.
    """
    fwhm_x = torch.tensor(80.0, dtype=torch.float64, requires_grad=True)
    coll_x = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    collimated_fraction(
        fwhm_x,
        torch.tensor(60.0, dtype=torch.float64),
        coll_x,
        torch.tensor(100.0, dtype=torch.float64),
    ).backward()
    assert coll_x.grad.item() > 0, "opening the slits must admit more beam"
    assert fwhm_x.grad.item() < 0, "a broader beam must lose more to the slits"


def test_promote_leaves_numpy_untouched() -> None:
    xp, values = promote(1.0, np.float64(2.0))
    assert xp is np
    assert values == (1.0, np.float64(2.0))


def test_promote_converts_scalars_for_torch() -> None:
    """The bug this exists for: torch.cos rejects a Python float, so a
    function with a float default such as `angle=90.0` breaks the moment one
    of its other arguments is a tensor.
    """
    xp, (_tensor, scalar) = promote(torch.tensor(1.0, dtype=torch.float64), 90.0)
    assert xp is torch
    assert is_torch(scalar)
    assert torch.cos(scalar).item() == pytest.approx(math.cos(90.0))


def test_promote_keeps_the_tensor_dtype() -> None:
    """A float32 calculation must not be silently widened to float64."""
    _, (_tensor, scalar) = promote(torch.tensor(1.0, dtype=torch.float32), 2.0)
    assert scalar.dtype == torch.float32


def test_promote_does_not_detach_gradients() -> None:
    leaf = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    _, (promoted, _) = promote(leaf, 1.0)
    assert promoted is leaf
    assert promoted.requires_grad


def test_promote_never_narrows_a_numpy_array() -> None:
    """A float64 mesh beside a float32 tensor stays float64: it was cut to
    float32, which moved the torch path's doses by 2e-4."""
    vertices = np.zeros((4, 3))
    _, (promoted, scalar, _tensor) = promote(
        vertices, 0.5, torch.zeros(3, dtype=torch.float32)
    )
    assert promoted.dtype == torch.float64
    assert scalar.dtype == torch.float32


def test_promote_gives_numbers_a_floating_dtype() -> None:
    """Beside an integer tensor, 0.5 is 0.5, not 0."""
    _, (_index, half) = promote(torch.tensor([1, 2]), 0.5)
    assert half.dtype == torch.float64
    assert half.item() == 0.5


def test_numbers_become_float64_tensors() -> None:
    """torch.as_tensor made Python floats float32, torch's default."""
    assert to_namespace(torch, (0.0, 0.0, 0.0)).dtype == torch.float64
    assert to_namespace(torch, 0.5).dtype == torch.float64
    assert to_namespace(torch, [1, 2]).dtype == torch.int64
    assert to_namespace(torch, [True, False]).dtype == torch.bool


def test_conversions_follow_the_tensors_device() -> None:
    """Tensors made to meet a tensor are made on its device, so a CUDA run
    does not mix devices. The meta device checks it without a GPU."""
    meta = torch.zeros(3, dtype=torch.float64, device="meta")
    _, (vertices, angle, _tensor) = promote(np.zeros((4, 3)), 0.5, meta)
    assert vertices.device.type == "meta"
    assert angle.device.type == "meta"
    assert to_namespace(torch, (0.0, 1.0), like=meta).device.type == "meta"
    assert to_namespace(torch, torch.zeros(2), like=meta).device.type == "meta"
    assert to_namespace(torch, np.zeros(2)).device.type == "cpu"
