"""The Leal et al. (2012) dose decay model.

Ports ``DDMLeal``. A crystal's diffraction fades with dose for two reasons,
which Leal et al. separate: an overall scale factor that falls as a
Gaussian in dose, and a B factor that rises linearly with it, fading high
resolution first. Integrating over the average intensity a typical protein
crystal gives at each resolution -- the BEST curve, vendored in
``best_intensity.tsv`` -- gives the integrated intensity at a dose,

    I(D) = exp(-(gamma D)^2) * sum_h h^2 J(h) dh exp(-(B0 + beta D) h^2 / 2)

and the relative diffraction efficiency is ``I(D) / I(0)``. Leal et al.
2012, equation 4, with ``h^2 = 1/d^2``.

The sum is over the 299 midpoints between the table's rows, formed as the
Java forms them: each term's constant part is held as a logarithm and added
to its dose-dependent exponent inside a single ``exp``.
"""

from __future__ import annotations

import functools
from importlib import resources
from typing import Any

import numpy as np

from pyraddose_3d.backend._array import array_namespace

__all__ = ["leal_decay"]

Array = Any

#: Voxels per block when summing over the table, to keep the
#: (voxels, 299) intermediate small for a large crystal.
_BLOCK = 16384


@functools.cache
def _best_table() -> tuple[np.ndarray, np.ndarray]:
    """``(h2, J)`` from the vendored BEST table."""
    text = resources.files("pyraddose_3d.data").joinpath("best_intensity.tsv")
    rows = [
        line.split("\t")
        for line in text.read_text("ascii").splitlines()
        if line and not line.startswith(("#", "h2"))
    ]
    table = np.array(rows, dtype=np.float64)
    return table[:, 0], table[:, 1]


def _terms(b_zero: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """``DDMLeal``'s ``exponentialProduct`` and ``logIntegralProduct``."""
    h2, intensity = _best_table()
    middle_h2 = (h2[:-1] + h2[1:]) / 2
    middle_j = (intensity[:-1] + intensity[1:]) / 2
    width = np.sqrt(h2[1:]) - np.sqrt(h2[:-1])
    log_constant = np.log(
        middle_h2 * middle_j * width * np.exp(-0.5 * b_zero * middle_h2)
    )
    return beta * -0.5 * middle_h2, log_constant


def leal_decay(gamma: float, b_zero: float, beta: float) -> Any:
    """The Leal et al. relative diffraction efficiency, as a function of dose.

    Parameters
    ----------
    gamma : float
        Width of the Gaussian decay of the scale factor, per MGy.
    b_zero : float
        B factor at zero dose, square Angstroms.
    beta : float
        Rate the B factor rises with dose, square Angstroms per MGy.

    Returns
    -------
    Any
        A callable from dose in MGy, an array or a tensor, to the efficiency
        relative to zero dose. Differentiable in the dose.
    """
    slope_np, log_constant_np = _terms(b_zero, beta)
    # Summed in order, as the Java sums it; Python's own sum compensates.
    zero_dose = 0.0
    for term in np.exp(log_constant_np):
        zero_dose += float(term)

    def decay(dose: Array) -> Array:
        xp = array_namespace(dose)
        if xp is np:
            dose = np.asarray(dose, dtype=np.float64)
            slope, log_constant = slope_np, log_constant_np
        else:
            slope = xp.asarray(slope_np, dtype=dose.dtype)
            log_constant = xp.asarray(log_constant_np, dtype=dose.dtype)
        flat = xp.reshape(dose, (-1,))
        sums = [
            xp.sum(xp.exp(block[:, None] * slope + log_constant), axis=1)
            for block in (flat[i : i + _BLOCK] for i in range(0, flat.shape[0], _BLOCK))
        ]
        total = xp.concat(sums) if len(sums) > 1 else sums[0]
        intensity = xp.exp(-((gamma * flat) ** 2)) * total
        return xp.reshape(intensity / zero_dose, dose.shape)

    return decay
