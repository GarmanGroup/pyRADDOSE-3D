"""Absorption, attenuation and scattering coefficients for a material.

Ports the X-ray half of ``CoefCalcCompute.java``: given what is in a unit cell
and how big that cell is, work out how strongly the material absorbs and
scatters at a given photon energy.

This is the step between :mod:`pyraddose_3d.backend.elements`, which gives
cross-sections per atom, and the exposure loop, which needs coefficients per
unit length.

Units
-----
Deliberately explicit, because the Java carries them only in constant names:

==================  =========================================
:class:`Composition`.counts        atoms per unit cell
:class:`Composition`.cell_volume   cubic Angstroms
``energy``                         keV
:class:`MaterialCoefficients`      inverse micrometres
:func:`density`                    grams per cubic centimetre
==================  =========================================

Inverse micrometres because RADDOSE-3D measures crystals in micrometres. The
conversion is exact and worth stating once: a barn is 1e-24 cm^2 and a cubic
Angstrom is 1e-24 cm^3, so barns per cubic Angstrom is already inverse
centimetres, and the only remaining step is 1/cm to 1/um.

Differentiability
-----------------
Every function here is differentiable in all of its floating-point arguments
-- energy, cell edges and angles, and atom counts. Nothing branches on a
traced value, so the gradient needs no special handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend._array import array_namespace, promote

if TYPE_CHECKING:
    from pyraddose_3d.backend.composition import Composition

__all__ = [
    "AVERAGE_COEFFICIENTS",
    "AVERAGE_DENSITY",
    "MaterialCoefficients",
    "density",
    "material_coefficients",
    "unit_cell_volume",
]

Array = Any
"""A NumPy array, a torch tensor, or a scalar of either."""

# Java spells this conversion as two divisions, by UNITSPERDECIUNIT (10) and
# then UNITSPERMILLIUNIT (1000), applied at different points in the sum. Kept
# separate here so the arithmetic is performed in the same order and rounds
# identically: 1/cm -> 1/mm elementwise, then 1/mm -> 1/um on the total.
_PER_CM_TO_PER_MM = 10.0
_PER_MM_TO_PER_UM = 1000.0

#: Grams per cubic Angstrom to grams per cubic centimetre, as the Java
#: factors it: MASS_TO_CELL_VOLUME (1e27) over UNITSPERMILLIUNIT (1000).
_MASS_TO_CELL_VOLUME = 1e27


@dataclass(frozen=True)
class MaterialCoefficients:
    """Interaction coefficients for a material, in inverse micrometres.

    Attributes
    ----------
    absorption : Array
        Photoelectric absorption. This is the one that deposits dose.
    attenuation : Array
        Total attenuation: photoelectric plus both scattering channels. This
        is what removes photons from the beam.
    elastic : Array
        Coherent (elastic) scattering, which produces the diffraction.
    inelastic : Array
        Incoherent (Compton) scattering.
    """

    absorption: Array
    attenuation: Array
    elastic: Array
    inelastic: Array


def unit_cell_volume(
    cell_a: Array,
    cell_b: Array,
    cell_c: Array,
    alpha: Array = 90.0,
    beta: Array = 90.0,
    gamma: Array = 90.0,
) -> Array:
    """Volume of a unit cell from its edge lengths and angles.

    Parameters
    ----------
    cell_a, cell_b, cell_c : Array
        Edge lengths in Angstroms.
    alpha, beta, gamma : Array, optional
        Cell angles in degrees. Default 90, giving an orthorhombic cell.

    Returns
    -------
    Array
        Volume in cubic Angstroms.

    Raises
    ------
    ValueError
        If the angles do not describe a realisable cell.

    Notes
    -----
    The Java prints a warning here and carries on, which lets
    ``sqrt`` of a negative number turn the volume into NaN and every
    downstream quantity with it. Raising instead is the one deliberate
    behavioural divergence in this module: no valid calculation can reach it,
    and a NaN dose that traces back to a typo in a cell angle is a poor way
    to find out.
    """
    xp, (cell_a, cell_b, cell_c, alpha, beta, gamma) = promote(
        cell_a, cell_b, cell_c, alpha, beta, gamma
    )
    cos_alpha = xp.cos(_radians(alpha))
    cos_beta = xp.cos(_radians(beta))
    cos_gamma = xp.cos(_radians(gamma))

    under_root = (
        1.0
        + 2.0 * cos_alpha * cos_beta * cos_gamma
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
    )
    if bool(xp.any(under_root < 0.0)):
        raise ValueError(
            f"cell angles alpha={alpha}, beta={beta}, gamma={gamma} do not "
            f"describe a realisable unit cell"
        )
    return cell_a * cell_b * cell_c * xp.sqrt(under_root)


def _radians(degrees: Array) -> Array:
    """Convert degrees to radians.

    Multiplication rather than ``numpy.radians`` / ``torch.deg2rad``, which
    are spelled differently and neither of which accepts a Python float.
    """
    return degrees * (math.pi / 180.0)


def material_coefficients(
    composition: Composition, energy: Array
) -> MaterialCoefficients:
    """Interaction coefficients for a composition at a photon energy.

    Each element contributes its cross-section weighted by how many of its
    atoms are in the cell, divided by the cell volume.

    Parameters
    ----------
    composition : Composition
        Atoms per unit cell and the cell volume.
    energy : Array
        Photon energy in keV. May be an array, in which case every returned
        coefficient has that shape.

    Returns
    -------
    MaterialCoefficients
        Coefficients in inverse micrometres.
    """
    # Promote together: a plain-float energy would otherwise send the
    # cross-section calculation down the NumPy path while the counts stayed
    # tensors, and multiplying the two mixes libraries mid-expression.
    xp, (energy, counts, cell_volume) = promote(
        energy, composition.counts, composition.cell_volume
    )

    totals = {"photoelectric": None, "coherent": None, "compton": None, "total": None}
    for index, element in enumerate(composition.elements):
        cross_sections = element.absorption_coefficients(energy)
        count = counts[index]
        for key, value in (
            ("photoelectric", cross_sections.photoelectric),
            ("coherent", cross_sections.coherent),
            ("compton", cross_sections.compton),
            ("total", cross_sections.total),
        ):
            # Divided per element, then the sum divided again, matching the
            # order the Java performs these in.
            term = count * value / cell_volume / _PER_CM_TO_PER_MM
            totals[key] = term if totals[key] is None else totals[key] + term

    if totals["total"] is None:
        zero = xp.zeros_like(xp.asarray(energy))
        return MaterialCoefficients(zero, zero, zero, zero)

    return MaterialCoefficients(
        absorption=totals["photoelectric"] / _PER_MM_TO_PER_UM,
        attenuation=totals["total"] / _PER_MM_TO_PER_UM,
        elastic=totals["coherent"] / _PER_MM_TO_PER_UM,
        inelastic=totals["compton"] / _PER_MM_TO_PER_UM,
    )


def density(composition: Composition) -> Array:
    """Density of a composition.

    Parameters
    ----------
    composition : Composition
        Atoms per unit cell and the cell volume.

    Returns
    -------
    Array
        Density in grams per cubic centimetre.
    """
    xp = array_namespace(composition.counts, composition.cell_volume)
    weights = xp.asarray(
        np.array([e.atomic_weight_in_grams for e in composition.elements])
    )
    mass = xp.sum(composition.counts * weights)
    return mass * _MASS_TO_CELL_VOLUME / (composition.cell_volume * _PER_MM_TO_PER_UM)


#: Density in g/cm^3 that goes with :data:`AVERAGE_COEFFICIENTS`.
AVERAGE_DENSITY = 1.2

AVERAGE_COEFFICIENTS = MaterialCoefficients(
    absorption=0.000237,
    attenuation=0.000281,
    elastic=0.00001799,
    inelastic=0.0,
)
"""Fixed coefficients for a nominal protein crystal, in inverse micrometres.

Ports ``CoefCalcAverage``, which RADDOSE-3D selects for ``AbsCoefCalc Dummy``
or ``Average``. There is no composition and no unit cell: these are constants,
and in particular they do **not** depend on photon energy, so a dose
calculated this way is the same at 6 keV as at 20 keV. That is a property of
the model, not an oversight, but it makes these the wrong coefficients for
anything but a rough first look.

``inelastic`` is zero because the Java returns zero for it, rather than
because Compton scattering is absent: the constant was never fitted.
"""
