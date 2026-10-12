"""Definition of the conformance corpus.

One place defines every case; ``generate.py`` writes the ``config.yaml`` files
and reference outputs from these, and the generated YAML is committed so the
corpus is reviewable as data.

Sampling is one-factor-at-a-time from a baseline, plus a handful of
interaction cases chosen because they correspond to defects actually found in
the reference implementation. The full cross-product of the input space is
~37,000 combinations, so exhaustion is not an option.

Cases are split into two tiers:

``A``
    Deterministic. No Monte Carlo escape modelling, so results are compared
    value by value. This is the differentiable core and where the value is.
``B``
    Monte Carlo. Escape modelling is enabled, so a vectorised implementation
    cannot reproduce the reference's random draw order even with a matching
    generator. Compared statistically only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["REMOVE", "Case", "all_cases"]


class _Remove:
    """Sentinel: drop this key from the baseline rather than override it."""

    def __repr__(self) -> str:
        return "REMOVE"


#: Use in an override to delete a baseline key, e.g. a cuboid's ``dimensions``
#: when switching to a crystal type that does not have that field.
REMOVE = _Remove()

#: Seed for every case. Only affects tier B, but pinned everywhere for clarity.
SEED = 20260917


@dataclass(frozen=True)
class Case:
    """One conformance case.

    Attributes
    ----------
    case_id : str
        Directory name, e.g. ``010-crystal-spherical``.
    tier : {"A", "B"}
        Comparison tier. See the module docstring.
    exercises : str
        What this case is for, in one line. Appears in the committed metadata
        so a failure is interpretable without reading this file.
    overrides : dict
        Deep-merged over the baseline experiment.
    """

    case_id: str
    tier: Literal["A", "B"]
    exercises: str
    overrides: dict[str, Any] = field(default_factory=dict)


#: The composition every case uses unless it is specifically varying it.
BASELINE_COMPOSITION: dict[str, Any] = {
    "method": "unit_cell",
    "unit_cell": [78.02, 78.02, 78.02],
    "num_monomers": 24,
    "num_residues": 51,
    "protein_heavy_atoms": {"Zn": 0.333, "S": 6.0},
    "solvent_heavy_conc": {"P": 425.0},
    "solvent_fraction": 0.64,
}


def baseline() -> dict[str, Any]:
    """The experiment every case is a variation of.

    Deliberately small -- a few thousand voxels -- so the whole corpus
    regenerates in minutes rather than hours.
    """
    return {
        "seed": SEED,
        "crystal": {
            "type": "cuboid",
            "dimensions": [20.0, 16.0, 12.0],
            "pixels_per_micron": 0.5,
            "composition": copy.deepcopy(BASELINE_COMPOSITION),
            "decay_model": "simple",
        },
        "beam": {
            "type": "gaussian",
            "flux": 2.0e12,
            "fwhm": [20.0, 20.0],
            "energy": 12.1,
            "collimation": {"shape": "rectangular", "x": 40.0, "y": 40.0},
        },
        "wedges": [{"start": 0.0, "end": 90.0, "exposure_time": 10.0}],
    }


#: Keys that select which model a union resolves to.
DISCRIMINATORS = ("type", "method")


def _changes_discriminator(base: dict[str, Any], override: dict[str, Any]) -> bool:
    """Whether an override switches a union to a different member."""
    return any(
        key in override and key in base and override[key] != base[key]
        for key in DISCRIMINATORS
    )


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overrides`` into a copy of ``base``.

    An override that changes a discriminator replaces its subtree wholesale
    rather than merging: the target model has different fields, and a merged
    remnant of the old one would be rejected by ``extra="forbid"``.
    """
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, _Remove):
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            if _changes_discriminator(out[key], value):
                out[key] = copy.deepcopy(value)
            else:
                out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_for(case: Case) -> dict[str, Any]:
    """Build the full experiment configuration for a case."""
    return _merge(baseline(), case.overrides)


_CASES: list[Case] = [
    Case("000-baseline", "A", "cuboid, gaussian, RD3D composition, no escape"),
    # --- crystal geometry ---------------------------------------------------
    Case(
        "010-crystal-spherical",
        "A",
        "spherical crystal (icosphere mesh)",
        {
            "crystal": {
                "type": "spherical",
                "diameter": 20.0,
                "pixels_per_micron": 0.5,
                "composition": BASELINE_COMPOSITION,
            }
        },
    ),
    Case(
        "011-crystal-cylinder",
        "A",
        "cylinder; axis along y, DIM_Z ignored",
        {
            "crystal": {
                "type": "cylinder",
                "diameter": 20.0,
                "height": 16.0,
                "pixels_per_micron": 0.5,
                "composition": BASELINE_COMPOSITION,
            }
        },
    ),
    Case(
        "012-crystal-default-resolution",
        "A",
        "no PixelsPerMicron: RADDOSE-3D uses ten voxels across the first dimension",
        {"crystal": {"dimensions": [25.0, 16.0, 12.0], "pixels_per_micron": REMOVE}},
    ),
    Case(
        "013-crystal-polyhedron",
        "A",
        "a hexagonal prism read from an .obj file; with no PixelsPerMicron and "
        "no Dimensions, the Java's static default of 0.5",
        {
            "crystal": {
                "type": "polyhedron",
                "model_file": "tests/data/hexagonal_prism.obj",
                "composition": BASELINE_COMPOSITION,
            }
        },
    ),
    Case(
        "014-crystal-polyhedron-rotated",
        "A",
        "the prism tilted by AngleP and AngleL, with a placeholder Dimensions "
        "setting the default resolution to ten voxels across it",
        {
            "crystal": {
                "type": "polyhedron",
                "model_file": "tests/data/hexagonal_prism.obj",
                "dimensions": [16.0],
                "angle_p": 30.0,
                "angle_l": 20.0,
                "composition": BASELINE_COMPOSITION,
            }
        },
    ),
    # --- beam ---------------------------------------------------------------
    Case(
        "020-beam-tophat",
        "A",
        "top hat beam profile",
        {
            "beam": {
                "type": "tophat",
                "flux": 2.0e12,
                "energy": 12.1,
                "collimation": {"shape": "rectangular", "x": 40.0, "y": 40.0},
            }
        },
    ),
    Case(
        "021-beam-circular-collimation",
        "A",
        "circular collimation; guards the d2fbcf1 elliptical-aperture rewrite",
        {"beam": {"collimation": {"shape": "circular"}}},
    ),
    Case(
        "022-beam-elliptical-circular",
        "A",
        "circular collimation on a strongly elliptical beam",
        {
            "beam": {
                "fwhm": [10.0, 40.0],
                "collimation": {"shape": "circular", "x": 20.0, "y": 60.0},
            }
        },
    ),
    Case("023-beam-low-energy", "A", "6 keV", {"beam": {"energy": 6.0}}),
    Case("024-beam-high-energy", "A", "20 keV", {"beam": {"energy": 20.0}}),
    Case(
        "026-container-mixture",
        "A",
        "a 100 um Mylar container, from NIST's table for the compound",
        {
            "crystal": {
                "container": {
                    "type": "mixture",
                    "material": "mylar",
                    "thickness": 100.0,
                    "density": 1.4,
                }
            }
        },
    ),
    Case(
        "027-container-elemental",
        "A",
        "a 50 um Kapton container by formula, from NIST's per-element tables",
        {
            "crystal": {
                "container": {
                    "type": "elemental",
                    "elements": {"C": 22.0, "H": 10.0, "N": 2.0, "O": 5.0},
                    "thickness": 50.0,
                    "density": 1.42,
                }
            }
        },
    ),
    Case(
        "025-beam-uncollimated",
        "A",
        "no Collimation line: RADDOSE-3D's beam is then uncollimated",
        {"beam": {"collimation": REMOVE}},
    ),
    # --- composition --------------------------------------------------------
    Case(
        "036-composition-carbohydrates",
        "A",
        "NumCarb: carbohydrate residues in each monomer",
        {"crystal": {"composition": {"num_carb": 40}}},
    ),
    Case(
        "030-composition-average",
        "A",
        "generic average-protein coefficients",
        {"crystal": {"composition": {"method": "average"}}},
    ),
    Case(
        "031-composition-small-molecule",
        "A",
        "small molecule crystal",
        {
            "crystal": {
                "composition": {
                    "method": "small_molecule",
                    "unit_cell": [10.012, 18.808, 6.334],
                    "atoms": {"Mg": 1.0, "O": 3.0},
                    "num_monomers": 8,
                }
            }
        },
    ),
    Case(
        "032-composition-saxs",
        "A",
        "SAXS sample defined by protein concentration",
        {
            "crystal": {
                "composition": {
                    "method": "saxs",
                    "unit_cell": [100.0, 100.0, 100.0],
                    "protein_concentration": 10.0,
                    "num_residues": 129,
                    "solvent_fraction": 0.64,
                }
            }
        },
    ),
    Case(
        "033-composition-pdb",
        "A",
        "composition read from a local PDB file",
        {
            "crystal": {
                "composition": {
                    "method": "pdb",
                    "pdb": "tests/data/3I40.pdb",
                    "solvent_heavy_conc": {"P": 425.0},
                }
            }
        },
    ),
    # COD 1008123, bis(urea) antimony tetrafluoride oxide, public domain.
    # Heavy atoms (Sb), so absorption differs sharply from the organic
    # baseline. Case 035 covers the single-letter-tail formula that used to
    # crash the reference implementation's CIF parser.
    Case(
        "034-composition-cif",
        "A",
        "small-molecule composition from a CIF",
        {
            "crystal": {
                "composition": {
                    "method": "cif",
                    "cif": "tests/data/COD-1008123.cif",
                }
            }
        },
    ),
    # COD 1008775, urea, 'C H4 N2 O'. Specifically a formula ending in a bare
    # single-letter symbol: that crashed CoefCalcFromCIF until 2026-09-17, and
    # the crash was masked by a System.exit(0) that made the build look green.
    # Keeping it in the corpus means the Python port cannot inherit the bug.
    Case(
        "035-composition-cif-single-letter-tail",
        "A",
        "CIF whose formula ends in a bare single-letter symbol (urea)",
        {
            "crystal": {
                "composition": {
                    "method": "cif",
                    "cif": "tests/data/COD-1008775-urea.cif",
                }
            }
        },
    ),
    # --- decay models -------------------------------------------------------
    Case(
        "040-ddm-linear",
        "A",
        "linear diffraction decay model",
        {"crystal": {"decay_model": "linear"}},
    ),
    # DecayParam is required: DDMLeal zeroes all three parameters when any is
    # missing, which makes it degenerate to DDMSimple (no decay at all).
    # Values as used by the reference implementation's own DDM tests.
    Case(
        "041-ddm-leal",
        "A",
        "Leal diffraction decay model",
        {"crystal": {"decay_model": "leal", "decay_parameters": [0.3, 10.0, 0.03]}},
    ),
    Case(
        "042-ddm-bfactor",
        "A",
        "B-factor diffraction decay model",
        {"crystal": {"decay_model": "bfactor", "decay_parameters": [2.0, 3.0, 4.0]}},
    ),
    # --- crystal orientation ------------------------------------------------
    Case("050-rotation-anglep-45", "A", "AngleP 45", {"crystal": {"angle_p": 45.0}}),
    Case("051-rotation-anglel-45", "A", "AngleL 45", {"crystal": {"angle_l": 45.0}}),
    Case("052-rotation-anglep-90", "A", "AngleP 90", {"crystal": {"angle_p": 90.0}}),
    # --- exposure schedule --------------------------------------------------
    Case(
        "060-wedge-multiple",
        "A",
        "two consecutive wedges",
        {
            "wedges": [
                {"start": 0.0, "end": 90.0, "exposure_time": 10.0},
                {"start": 90.0, "end": 180.0, "exposure_time": 20.0},
            ]
        },
    ),
    Case(
        "061-wedge-translation",
        "A",
        "crystal translated during rotation",
        {
            "wedges": [
                {
                    "start": 0.0,
                    "end": 90.0,
                    "exposure_time": 10.0,
                    "translate_per_degree": [0.05, 0.0, 0.02],
                }
            ]
        },
    ),
    Case(
        "062-wedge-offset",
        "A",
        "start offset and rotation axis offset",
        {
            "wedges": [
                {
                    "start": 0.0,
                    "end": 90.0,
                    "exposure_time": 10.0,
                    "start_offset": [2.0, 1.0, 0.0],
                    "rot_axis_beam_offset": 3.0,
                }
            ]
        },
    ),
    Case(
        "063-wedge-full-rotation",
        "A",
        "full 360 degree rotation",
        {
            "wedges": [
                {
                    "start": 0.0,
                    "end": 360.0,
                    "exposure_time": 20.0,
                    "angular_resolution": 5.0,
                }
            ]
        },
    ),
    Case(
        "064-wedge-still",
        "A",
        "a still: start equals end, so the exposure does not rotate",
        {"wedges": [{"start": 45.0, "end": 45.0, "exposure_time": 10.0}]},
    ),
    Case(
        "065-wedge-still-decaying",
        "A",
        "a still with a decay model, where the number of steps is observable",
        {
            "crystal": {"decay_model": "bfactor"},
            "wedges": [{"start": 45.0, "end": 45.0, "exposure_time": 10.0}],
        },
    ),
    # --- solvent ------------------------------------------------------------
    Case(
        "066-wedge-resolution-overridden",
        "A",
        "AngularResolution 5 on a 30 degree wedge: six steps, so RADDOSE-3D uses 2",
        {
            "wedges": [
                {
                    "start": 0.0,
                    "end": 30.0,
                    "exposure_time": 10.0,
                    "angular_resolution": 5.0,
                }
            ]
        },
    ),
    Case(
        "070-solvent-fraction-computed",
        "A",
        "solvent fraction derived from the composition, not stated",
        {"crystal": {"composition": {"solvent_fraction": REMOVE}}},
    ),
    # --- escape modelling (tier B: Monte Carlo) -----------------------------
    Case(
        "100-escape-pe",
        "B",
        "photoelectron escape only",
        {"crystal": {"escape": {"photoelectron": True}}},
    ),
    Case(
        "101-escape-fl",
        "B",
        "fluorescence escape only",
        {"crystal": {"escape": {"fluorescence": True}}},
    ),
    Case(
        "102-escape-pe-fl",
        "B",
        "photoelectron and fluorescence escape",
        {"crystal": {"escape": {"photoelectron": True, "fluorescence": True}}},
    ),
    Case(
        "103-escape-surrounding",
        "B",
        "escape into a surrounding medium",
        {
            "crystal": {
                "escape": {
                    "photoelectron": True,
                    "fluorescence": True,
                    "surrounding": True,
                }
            }
        },
    ),
    Case(
        "104-escape-rotated-surrounding",
        "B",
        "rotated crystal with surrounding; guards the cryo grid frame fix",
        {
            "crystal": {
                "angle_p": 90.0,
                "escape": {"photoelectron": True, "surrounding": True},
            }
        },
    ),
]


def all_cases() -> list[Case]:
    """Every conformance case, in id order."""
    return sorted(_CASES, key=lambda c: c.case_id)
