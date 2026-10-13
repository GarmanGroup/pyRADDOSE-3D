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
    Reproducible. Compared value by value. This includes escape modelling:
    it is random, but the port draws from ``java.util.Random`` in exactly
    the Java's order, so with the corpus seed it reproduces the Java's
    result rather than only its statistics.
``B``
    Random in a way the port cannot reproduce draw for draw, and so
    compared statistically. No case needs it at present; the Monte Carlo,
    XFEL and MicroED subprograms, if ported, would.
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

#: Seed for every case. Escape modelling draws from it; pinned everywhere.
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


def _microed_crystal(
    composition: dict[str, Any] | None = None,
    dimensions: list[float] | None = None,
) -> dict[str, Any]:
    """A thin cuboid plate for the MicroED engine."""
    return {
        "type": "cuboid",
        "dimensions": dimensions or [2.0, 2.0, 0.3],
        "pixels_per_micron": 20.0,
        "subprogram": "microed",
        "composition": copy.deepcopy(composition or BASELINE_COMPOSITION),
    }


def _electron_beam(
    energy: float = 300.0,
    exposure: float = 1.0,
    collimation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A MicroED beam: a top-hat of electrons."""
    return {
        "type": "electron",
        "energy": energy,
        "exposure": exposure,
        "collimation": collimation or {"shape": "rectangular", "x": 4.0, "y": 4.0},
    }


#: The overrides every MicroED case starts from.
MICROED: dict[str, Any] = {
    "crystal": _microed_crystal(),
    "beam": _electron_beam(),
    "wedges": [{"start": 0.0, "end": 0.0, "exposure_time": 1.0}],
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
        "028-beam-pink",
        "A",
        "a pink beam: 200 energies across a 0.5 keV FWHM",
        {"beam": {"energy_fwhm": 0.5}},
    ),
    Case(
        "029-beam-pink-across-edge",
        "A",
        "a pink beam straddling the zinc K edge, so the crystal absorbs "
        "differently on either side of it",
        {"beam": {"energy": 9.7, "energy_fwhm": 0.3}},
    ),
    Case(
        "025-beam-uncollimated",
        "A",
        "no Collimation line: RADDOSE-3D's beam is then uncollimated",
        {"beam": {"collimation": REMOVE}},
    ),
    # An off-centre peak on a ramp with a dead corner, 20 by 14 pixels of
    # 1.5 by 1.4 um: a flipped or transposed image reads differently.
    Case(
        "080-beam-experimental",
        "A",
        "measured beam profile from a PGM image, interpolated bilinearly",
        {
            "beam": {
                "type": "experimental",
                "flux": 2.0e12,
                "energy": 12.1,
                "file": "tests/data/beam_profile.pgm",
                "pixel_size": [1.5, 1.4],
            }
        },
    ),
    # The image is 24 by 30.8 um. RADDOSE-3D used to ignore a collimation's
    # sizes here (UPSTREAM_FIXES.md item 12): Circular meant the ellipse
    # inscribed in the image, and Rectangular nothing.
    Case(
        "081-beam-experimental-circular",
        "A",
        "measured beam behind a circular aperture smaller than its image",
        {
            "beam": {
                "type": "experimental",
                "flux": 2.0e12,
                "energy": 12.1,
                "file": "tests/data/beam_profile.pgm",
                "pixel_size": [1.5, 1.4],
                "collimation": {"shape": "circular", "x": 18.0, "y": 20.0},
            }
        },
    ),
    Case(
        "082-beam-experimental-rectangular",
        "A",
        "measured beam behind a rectangular aperture smaller than its image",
        {
            "beam": {
                "type": "experimental",
                "flux": 2.0e12,
                "energy": 12.1,
                "file": "tests/data/beam_profile.pgm",
                "pixel_size": [1.5, 1.4],
                "collimation": {"shape": "rectangular", "x": 16.0, "y": 14.0},
            }
        },
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
    # PDB entries, public domain, chosen for what 3I40 lacks: NCS operators
    # to apply, nucleic acids, partly occupied ligands and residue codes the
    # residue table does not have. They pin the PDB route to the Java beyond
    # one simple protein, which matters now that the port reads the files
    # with Gemmi rather than RADDOSE-3D's own way.
    Case(
        "071-composition-pdb-ncs-rna-ligands",
        "A",
        "PDB 1A34: 14 NCS operators to apply, RNA chains, partly occupied ligands",
        {
            "crystal": {
                "composition": {
                    "method": "pdb",
                    "pdb": "tests/data/1A34.pdb",
                    "solvent_heavy_conc": {"P": 425.0},
                }
            }
        },
    ),
    Case(
        "072-composition-pdb-capsid-ncs",
        "A",
        "PDB 2BUK: a capsid with 59 NCS operators to apply",
        {
            "crystal": {
                "composition": {
                    "method": "pdb",
                    "pdb": "tests/data/2BUK.pdb",
                    "solvent_heavy_conc": {"P": 425.0},
                }
            }
        },
    ),
    Case(
        "073-composition-pdb-modified-nucleotides",
        "A",
        "PDB 1EHZ: tRNA whose modified nucleotides are not in the residue table",
        {
            "crystal": {
                "composition": {
                    "method": "pdb",
                    "pdb": "tests/data/1EHZ.pdb",
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
    # Insulin's two chains, then a DNA and an RNA strand. Each header says
    # what follows it; the port used to drop headers and read every residue
    # as protein, which ACGU, all valid amino acid codes, did not reveal.
    Case(
        "037-composition-sequence",
        "A",
        "residues from a sequence file of protein, DNA and RNA chains",
        {
            "crystal": {
                "composition": {
                    "method": "sequence",
                    "unit_cell": [78.02, 78.02, 78.02],
                    "num_monomers": 24,
                    "sequence_file": "tests/data/insulin_and_nucleic_acids.fasta",
                    "protein_heavy_atoms": {"Zn": 0.333},
                    "solvent_heavy_conc": {"P": 425.0},
                }
            }
        },
    ),
    # No unit cell: CoefCalcSAXS's 1000 A box of solution.
    Case(
        "038-composition-saxs-sequence",
        "A",
        "SAXS sample whose monomer is read from a sequence file",
        {
            "crystal": {
                "composition": {
                    "method": "saxs_sequence",
                    "protein_concentration": 10.0,
                    "sequence_file": "tests/data/insulin_and_nucleic_acids.fasta",
                    "solvent_heavy_conc": {"Na": 150.0},
                }
            }
        },
    ),
    # RADDOSE-3D read NumCarb here and never added the carbohydrate, until
    # UPSTREAM_FIXES.md item 8.
    Case(
        "039-composition-saxs-sequence-carbohydrates",
        "A",
        "SAXS from a sequence, with carbohydrate residues on each monomer",
        {
            "crystal": {
                "composition": {
                    "method": "saxs_sequence",
                    "protein_concentration": 10.0,
                    "sequence_file": "tests/data/insulin_and_nucleic_acids.fasta",
                    "num_carb": 40,
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
    # A wedge that runs backwards. RADDOSE-3D counted its images as
    # sign * (int)(span / step + 1) of a negative span, two short, and
    # tested its requested step against a negative span, so ignored it.
    Case(
        "067-wedge-reversed",
        "A",
        "a wedge from 90 to 0 degrees, asking for 3 degree steps",
        {
            "wedges": [
                {
                    "start": 90.0,
                    "end": 0.0,
                    "exposure_time": 10.0,
                    "angular_resolution": 3.0,
                }
            ]
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
    # --- escape modelling ---------------------------------------------------
    # Random, but reproducibly: the port draws from java.util.Random in the
    # Java's order, so with the corpus seed these match value by value.
    Case(
        "100-escape-pe",
        "A",
        "photoelectron escape only",
        {"crystal": {"escape": {"photoelectron": True}}},
    ),
    Case(
        "101-escape-fl",
        "A",
        "fluorescence escape only",
        {"crystal": {"escape": {"fluorescence": True}}},
    ),
    Case(
        "102-escape-pe-fl",
        "A",
        "photoelectron and fluorescence escape",
        {"crystal": {"escape": {"photoelectron": True, "fluorescence": True}}},
    ),
    Case(
        "103-escape-surrounding",
        "A",
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
    # The surrounding's extent follows the beam's narrowest dimension, which
    # for a measured beam is its image's.
    Case(
        "108-escape-surrounding-experimental-beam",
        "A",
        "escape into a surrounding sized by an experimental beam's image",
        {
            "beam": {
                "type": "experimental",
                "flux": 2.0e12,
                "energy": 12.1,
                "file": "tests/data/beam_profile.pgm",
                "pixel_size": [1.5, 1.4],
            },
            "crystal": {
                "escape": {"photoelectron": True, "surrounding": True},
            },
        },
    ),
    Case(
        "104-escape-rotated-surrounding",
        "A",
        "rotated crystal with surrounding; guards the cryo grid frame fix",
        {
            "crystal": {
                "angle_p": 90.0,
                "escape": {"photoelectron": True, "surrounding": True},
            }
        },
    ),
    Case(
        "105-escape-pink",
        "A",
        "escape under a pink beam: a draw per voxel for every energy sampled; "
        "a short wedge, as every image costs 200",
        {
            "crystal": {"escape": {"photoelectron": True, "fluorescence": True}},
            "beam": {"energy_fwhm": 0.3},
            "wedges": [{"start": 0.0, "end": 10.0, "exposure_time": 2.0}],
        },
    ),
    Case(
        "106-escape-vertical-goniometer",
        "A",
        "GoniometerAxis 90: the polarisation biases photoelectrons along x",
        {"crystal": {"escape": {"photoelectron": True, "goniometer_axis": 90.0}}},
    ),
    Case(
        "107-escape-surrounding-salt",
        "A",
        "a surrounding with heavy atoms in it",
        {
            "crystal": {
                "escape": {
                    "photoelectron": True,
                    "surrounding": True,
                    "surrounding_heavy_conc": {"Na": 500.0, "Cl": 500.0},
                }
            }
        },
    ),
    # The next three exist to show RADDOSE-3D defects before they are fixed
    # (UPSTREAM_FIXES.md), so that each fix appears as a change to them.
    #
    # Photoelectron escape alone loses the relaxation energy other than the
    # K-shell Auger part: here, a SeMet crystal just above the Se K edge,
    # about 15% of the absorbed energy.
    Case(
        "109-escape-pe-semet-above-edge",
        "A",
        "photoelectron escape alone, SeMet above the Se K edge",
        {
            "beam": {"energy": 12.7},
            "crystal": {
                "composition": {"protein_heavy_atoms": {"Se": 1.0, "S": 5.0}},
                "escape": {"photoelectron": True},
            },
        },
    ),
    # Iron's L-shell fluorescence has no tabulated energy, so its escape
    # distances are NaN.
    Case(
        "110-escape-fl-iron",
        "A",
        "fluorescence escape from iron, whose L-shell lines are untabulated",
        {
            "crystal": {
                "composition": {"protein_heavy_atoms": {"Fe": 2.0, "S": 6.0}},
                "escape": {"fluorescence": True},
            },
        },
    ),
    # A pink beam gives the surrounding its whole exposure at every one of
    # its 200 energies. Two images: the surrounding runs 200 times per image.
    Case(
        "111-escape-surrounding-pink",
        "A",
        "a pink beam with a surrounding",
        {
            "beam": {"energy_fwhm": 0.3},
            "crystal": {
                "escape": {"photoelectron": True, "surrounding": True},
            },
            "wedges": [{"start": 0.0, "end": 2.0, "exposure_time": 1.0}],
        },
    ),
    # --- MicroED (Subprogram EMED) -----------------------------------------
    # Deterministic, so every case is tier A. The crystal is a thin plate in
    # an electron beam; MicroED ignores the wedge, which the grammar requires.
    Case("120-microed-baseline", "A", "MicroED: protein plate, 300 keV", MICROED),
    Case(
        "121-microed-circular-selenium",
        "A",
        "MicroED: circular beam inside the crystal, 200 keV, selenium",
        {
            **MICROED,
            "crystal": _microed_crystal(
                composition={
                    **BASELINE_COMPOSITION,
                    "protein_heavy_atoms": {"Se": 2.0, "S": 4.0},
                },
                dimensions=[3.0, 3.0, 0.2],
            ),
            "beam": _electron_beam(
                energy=200.0,
                exposure=2.5,
                collimation={"shape": "circular", "x": 1.0, "y": 1.5},
            ),
        },
    ),
    Case(
        "122-microed-uneven-slices",
        "A",
        "MicroED: a thickness that is not a whole number of 5 nm slices",
        {
            **MICROED,
            "crystal": _microed_crystal(dimensions=[3.0, 3.0, 0.1237]),
            "beam": _electron_beam(energy=120.0),
        },
    ),
    Case(
        "123-microed-electrons-stop",
        "A",
        "MicroED: 10 keV electrons that stop inside the crystal",
        {
            **MICROED,
            "crystal": _microed_crystal(dimensions=[3.0, 3.0, 2.0]),
            "beam": _electron_beam(energy=10.0),
        },
    ),
    Case(
        "124-microed-cylinder",
        "A",
        "MicroED: cylinder, its thickness less 0.001 nm",
        {
            **MICROED,
            "crystal": {
                "type": "cylinder",
                "diameter": 0.4,
                "height": 1.5,
                "pixels_per_micron": 20.0,
                "subprogram": "microed",
                "composition": BASELINE_COMPOSITION,
            },
        },
    ),
    Case(
        "125-microed-rotated",
        "A",
        "MicroED: a rotated plate, sized by its rotated extent",
        {
            **MICROED,
            "crystal": {**_microed_crystal(), "angle_p": 30.0, "angle_l": 10.0},
        },
    ),
    # RADDOSE-3D's own example input, ED_input.txt.
    Case(
        "126-microed-small-molecule",
        "A",
        "MicroED: RADDOSE-3D's ED_input.txt, a small molecule",
        {
            **MICROED,
            "crystal": _microed_crystal(
                composition={
                    "method": "small_molecule",
                    "unit_cell": [13.0824, 18.808, 29.608],
                    "atoms": {"C": 10.0, "O": 50.0},
                    "num_monomers": 1,
                },
                dimensions=[0.2, 0.2, 0.2],
            ),
            "beam": _electron_beam(
                collimation={"shape": "circular", "x": 2.0, "y": 2.0}
            ),
        },
    ),
    Case(
        "127-microed-sodium-nan",
        "A",
        "MicroED: sodium, which the GOS model has no binding energy for",
        {
            **MICROED,
            "crystal": _microed_crystal(
                composition={
                    **BASELINE_COMPOSITION,
                    "solvent_heavy_conc": {"P": 425.0, "Na": 150.0},
                }
            ),
        },
    ),
    # The composition routes other than the unit cell's. Until RADDOSE-3D
    # 11d03e5 its electron calculations kept their own count of the atoms,
    # which the PDB route never multiplied by the molecules in the cell, and
    # the sequence and CIF routes never filled.
    Case(
        "128-microed-pdb",
        "A",
        "MicroED: composition from PDB 3I40",
        {
            **MICROED,
            "crystal": _microed_crystal(
                composition={
                    "method": "pdb",
                    "pdb": "tests/data/3I40.pdb",
                    "solvent_heavy_conc": {"P": 425.0},
                }
            ),
        },
    ),
    Case(
        "129-microed-pdb-ncs-ligands",
        "A",
        "MicroED: composition from PDB 1A34, NCS operators and ligands",
        {
            **MICROED,
            "crystal": _microed_crystal(
                composition={
                    "method": "pdb",
                    "pdb": "tests/data/1A34.pdb",
                    "solvent_heavy_conc": {"P": 425.0},
                }
            ),
        },
    ),
    Case(
        "130-microed-sequence",
        "A",
        "MicroED: composition from a sequence file",
        {
            **MICROED,
            "crystal": _microed_crystal(
                composition={
                    "method": "sequence",
                    "unit_cell": [78.02, 78.02, 78.02],
                    "num_monomers": 24,
                    "sequence_file": "tests/data/insulin_and_nucleic_acids.fasta",
                    "protein_heavy_atoms": {"Zn": 0.333},
                    "solvent_heavy_conc": {"P": 425.0},
                }
            ),
        },
    ),
    Case(
        "131-microed-cif",
        "A",
        "MicroED: composition from a CIF, urea",
        {
            **MICROED,
            "crystal": _microed_crystal(
                composition={
                    "method": "cif",
                    "cif": "tests/data/COD-1008775-urea.cif",
                }
            ),
        },
    ),
    Case(
        "132-microed-aluminium-nan",
        "A",
        "MicroED: aluminium, whose blank M1 binding energy crashed RADDOSE-3D",
        {
            **MICROED,
            "crystal": _microed_crystal(
                composition={
                    **BASELINE_COMPOSITION,
                    "protein_heavy_atoms": {"Al": 1.0, "S": 6.0},
                }
            ),
        },
    ),
    Case(
        "133-microed-thinner-than-a-slice",
        "A",
        "MicroED: 4 nm thick, less than one 5 nm slice",
        {**MICROED, "crystal": _microed_crystal(dimensions=[2.0, 2.0, 0.004])},
    ),
]


def all_cases() -> list[Case]:
    """Every conformance case, in id order."""
    return sorted(_CASES, key=lambda c: c.case_id)
