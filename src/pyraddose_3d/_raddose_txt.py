"""Render an experiment as a legacy RADDOSE-3D ``.txt`` input file.

This exists so that YAML can be the single source of truth. A conformance case
is written once as YAML, and the ``.txt`` handed to the Java reference
implementation is generated from it -- so the two inputs cannot drift apart,
and a conformance failure is unambiguous.

Only emitting is supported. Parsing the legacy format would mean
reimplementing an ANTLR grammar, which this port deliberately does not do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyraddose_3d.models.beam import GaussianBeam, TophatBeam
from pyraddose_3d.models.composition import (
    AverageComposition,
    SaxsComposition,
    SequenceComposition,
    SmallMoleculeComposition,
    UnitCellComposition,
)
from pyraddose_3d.models.crystal import (
    CuboidCrystal,
    CylinderCrystal,
    PolyhedronCrystal,
    SphericalCrystal,
)

if TYPE_CHECKING:
    from pyraddose_3d.models.composition import Composition
    from pyraddose_3d.models.crystal import Crystal, EscapeOptions
    from pyraddose_3d.models.experiment import Experiment
    from pyraddose_3d.models.wedge import Wedge

__all__ = ["emit_raddose_txt"]

# AbsCoefCalc keyword for each composition variant. Note "EXPSM" is the
# keyword for a CIF-based calculation: the grammar's token *named* CIF matches
# the literal "EXPSM", which is why "AbsCoefCalc CIF" can never parse.
_ABSCOEFCALC = {
    "average": "Average",
    "unit_cell": "RD3D",
    "small_molecule": "SmallMole",
    "saxs": "SAXS",
    "sequence": "Sequence",
}

_DECAY_MODEL = {
    "simple": "Simple",
    "linear": "Linear",
    "leal": "Leal",
    "bfactor": "BFactor",
}


def _number(value: float) -> str:
    """Format a number without losing precision or gaining noise."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _elements(amounts: dict[str, float]) -> str:
    """Format an element/amount mapping as RADDOSE-3D expects it."""
    return " ".join(f"{sym} {_number(n)}" for sym, n in amounts.items())


def _composition_lines(composition: Composition) -> list[str]:
    """Emit the AbsCoefCalc directive and its supporting parameters."""
    lines = [f"AbsCoefCalc {_ABSCOEFCALC[composition.method]}"]

    if isinstance(composition, AverageComposition):
        return lines

    cell = composition.unit_cell
    cell_text = " ".join(_number(v) for v in cell)
    if composition.unit_cell_angles is not None:
        cell_text += " " + " ".join(_number(v) for v in composition.unit_cell_angles)
    lines.append(f"UnitCell {cell_text}")

    if isinstance(composition, SmallMoleculeComposition):
        lines.append(f"SmallMoleAtoms {_elements(composition.atoms)}")
        lines.append(f"NumMonomers {composition.num_monomers}")
        return lines

    if isinstance(composition, SequenceComposition):
        lines.append(f"SeqFile {composition.sequence_file}")
        lines.append(f"NumMonomers {composition.num_monomers}")
    elif isinstance(composition, UnitCellComposition):
        lines.append(f"NumMonomers {composition.num_monomers}")
        lines.append(f"NumResidues {composition.num_residues}")
        if composition.num_rna:
            lines.append(f"NumRNA {composition.num_rna}")
        if composition.num_dna:
            lines.append(f"NumDNA {composition.num_dna}")
    elif isinstance(composition, SaxsComposition):
        lines.append(f"NumResidues {composition.num_residues}")
        if composition.num_rna:
            lines.append(f"NumRNA {composition.num_rna}")
        if composition.num_dna:
            lines.append(f"NumDNA {composition.num_dna}")
        lines.append(f"ProteinConc {_number(composition.protein_concentration)}")

    if composition.protein_heavy_atoms:
        lines.append(f"ProteinHeavyAtoms {_elements(composition.protein_heavy_atoms)}")
    if composition.solvent_heavy_conc:
        lines.append(f"SolventHeavyConc {_elements(composition.solvent_heavy_conc)}")
    if composition.solvent_fraction is not None:
        lines.append(f"SolventFraction {_number(composition.solvent_fraction)}")
    return lines


def _escape_lines(escape: EscapeOptions) -> list[str]:
    """Emit the escape-modelling flags that are switched on."""
    lines: list[str] = []
    if escape.photoelectron:
        lines.append("CalculatePEEscape TRUE")
    if escape.fluorescence:
        lines.append("CalculateFLEscape TRUE")
    if escape.surrounding:
        lines.append("CalcSurrounding TRUE")
    if escape.pe_resolution is not None:
        lines.append(f"PEResolution {escape.pe_resolution}")
    if escape.fl_resolution is not None:
        lines.append(f"FLResolution {escape.fl_resolution}")
    return lines


def _crystal_lines(crystal: Crystal) -> list[str]:
    """Emit the whole Crystal block."""
    lines = ["Crystal"]

    if isinstance(crystal, CuboidCrystal):
        lines.append("Type Cuboid")
        lines.append(f"Dimensions {' '.join(_number(v) for v in crystal.dimensions)}")
    elif isinstance(crystal, SphericalCrystal):
        d = _number(crystal.diameter)
        lines.append("Type Spherical")
        lines.append(f"Dimensions {d} {d} {d}")
    elif isinstance(crystal, CylinderCrystal):
        # Dimensions are (diameter, height, ignored) -- see CylinderCrystal.
        lines.append("Type Cylinder")
        lines.append(
            f"Dimensions {_number(crystal.diameter)} {_number(crystal.height)} "
            f"{_number(crystal.diameter)}"
        )
    elif isinstance(crystal, PolyhedronCrystal):
        lines.append("Type Polyhedron")
        lines.append(f"WireframeType {crystal.wireframe_type}")
        lines.append(f"ModelFile {crystal.model_file}")

    lines.append(f"PixelsPerMicron {_number(crystal.pixels_per_micron)}")
    if crystal.angle_p:
        lines.append(f"AngleP {_number(crystal.angle_p)}")
    if crystal.angle_l:
        lines.append(f"AngleL {_number(crystal.angle_l)}")

    lines.extend(_composition_lines(crystal.composition))

    lines.append(f"DiffractionDecayModel {_DECAY_MODEL[crystal.decay_model]}")
    if crystal.decay_parameters is not None:
        params = " ".join(_number(v) for v in crystal.decay_parameters)
        lines.append(f"DecayParam {params}")

    lines.extend(_escape_lines(crystal.escape))
    return lines


def _beam_lines(beam: GaussianBeam | TophatBeam) -> list[str]:
    """Emit the whole Beam block."""
    lines = ["Beam"]
    if isinstance(beam, GaussianBeam):
        lines.append("Type Gaussian")
        lines.append(f"Flux {_number(beam.flux)}")
        lines.append(f"FWHM {' '.join(_number(v) for v in beam.fwhm)}")
    else:
        lines.append("Type TopHat")
        lines.append(f"Flux {_number(beam.flux)}")
    lines.append(f"Energy {_number(beam.energy)}")
    lines.append(
        f"Collimation {beam.collimation.shape.capitalize()} "
        f"{_number(beam.collimation.x)} {_number(beam.collimation.y)}"
    )
    return lines


def _wedge_lines(wedge: Wedge) -> list[str]:
    """Emit one Wedge block."""
    lines = [f"Wedge {_number(wedge.start)} {_number(wedge.end)}"]
    lines.append(f"ExposureTime {_number(wedge.exposure_time)}")
    if wedge.angular_resolution is not None:
        lines.append(f"AngularResolution {_number(wedge.angular_resolution)}")
    if wedge.start_offset is not None:
        lines.append(f"StartOffset {' '.join(_number(v) for v in wedge.start_offset)}")
    if wedge.translate_per_degree is not None:
        values = " ".join(_number(v) for v in wedge.translate_per_degree)
        lines.append(f"TranslatePerDegree {values}")
    if wedge.rot_axis_beam_offset is not None:
        lines.append(f"RotAxBeamOffset {_number(wedge.rot_axis_beam_offset)}")
    return lines


def emit_raddose_txt(experiment: Experiment) -> str:
    """Render an experiment as a legacy RADDOSE-3D input file.

    Parameters
    ----------
    experiment : Experiment
        The experiment to render.

    Returns
    -------
    str
        Contents of an equivalent ``.txt`` input file, newline terminated.
    """
    blocks: list[list[str]] = [
        _crystal_lines(experiment.crystal),
        _beam_lines(experiment.beam),
    ]
    blocks.extend(_wedge_lines(wedge) for wedge in experiment.wedges)
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"
