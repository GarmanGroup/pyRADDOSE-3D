"""RADDOSE-3D's ``.txt`` input format, in both directions.

Writing lets YAML be the single source of truth for the conformance corpus:
each case is written once as YAML and the ``.txt`` handed to the Java is
generated from it, so the two cannot drift apart.

Reading lets existing input files run unchanged. The format is specified by
an ANTLR grammar, ``lib/antlrworks-parsergenerator/Inputfile.g`` in the Java
repository, and :func:`parse_raddose_txt` follows it rule by rule rather than
approximating it. Four properties of that grammar are easy to get wrong:

- **It is not line-based.** Newlines are whitespace, so a keyword's values
  may continue on the next line.
- **Keywords are case-insensitive and some have several spellings**:
  ``AbsCoefCalc`` or ``CoefCalc``, ``Dimension`` or ``Dimensions``,
  ``RD3D`` or ``RD3``, ``DDM`` or ``DiffractionDecayModel``.
- **Omitted settings take RADDOSE-3D's defaults**, which are what the models
  default to; settings RADDOSE-3D requires are required here too. A missing
  value is never filled in with a guess.
- **Comments** start with ``#``, ``!`` or ``//`` and run to the end of the
  line. ``//`` inside a path does not start one, because ``/`` is part of a
  word in the grammar.

What it refuses, and why
------------------------
Anything this port cannot represent is refused with the reason, never
dropped. The exceptions are keywords confirmed to reach nothing but the
Monte Carlo and XFEL subprograms -- ``GoniometerAxis``,
``PolarisationDirection``, and ``Subprogram RD3D`` -- which a RADDOSE-3D MX
run ignores and so does this one. That was checked keyword by keyword rather
than assumed: ``EnergyFWHM`` looks like one of them and is not, because it
turns the MX calculation into a 200-energy pink-beam sum.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyraddose_3d.models.beam import GaussianBeam, TophatBeam
from pyraddose_3d.models.composition import (
    AverageComposition,
    CifComposition,
    PdbComposition,
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

__all__ = [
    "RaddoseTxtError",
    "RaddoseTxtWarning",
    "emit_raddose_txt",
    "parse_raddose_txt",
]

# AbsCoefCalc keyword for each composition variant. Note "EXPSM" is the
# keyword for a CIF-based calculation: the grammar's token *named* CIF matches
# the literal "EXPSM", which is why "AbsCoefCalc CIF" can never parse.
_ABSCOEFCALC = {
    "average": "Average",
    "unit_cell": "RD3D",
    "small_molecule": "SmallMole",
    "saxs": "SAXS",
    "sequence": "Sequence",
    # The token named PDB matches the literal "EXP"; see PdbComposition.
    "pdb": "EXP",
    # Likewise the token named CIF matches "EXPSM".
    "cif": "EXPSM",
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

    if isinstance(composition, CifComposition):
        lines.append(f"CIF {composition.cif}")
        if composition.solvent_heavy_conc:
            lines.append(
                f"SolventHeavyConc {_elements(composition.solvent_heavy_conc)}"
            )
        return lines

    if isinstance(composition, PdbComposition):
        lines.append(f"PDB {composition.pdb}")
        if composition.solvent_heavy_conc:
            lines.append(
                f"SolventHeavyConc {_elements(composition.solvent_heavy_conc)}"
            )
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
        if composition.num_carb:
            lines.append(f"NumCarb {composition.num_carb}")
    elif isinstance(composition, UnitCellComposition):
        lines.append(f"NumMonomers {composition.num_monomers}")
        lines.append(f"NumResidues {composition.num_residues}")
        if composition.num_rna:
            lines.append(f"NumRNA {composition.num_rna}")
        if composition.num_dna:
            lines.append(f"NumDNA {composition.num_dna}")
        if composition.num_carb:
            lines.append(f"NumCarb {composition.num_carb}")
    elif isinstance(composition, SaxsComposition):
        lines.append(f"NumResidues {composition.num_residues}")
        if composition.num_rna:
            lines.append(f"NumRNA {composition.num_rna}")
        if composition.num_dna:
            lines.append(f"NumDNA {composition.num_dna}")
        if composition.num_carb:
            lines.append(f"NumCarb {composition.num_carb}")
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

    if crystal.pixels_per_micron is not None:
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
    if beam.collimation is not None:
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
    if wedge.max_resolution != 2.0:
        lines.append(f"MaxResolution {_number(wedge.max_resolution)}")
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


# ============================================================================
# Reading
# ============================================================================


class RaddoseTxtError(ValueError):
    """A RADDOSE-3D input file that cannot become an experiment."""


#: A number, exactly as the grammar's ``FLOAT`` token defines one.
_FLOAT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
#: What ``Integer.parseInt`` accepts. Counts go through it in the Java, so
#: ``NumMonomers 24.0`` is an error there and here.
_INTEGER = re.compile(r"[+-]?\d+")
#: The grammar's ``ELEMENT`` token: one or two letters. No keyword is that
#: short, so this is unambiguous.
_ELEMENT = re.compile(r"[A-Za-z]{1,2}")

_BLOCKS = frozenset({"CRYSTAL", "BEAM", "WEDGE"})


@dataclass(frozen=True)
class _Token:
    text: str
    line: int


#: Characters some token can contain: the grammar's STRING set (ASCII letters,
#: digits and ``. $ - _ : \\ /``) plus ``+``, which FLOAT accepts as a sign.
_WORD_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.$-_:\\/+"
)


class RaddoseTxtWarning(UserWarning):
    """Something the Java reports and then carries on past."""


def _tokenise(text: str, source: str) -> list[_Token]:
    """Split into words and drop comments, as the grammar's lexer does.

    ``#`` and ``!`` cannot be part of a word, so they start a comment
    wherever they appear. ``/`` can be, so ``//`` starts a comment only at
    the beginning of a word -- a path containing ``//`` is still one word.

    A character no token can contain is skipped with a warning, which is
    what ANTLR's lexer does: it reports "no viable alternative at character"
    and carries on, and RADDOSE-3D runs. One of the example inputs that ships
    with it, ``input-b3x14.txt``, has ``AngularResolution 0.5`` followed by a
    stray ``ß``, and runs at 0.5.
    """
    tokens = []
    for number, line in enumerate(text.splitlines(), start=1):
        position, length = 0, len(line)
        while position < length:
            character = line[position]
            if character in " \t\r":
                position += 1
                continue
            if character in "#!" or line.startswith("//", position):
                break
            if character not in _WORD_CHARACTERS:
                warnings.warn(
                    RaddoseTxtWarning(
                        f"{source}, line {number}: skipped {character!r}, which "
                        f"is not part of any keyword or value; RADDOSE-3D "
                        f"skips it too"
                    ),
                    stacklevel=4,
                )
                position += 1
                continue
            end = position
            while end < length and line[end] in _WORD_CHARACTERS:
                end += 1
            tokens.append(_Token(line[position:end], number))
            position = end
    return tokens


class _Stream:
    """The token sequence, with errors that name the file and line."""

    def __init__(self, tokens: list[_Token], source: str) -> None:
        self._tokens = tokens
        self._index = 0
        self.source = source

    def peek(self, offset: int = 0) -> _Token | None:
        index = self._index + offset
        return self._tokens[index] if index < len(self._tokens) else None

    def take(self) -> _Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def done(self) -> bool:
        return self._index >= len(self._tokens)

    def error(self, line: int, message: str) -> RaddoseTxtError:
        return RaddoseTxtError(f"{self.source}, line {line}: {message}")


def _is_number(token: _Token | None) -> bool:
    return token is not None and _FLOAT.fullmatch(token.text) is not None


def _read_number(stream: _Stream, keyword: _Token) -> float:
    if not _is_number(stream.peek()):
        found = stream.peek()
        raise stream.error(
            keyword.line,
            f"{keyword.text} needs a number, "
            + (f"found {found.text!r}" if found else "but the file ends"),
        )
    return float(stream.take().text)


def _read_numbers(
    stream: _Stream, keyword: _Token, least: int, most: int
) -> list[float]:
    values: list[float] = []
    while len(values) < most and _is_number(stream.peek()):
        values.append(float(stream.take().text))
    if len(values) < least:
        expected = str(least) if least == most else f"{least} to {most}"
        raise stream.error(
            keyword.line,
            f"{keyword.text} needs {expected} numbers, found {len(values)}",
        )
    return values


def _read_count(stream: _Stream, keyword: _Token) -> int:
    if not _is_number(stream.peek()):
        return int(_read_number(stream, keyword))  # raises with the message
    token = stream.take()
    if not _INTEGER.fullmatch(token.text):
        raise stream.error(
            keyword.line,
            f"{keyword.text} must be a whole number, not {token.text!r} -- "
            f"RADDOSE-3D reads it with Integer.parseInt",
        )
    return int(token.text)


def _read_word(stream: _Stream, keyword: _Token) -> str:
    found = stream.peek()
    if found is None or found.text.upper() in _BLOCKS:
        raise stream.error(keyword.line, f"{keyword.text} needs a value")
    return stream.take().text


def _read_elements(stream: _Stream, keyword: _Token) -> dict[str, float]:
    amounts: dict[str, float] = {}
    while (
        (symbol := stream.peek()) is not None
        and _ELEMENT.fullmatch(symbol.text)
        and _is_number(stream.peek(1))
    ):
        stream.take()
        if symbol.text in amounts:
            # The Java keeps both in a list; what it then does with a repeat
            # depends on the calculation. Refused rather than guessed.
            raise stream.error(
                symbol.line, f"{symbol.text} appears twice in {keyword.text}"
            )
        amounts[symbol.text] = float(stream.take().text)
    if not amounts:
        raise stream.error(
            keyword.line, f"{keyword.text} needs element and amount pairs, e.g. 'S 6'"
        )
    return amounts


def _read_nothing(stream: _Stream, keyword: _Token) -> None:
    return None


Reader = Any

#: Keyword spelling -> (canonical name, how to read its values).
_CRYSTAL_KEYWORDS: dict[str, tuple[str, Reader]] = {
    "TYPE": ("type", _read_word),
    "DIFFRACTIONDECAYMODEL": ("ddm", _read_word),
    "DDM": ("ddm", _read_word),
    "DECAYPARAM": ("decay_param", lambda s, k: _read_numbers(s, k, 3, 3)),
    "ABSCOEFCALC": ("coefcalc", _read_word),
    "COEFCALC": ("coefcalc", _read_word),
    "DIMENSION": ("dimensions", lambda s, k: _read_numbers(s, k, 1, 3)),
    "DIMENSIONS": ("dimensions", lambda s, k: _read_numbers(s, k, 1, 3)),
    "PIXELSPERMICRON": ("ppm", _read_number),
    "ANGLEP": ("angle_p", _read_number),
    "ANGLEL": ("angle_l", _read_number),
    "UNITCELL": ("unit_cell", lambda s, k: _read_unit_cell(s, k)),
    "NUMMONOMERS": ("num_monomers", _read_count),
    "NUMRESIDUES": ("num_residues", _read_count),
    "NUMRNA": ("num_rna", _read_count),
    "NUMDNA": ("num_dna", _read_count),
    "NUMCARB": ("num_carb", _read_count),
    "PROTEINHEAVYATOMS": ("protein_heavy_atoms", _read_elements),
    "SMALLMOLEATOMS": ("small_mole_atoms", _read_elements),
    "SOLVENTHEAVYCONC": ("solvent_heavy_conc", _read_elements),
    "SOLVENTFRACTION": ("solvent_fraction", _read_number),
    "PROTEINCONC": ("protein_conc", _read_number),
    "PROTEINCONCENTRATION": ("protein_conc", _read_number),
    "PDB": ("pdb", _read_word),
    "CIF": ("cif", _read_word),
    "SEQFILE": ("seq_file", _read_word),
    "SEQUENCEFILE": ("seq_file", _read_word),
    "MODELFILE": ("model_file", _read_word),
    # The grammar's WIREFRAMETYPE has an empty alternative where the 'y' is,
    # so the misspelling is accepted too.
    "WIREFRAMETYPE": ("wireframe_type", _read_word),
    "WIREFRAMETPE": ("wireframe_type", _read_word),
    "CALCULATEPEESCAPE": ("pe_escape", _read_word),
    "CALCULATEFLESCAPE": ("fl_escape", _read_word),
    "CALCSURROUNDING": ("surrounding", _read_word),
    "PERESOLUTION": ("pe_resolution", _read_count),
    "FLRESOLUTION": ("fl_resolution", _read_count),
    # Read only by the Monte Carlo and XFEL subprograms (Crystal.java passes
    # them to startMC and startXFEL and nowhere else), so an MX run ignores
    # them in the Java and here.
    "GONIOMETERAXIS": ("ignored", _read_number),
    "POLARISATIONDIRECTION": ("ignored", _read_number),
    "SUBPROGRAM": ("subprogram", _read_word),
}

_BEAM_KEYWORDS: dict[str, tuple[str, Reader]] = {
    "TYPE": ("type", _read_word),
    "FLUX": ("flux", _read_number),
    "FWHM": ("fwhm", lambda s, k: _read_numbers(s, k, 2, 2)),
    "ENERGY": ("energy", lambda s, k: _read_energy(s, k)),
    # The grammar's collimation rule matches "Collimation" on its own, with
    # the shape and sizes as a separate line; both orders and both spellings
    # are valid, and "Rectangular 100 100" needs no "Collimation" at all.
    "COLLIMATION": ("collimation", _read_nothing),
    "RECTANGULAR": ("rectangular", lambda s, k: _read_numbers(s, k, 2, 2)),
    "CIRCULAR": ("circular", lambda s, k: _read_numbers(s, k, 2, 2)),
    "HORIZONTAL": ("horizontal", _read_number),
    "VERTICAL": ("vertical", _read_number),
}

_WEDGE_KEYWORDS: dict[str, tuple[str, Reader]] = {
    "EXPOSURETIME": ("exposure_time", _read_number),
    "ANGULARRESOLUTION": ("angular_resolution", _read_number),
    "STARTOFFSET": ("start_offset", lambda s, k: _read_numbers(s, k, 2, 3)),
    "TRANSLATEPERDEGREE": (
        "translate_per_degree",
        lambda s, k: _read_numbers(s, k, 2, 3),
    ),
    "ROTAXBEAMOFFSET": ("rot_axis_beam_offset", _read_number),
    "MAXRESOLUTION": ("max_resolution", _read_number),
}

_NOT_PORTED = "which this port does not implement"
_SURROUNDING = f"is part of modelling the material around the crystal, {_NOT_PORTED}"

#: Keywords that would change a result this port cannot compute.
_REFUSED: dict[str, dict[str, str]] = {
    "Crystal": {
        "CONTAINERMATERIALTYPE": "sample containers are not supported yet",
        "MATERIALTYPE": "sample containers are not supported yet",
        "CONTAINERTHICKNESS": "sample containers are not supported yet",
        "CONTAINERDENSITY": "sample containers are not supported yet",
        "CONTAINERMATERIALMIXTURE": "sample containers are not supported yet",
        "MATERIALMIXTURE": "sample containers are not supported yet",
        "CONTAINERMATERIALELEMENTS": "sample containers are not supported yet",
        "MATERIALELEMENTS": "sample containers are not supported yet",
        "SURROUNDINGTHICKNESS": _SURROUNDING,
        "SURROUNDINGDENSITY": _SURROUNDING,
        "SURROUNDINGELEMENTS": _SURROUNDING,
        "SURROUNDINGHEAVYCONC": _SURROUNDING,
        "DENSITYBASED": _SURROUNDING,
        "SIMELECTRONS": f"sets up the Monte Carlo simulation, {_NOT_PORTED}",
        "SIMPHOTONS": f"sets up the Monte Carlo simulation, {_NOT_PORTED}",
        "RUNS": f"repeats the Monte Carlo or XFEL simulation, {_NOT_PORTED}",
    },
    "Beam": {
        "ENERGYFWHM": (
            f"makes a pink beam, summed over 200 sampled energies, {_NOT_PORTED} "
            f"-- ignoring it would silently compute a monochromatic dose"
        ),
        "FILE": f"belongs to an experimental beam profile, {_NOT_PORTED}",
        "PIXELSIZE": f"belongs to an experimental beam profile, {_NOT_PORTED}",
        "EXPOSURE": f"belongs to a MicroED beam, {_NOT_PORTED}",
        "PULSEENERGY": f"belongs to an XFEL beam, {_NOT_PORTED}",
        "SEMIANGLE": f"belongs to a MicroED beam, {_NOT_PORTED}",
        "APERTURERADIUS": f"belongs to a MicroED beam, {_NOT_PORTED}",
        "IMAGEDIM": f"belongs to a MicroED beam, {_NOT_PORTED}",
    },
    "Wedge": {},
}

#: AbsCoefCalc value -> composition method, following the grammar's tokens:
#: RDJAVA matches RD3 or RD3D, and the tokens *named* PDB and CIF match the
#: literals EXP and EXPSM.
_COEFCALC = {
    "DUMMY": "average",
    "AVERAGE": "average",
    "DEFAULT": "unit_cell",
    "RD3D": "unit_cell",
    "RD3": "unit_cell",
    "EXP": "pdb",
    "PDB": "pdb",
    "SAXS": "saxs",
    "SEQUENCE": "sequence",
    "SMALLMOLE": "small_molecule",
    "EXPSM": "cif",
    "CIF": "cif",
}
_COEFCALC_REFUSED = {
    "RDV": "runs the original Fortran RADDOSE as an external program",
    "RDV2": "runs the original Fortran RADDOSE as an external program",
    "RDV3": "runs the original Fortran RADDOSE as an external program",
    "SAXSSEQ": "the SAXS-from-sequence calculation is not ported",
}

_SUBPROGRAMS = {
    "MONTECARLO": "Monte Carlo",
    "GOS": "Monte Carlo",
    "XFEL": "XFEL",
    "EMSP": "MicroED",
    "EMED": "MicroED",
}

_DECAY = {"SIMPLE": "simple", "LINEAR": "linear", "LEAL": "leal", "BFACTOR": "bfactor"}

_CRYSTAL_TYPES = {"CUBOID", "SPHERICAL", "CYLINDER", "POLYHEDRON"}
_BEAM_TYPES = {"GAUSSIAN": "gaussian", "TOPHAT": "tophat"}


def _read_unit_cell(stream: _Stream, keyword: _Token) -> list[float]:
    """Three edges, optionally followed by three angles -- never four or five."""
    values = _read_numbers(stream, keyword, 3, 6)
    if len(values) not in (3, 6):
        raise stream.error(
            keyword.line,
            f"UnitCell takes three edges, or three edges and three angles; "
            f"found {len(values)} numbers",
        )
    return values


def _read_energy(stream: _Stream, keyword: _Token) -> float:
    energy = _read_number(stream, keyword)
    following = stream.peek()
    if following is not None and following.text.upper() == "KEV":
        stream.take()
    return energy


def _read_block(
    stream: _Stream, name: str, keywords: dict[str, tuple[str, Reader]]
) -> dict[str, tuple[Any, int]]:
    """Read one block's lines, up to the next block keyword or the end."""
    values: dict[str, tuple[Any, int]] = {}
    while not stream.done():
        keyword = stream.peek()
        assert keyword is not None
        spelling = keyword.text.upper()
        if spelling in _BLOCKS:
            break
        stream.take()
        if spelling in _REFUSED[name]:
            raise stream.error(
                keyword.line, f"{keyword.text} {_REFUSED[name][spelling]}"
            )
        if spelling not in keywords:
            raise stream.error(
                keyword.line, f"{keyword.text!r} is not a keyword of a {name} block"
            )
        canonical, reader = keywords[spelling]
        # A repeated keyword replaces the earlier one, as properties.put does.
        values[canonical] = (reader(stream, keyword), keyword.line)
    return values


def _require(
    stream: _Stream,
    values: dict[str, tuple[Any, int]],
    key: str,
    label: str,
    line: int,
    why: str,
) -> Any:
    if key not in values:
        raise stream.error(line, f"{label} is required: {why}")
    return values[key][0]


def _composition(
    stream: _Stream, values: dict[str, tuple[Any, int]], line: int
) -> dict[str, Any]:
    """Build the composition from AbsCoefCalc and the keywords it reads.

    Keywords the chosen calculation does not read are ignored, as the Java
    ignores them -- with one exception the Java itself now enforces.
    """
    written, at = values.get("coefcalc", ("RD3D", line))
    choice = written.upper()
    if choice in _COEFCALC_REFUSED:
        raise stream.error(at, f"AbsCoefCalc {written}: {_COEFCALC_REFUSED[choice]}")
    if choice not in _COEFCALC:
        raise stream.error(at, f"AbsCoefCalc {written!r} is not a recognised method")
    method = _COEFCALC[choice]

    def get(key: str, default: Any = None) -> Any:
        return values[key][0] if key in values else default

    def cell() -> dict[str, Any]:
        edges = _require(
            stream,
            values,
            "unit_cell",
            "UnitCell",
            line,
            f"AbsCoefCalc {written} builds the composition from the unit cell",
        )
        out: dict[str, Any] = {"unit_cell": tuple(edges[:3])}
        if len(edges) == 6:
            out["unit_cell_angles"] = tuple(edges[3:])
        return out

    def monomers() -> int:
        return int(
            _require(
                stream,
                values,
                "num_monomers",
                "NumMonomers",
                line,
                "RADDOSE-3D would default it to 0, leaving no macromolecule at all",
            )
        )

    shared = {
        "protein_heavy_atoms": get("protein_heavy_atoms", {}),
        "solvent_heavy_conc": get("solvent_heavy_conc", {}),
        "solvent_fraction": get("solvent_fraction"),
    }
    counts = {
        "num_residues": get("num_residues", 0),
        "num_rna": get("num_rna", 0),
        "num_dna": get("num_dna", 0),
        "num_carb": get("num_carb", 0),
    }

    if method == "average":
        return {"method": "average"}
    if method == "unit_cell":
        return {
            "method": method,
            **cell(),
            "num_monomers": monomers(),
            **counts,
            **shared,
        }
    if method == "saxs":
        concentration = _require(
            stream,
            values,
            "protein_conc",
            "ProteinConc",
            line,
            "a SAXS sample's monomer count comes from its concentration",
        )
        return {
            "method": method,
            **cell(),
            "protein_concentration": concentration,
            **counts,
            **shared,
        }
    if method == "sequence":
        sequence = _require(
            stream,
            values,
            "seq_file",
            "SeqFile",
            line,
            "AbsCoefCalc Sequence reads the residues from a sequence file",
        )
        return {
            "method": method,
            **cell(),
            "sequence_file": sequence,
            "num_monomers": monomers(),
            "num_carb": counts["num_carb"],
            **shared,
        }
    if method == "small_molecule":
        if "solvent_heavy_conc" in values:
            raise stream.error(
                values["solvent_heavy_conc"][1],
                "SolventHeavyConc has no effect with AbsCoefCalc SmallMole, which "
                "has no solvent model, and RADDOSE-3D refuses it",
            )
        atoms = _require(
            stream,
            values,
            "small_mole_atoms",
            "SmallMoleAtoms",
            line,
            "AbsCoefCalc SmallMole builds the composition from the atoms",
        )
        return {"method": method, **cell(), "atoms": atoms, "num_monomers": monomers()}
    if method == "pdb":
        pdb = _require(
            stream,
            values,
            "pdb",
            "PDB",
            line,
            f"AbsCoefCalc {written} reads a PDB file",
        )
        return {
            "method": method,
            "pdb": pdb,
            "solvent_heavy_conc": shared["solvent_heavy_conc"],
        }
    cif = _require(
        stream, values, "cif", "CIF", line, f"AbsCoefCalc {written} reads a CIF file"
    )
    return {
        "method": method,
        "cif": cif,
        "solvent_heavy_conc": shared["solvent_heavy_conc"],
    }


def _crystal(
    stream: _Stream, values: dict[str, tuple[Any, int]], line: int
) -> dict[str, Any]:
    if "subprogram" in values:
        written, at = values["subprogram"]
        engine = _SUBPROGRAMS.get(written.upper())
        if engine is not None:
            raise stream.error(
                at,
                f"Subprogram {written} runs RADDOSE-3D's {engine} engine, "
                f"{_NOT_PORTED}",
            )
        if written.upper() != "RD3D":
            raise stream.error(at, f"Subprogram {written!r} is not one RADDOSE-3D has")

    written = _require(
        stream, values, "type", "Type", line, "RADDOSE-3D needs the crystal's shape"
    )
    shape = written.upper()
    if shape not in _CRYSTAL_TYPES:
        raise stream.error(
            values["type"][1],
            f"Type {written!r} is not a crystal shape; expected Cuboid, Spherical, "
            f"Cylinder or Polyhedron",
        )

    crystal: dict[str, Any] = {"type": shape.lower()}
    if shape == "POLYHEDRON":
        crystal["model_file"] = _require(
            stream,
            values,
            "model_file",
            "ModelFile",
            line,
            "a polyhedron is read from a file",
        )
        if "wireframe_type" in values:
            crystal["wireframe_type"] = values["wireframe_type"][0].lower()
    else:
        dims = _require(
            stream,
            values,
            "dimensions",
            "Dimensions",
            line,
            f"RADDOSE-3D needs the size of a {written.lower()} crystal",
        )
        needed = {"CUBOID": 3, "SPHERICAL": 1, "CYLINDER": 2}[shape]
        if len(dims) < needed:
            raise stream.error(
                values["dimensions"][1],
                f"a {written.lower()} crystal needs {needed} dimensions, "
                f"found {len(dims)}",
            )
        if shape == "CUBOID":
            crystal["dimensions"] = tuple(dims)
        elif shape == "SPHERICAL":
            crystal["diameter"] = dims[0]
        else:
            # The third value is ignored, with a warning in the Java.
            crystal["diameter"], crystal["height"] = dims[0], dims[1]

    for key, field_name in (
        ("ppm", "pixels_per_micron"),
        ("angle_p", "angle_p"),
        ("angle_l", "angle_l"),
    ):
        if key in values:
            crystal[field_name] = values[key][0]

    if "ddm" in values:
        written_ddm, at = values["ddm"]
        if written_ddm.upper() not in _DECAY:
            raise stream.error(
                at,
                f"DiffractionDecayModel {written_ddm!r} is not one of Simple, "
                f"Linear, Leal, BFactor",
            )
        crystal["decay_model"] = _DECAY[written_ddm.upper()]
    if "decay_param" in values:
        crystal["decay_parameters"] = tuple(values["decay_param"][0])

    escape: dict[str, Any] = {}
    for key, field_name in (
        ("pe_escape", "photoelectron"),
        ("fl_escape", "fluorescence"),
        ("surrounding", "surrounding"),
    ):
        if key in values:
            # "TRUE".equals(value.toUpperCase()), as Crystal.java tests it.
            escape[field_name] = values[key][0].upper() == "TRUE"
    for key in ("pe_resolution", "fl_resolution"):
        if key in values:
            escape[key] = values[key][0]
    if escape:
        crystal["escape"] = escape

    crystal["composition"] = _composition(stream, values, line)
    return crystal


def _beam(
    stream: _Stream, values: dict[str, tuple[Any, int]], line: int
) -> dict[str, Any]:
    written = _require(
        stream, values, "type", "Type", line, "RADDOSE-3D needs the beam profile"
    )
    profile = written.upper()
    if profile in ("EXPERIMENTAL", "EXPERIMENTALPGM"):
        raise stream.error(
            values["type"][1],
            f"Type {written}: experimental beam profiles are not ported",
        )
    if profile not in _BEAM_TYPES:
        raise stream.error(
            values["type"][1],
            f"Type {written!r} is not a beam profile; expected Gaussian or TopHat",
        )

    beam: dict[str, Any] = {
        "type": _BEAM_TYPES[profile],
        "flux": _require(
            stream, values, "flux", "Flux", line, "RADDOSE-3D's beams require it"
        ),
        "energy": _require(
            stream, values, "energy", "Energy", line, "RADDOSE-3D's beams require it"
        ),
    }
    if profile == "GAUSSIAN":
        beam["fwhm"] = tuple(
            _require(
                stream, values, "fwhm", "FWHM", line, "a Gaussian beam needs its widths"
            )
        )

    # Merged as the Java merges properties: Rectangular and Circular set both
    # sides, Horizontal and Vertical one each, and Circular also marks the
    # aperture elliptical.
    horizontal = vertical = None
    circular = False
    for key in ("rectangular", "circular", "horizontal", "vertical"):
        if key not in values:
            continue
        value = values[key][0]
        if key in ("rectangular", "circular"):
            horizontal, vertical = value
            circular = circular or key == "circular"
        elif key == "horizontal":
            horizontal = value
        else:
            vertical = value
    if (horizontal is None) != (vertical is None):
        raise stream.error(
            line,
            "the collimation has only one side; give both Horizontal and Vertical, "
            "or Rectangular or Circular with two sizes",
        )
    if horizontal is not None:
        beam["collimation"] = {
            "shape": "circular" if circular else "rectangular",
            "x": horizontal,
            "y": vertical,
        }
    return beam


def _wedge(
    stream: _Stream, values: dict[str, tuple[Any, int]], angles: list[float], line: int
) -> dict[str, Any]:
    wedge: dict[str, Any] = {
        "start": angles[0],
        "end": angles[1],
        "exposure_time": _require(
            stream,
            values,
            "exposure_time",
            "ExposureTime",
            line,
            "RADDOSE-3D requires it for every wedge",
        ),
    }
    for key in ("angular_resolution", "rot_axis_beam_offset", "max_resolution"):
        if key in values:
            wedge[key] = values[key][0]
    for key in ("start_offset", "translate_per_degree"):
        if key in values:
            # A missing z is left unset in the Java, which means zero.
            components = values[key][0]
            wedge[key] = tuple(components) + (0.0,) * (3 - len(components))
    return wedge


def parse_raddose_txt(text: str, source: str = "<input>") -> dict[str, Any]:
    """Read a RADDOSE-3D ``.txt`` input into the fields of an experiment.

    Parameters
    ----------
    text : str
        The file's contents.
    source : str, optional
        Name to use in error messages, usually the path.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for :class:`~pyraddose_3d.models.experiment.Experiment`.

    Raises
    ------
    RaddoseTxtError
        If the file does not follow the grammar, asks for something this port
        does not implement, or leaves out something RADDOSE-3D requires.

    Notes
    -----
    RADDOSE-3D reads blocks in order: each ``Crystal`` and ``Beam`` replaces
    the current one, and each ``Wedge`` is exposed at once with whatever is
    current. So a file may change the beam between wedges. An experiment
    here has one crystal and one beam, so that is refused rather than
    silently flattened. Every example input that ships with RADDOSE-3D is a
    crystal, a beam and then wedges.
    """
    stream = _Stream(_tokenise(text, source), source)
    crystal: dict[str, Any] | None = None
    beam: dict[str, Any] | None = None
    wedges: list[dict[str, Any]] = []

    while not stream.done():
        token = stream.take()
        block = token.text.upper()
        if block not in _BLOCKS:
            raise stream.error(
                token.line, f"expected Crystal, Beam or Wedge, found {token.text!r}"
            )
        if block in ("CRYSTAL", "BEAM"):
            name = block.capitalize()
            if wedges:
                raise stream.error(
                    token.line,
                    f"this {name} block comes after a Wedge, so RADDOSE-3D would "
                    f"change the {name.lower()} part way through the experiment; "
                    f"pyRADDOSE-3D runs one crystal and one beam per experiment",
                )
            if (crystal if block == "CRYSTAL" else beam) is not None:
                raise stream.error(
                    token.line, f"a second {name} block before any Wedge"
                )
            keywords = _CRYSTAL_KEYWORDS if block == "CRYSTAL" else _BEAM_KEYWORDS
            values = _read_block(stream, name, keywords)
            if block == "CRYSTAL":
                crystal = _crystal(stream, values, token.line)
            else:
                beam = _beam(stream, values, token.line)
            continue

        if crystal is None or beam is None:
            raise stream.error(
                token.line, "a Wedge needs a Crystal and a Beam before it"
            )
        angles = _read_numbers(stream, token, 2, 2)
        values = _read_block(stream, "Wedge", _WEDGE_KEYWORDS)
        wedges.append(_wedge(stream, values, angles, token.line))

    if crystal is None or beam is None or not wedges:
        missing = [
            n
            for n, v in (("Crystal", crystal), ("Beam", beam), ("Wedge", wedges))
            if not v
        ]
        raise RaddoseTxtError(f"{source}: no {', '.join(missing)} block")
    return {"crystal": crystal, "beam": beam, "wedges": wedges}
