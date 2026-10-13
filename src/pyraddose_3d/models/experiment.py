"""A complete experiment: one crystal, one beam, and one or more wedges."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import Field, model_validator

from pyraddose_3d.models._base import RD3DModel
from pyraddose_3d.models.beam import Beam, ElectronBeam
from pyraddose_3d.models.crystal import Crystal
from pyraddose_3d.models.wedge import Wedge

if TYPE_CHECKING:
    import os

__all__ = ["Experiment"]


class Experiment(RD3DModel):
    """Everything needed to compute a dose distribution.

    Attributes
    ----------
    crystal : Crystal
        The sample, its composition and its physics options.
    beam : Beam
        The incident beam.
    wedges : list[Wedge]
        One or more angular sweeps, applied in order.
    seed : int | None
        Seed for the Monte Carlo escape calculations. Required for a
        reproducible run whenever escape modelling is enabled; ignored
        otherwise.
    """

    crystal: Crystal
    beam: Beam
    wedges: Annotated[list[Wedge], Field(min_length=1)]
    seed: int | None = None

    @model_validator(mode="after")
    def _beam_suits_the_engine(self) -> Self:
        """MicroED takes an electron beam, and only MicroED does."""
        microed = self.crystal.subprogram == "microed"
        electron = isinstance(self.beam, ElectronBeam)
        if microed and not electron:
            raise ValueError(
                "MicroED needs an electron beam (type: electron, with an exposure "
                "in electrons per square Angstrom). RADDOSE-3D's MicroED takes its "
                "exposure from a top-hat beam only: a Gaussian beam gives it none, "
                "and an experimental beam crashes it"
            )
        if electron and not microed:
            raise ValueError(
                "an electron beam is for the MicroED engine; set the crystal's "
                "subprogram to microed"
            )
        return self

    @classmethod
    def from_raddose_txt(cls, path: str | os.PathLike[str]) -> Self:
        """Load a RADDOSE-3D ``.txt`` input file.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Path to the input file.

        Returns
        -------
        Self
            The validated experiment -- the same object the equivalent YAML
            would have produced.

        Raises
        ------
        RaddoseTxtError
            If the file does not parse, asks for something this port does
            not support, or leaves out something RADDOSE-3D requires. The
            message names the line.
        """
        from pyraddose_3d._raddose_txt import parse_raddose_txt

        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return cls(**parse_raddose_txt(text, source=str(path)))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Load either input format, chosen by the file's extension.

        ``.yaml`` and ``.yml`` are read as YAML; anything else as a
        RADDOSE-3D ``.txt`` input, since those have never had a fixed
        extension.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Path to the input file.

        Returns
        -------
        Self
            The validated experiment.
        """
        if Path(path).suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(path)
        return cls.from_raddose_txt(path)

    def to_raddose_txt(self) -> str:
        """Render this experiment as a legacy RADDOSE-3D input file.

        Used to drive the Java reference implementation for conformance
        testing, and as a migration path for existing users.

        Returns
        -------
        str
            The contents of an equivalent ``.txt`` input file.
        """
        from pyraddose_3d._raddose_txt import emit_raddose_txt

        return emit_raddose_txt(self)
