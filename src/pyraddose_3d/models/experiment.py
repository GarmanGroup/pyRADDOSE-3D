"""A complete experiment: one crystal, one beam, and one or more wedges."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from pyraddose_3d.models._base import RD3DModel
from pyraddose_3d.models.beam import Beam, GaussianBeam
from pyraddose_3d.models.crystal import Crystal, CuboidCrystal
from pyraddose_3d.models.wedge import Wedge

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

    crystal: Crystal = Field(default_factory=CuboidCrystal)
    beam: Beam = Field(default_factory=GaussianBeam)
    wedges: Annotated[list[Wedge], Field(min_length=1)] = Field(
        default_factory=lambda: [Wedge()]
    )
    seed: int | None = None

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
