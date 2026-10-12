"""Rotation wedge: how the crystal is exposed over an angular range."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from pyraddose_3d.models._base import RD3DModel

__all__ = ["Wedge"]


class Wedge(RD3DModel):
    """A single angular sweep of the crystal through the beam.

    A field has a default only where RADDOSE-3D has one, and it is
    RADDOSE-3D's, so that leaving a setting out of a YAML file means the same
    as leaving it out of a ``.txt`` input. Start, end and exposure time are
    required by the Java and are required here.

    Attributes
    ----------
    start : float
        Start of the rotation, in degrees.
    end : float
        End of the rotation, in degrees.
    exposure_time : float
        Total exposure over the whole angular range, in seconds.
    angular_resolution : float | None
        Angular step of the simulation, in degrees. ``None`` means 2.
        RADDOSE-3D also replaces a given step with 2 unless the wedge spans
        more than ten of them -- see ``solve.wedge_resolution``.
    start_offset : tuple[float, float, float] | None
        Crystal position at the start of the wedge, in micrometres.
    translate_per_degree : tuple[float, float, float] | None
        Crystal translation per degree of rotation, in micrometres.
    rot_axis_beam_offset : float | None
        Offset of the rotation axis from the beam axis, in micrometres.
    max_resolution : float
        Highest resolution of the data, in angstroms. Sets the resolution of
        the last relative diffraction efficiency column RADDOSE-3D reports;
        it has no effect on the dose. Defaults to 2, as in ``Wedge.java``.
    """

    start: float
    end: float
    exposure_time: Annotated[float, Field(gt=0.0)]
    angular_resolution: Annotated[float, Field(gt=0.0)] | None = None
    start_offset: tuple[float, float, float] | None = None
    translate_per_degree: tuple[float, float, float] | None = None
    rot_axis_beam_offset: float | None = None
    max_resolution: Annotated[float, Field(gt=0.0)] = 2.0
