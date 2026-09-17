"""Rotation wedge: how the crystal is exposed over an angular range."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from pyraddose_3d.models._base import RD3DModel

__all__ = ["Wedge"]


class Wedge(RD3DModel):
    """A single angular sweep of the crystal through the beam.

    Attributes
    ----------
    start : float
        Start of the rotation, in degrees.
    end : float
        End of the rotation, in degrees.
    exposure_time : float
        Total exposure over the whole angular range, in seconds.
    angular_resolution : float | None
        Angular step of the simulation, in degrees. ``None`` leaves the
        RADDOSE-3D default in place.
    start_offset : tuple[float, float, float] | None
        Crystal position at the start of the wedge, in micrometres.
    translate_per_degree : tuple[float, float, float] | None
        Crystal translation per degree of rotation, in micrometres.
    rot_axis_beam_offset : float | None
        Offset of the rotation axis from the beam axis, in micrometres.
    """

    start: float = 0.0
    end: float = 90.0
    exposure_time: Annotated[float, Field(gt=0.0)] = 50.0
    angular_resolution: Annotated[float, Field(gt=0.0)] | None = None
    start_offset: tuple[float, float, float] | None = None
    translate_per_degree: tuple[float, float, float] | None = None
    rot_axis_beam_offset: float | None = None
