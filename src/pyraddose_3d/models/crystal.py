"""Crystal geometry and the physics options attached to it.

A field has a default only where RADDOSE-3D has one, and it is RADDOSE-3D's,
so that leaving a setting out of a YAML file means the same as leaving it out
of a ``.txt`` input. The shape's size and the composition are required by the
Java and are required here.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from pyraddose_3d.models._base import RD3DModel
from pyraddose_3d.models.composition import Composition

__all__ = [
    "Crystal",
    "CuboidCrystal",
    "CylinderCrystal",
    "EscapeOptions",
    "PolyhedronCrystal",
    "SphericalCrystal",
]


class EscapeOptions(RD3DModel):
    """Photoelectron and fluorescence escape modelling.

    These are Monte Carlo calculations. They are not differentiable, and a
    vectorised implementation cannot reproduce the reference implementation's
    random draw order, so conformance cases that enable them are compared
    statistically rather than value by value.

    Attributes
    ----------
    photoelectron : bool
        Track photoelectrons escaping the crystal.
    fluorescence : bool
        Track fluorescent photons escaping the crystal.
    surrounding : bool
        Model a surrounding medium that escaped particles can enter.
    pe_resolution, fl_resolution : int | None
        Angular/distance resolution of the respective escape calculations.
    """

    photoelectron: bool = False
    fluorescence: bool = False
    surrounding: bool = False
    pe_resolution: Annotated[int, Field(gt=0)] | None = None
    fl_resolution: Annotated[int, Field(gt=1)] | None = None

    @property
    def any_enabled(self) -> bool:
        """Whether any escape modelling is switched on."""
        return self.photoelectron or self.fluorescence or self.surrounding


class _CrystalBase(RD3DModel):
    """Fields common to every crystal shape."""

    pixels_per_micron: Annotated[float, Field(gt=0.0)] | None = None
    angle_p: float = 0.0
    angle_l: float = 0.0
    composition: Composition
    decay_model: Literal["simple", "linear", "leal", "bfactor"] = "simple"
    decay_parameters: tuple[float, float, float] | None = None
    escape: EscapeOptions = Field(default_factory=EscapeOptions)

    def _first_dimension(self) -> float | None:
        """The size RADDOSE-3D bases its default resolution on."""
        return None

    @property
    def effective_pixels_per_micron(self) -> float:
        """The resolution the calculation uses: as given, or RADDOSE-3D's default.

        ``Crystal.java`` sets ``CRYSTAL_RESOLUTION_DEF = 10 / CRYSTAL_DIM_X``
        when no ``PixelsPerMicron`` is given -- ten voxels across the first
        dimension. A shape with no dimensions keeps the Java's static fallback
        of 0.5.

        The field itself stays ``None`` when nothing was given, so that a
        configuration records what its author wrote and the ``.txt`` handed to
        the Java leaves the line out -- which is what lets the conformance
        corpus check the Java's default rather than restate it.

        Returns
        -------
        float
            Voxels per micrometre.
        """
        if self.pixels_per_micron is not None:
            return self.pixels_per_micron
        first = self._first_dimension()
        return 0.5 if first is None else 10.0 / first


class CuboidCrystal(_CrystalBase):
    """A rectangular block.

    Attributes
    ----------
    dimensions : tuple[float, float, float]
        Extent in x, y and z, in micrometres.
    """

    type: Literal["cuboid"] = "cuboid"
    dimensions: tuple[float, float, float]

    def _first_dimension(self) -> float | None:
        return self.dimensions[0]


class SphericalCrystal(_CrystalBase):
    """A sphere, meshed as an icosphere.

    Attributes
    ----------
    diameter : float
        Sphere diameter, in micrometres.
    """

    type: Literal["spherical"] = "spherical"
    diameter: Annotated[float, Field(gt=0.0)]

    def _first_dimension(self) -> float | None:
        return self.diameter


class CylinderCrystal(_CrystalBase):
    """A cylinder, meshed as a 32-segment prism.

    Note the axis lies along *y*, not z. The reference implementation
    ignores the third ``Dimensions`` value, with a warning, which is why this
    model exposes ``diameter`` and ``height`` rather than a dimension
    triple.

    Attributes
    ----------
    diameter : float
        Diameter of the circular cross-section, in micrometres.
    height : float
        Axial length, in micrometres.
    """

    type: Literal["cylinder"] = "cylinder"
    diameter: Annotated[float, Field(gt=0.0)]
    height: Annotated[float, Field(gt=0.0)]

    def _first_dimension(self) -> float | None:
        return self.diameter


class PolyhedronCrystal(_CrystalBase):
    """An arbitrary mesh loaded from a wireframe file.

    Attributes
    ----------
    model_file : str
        Path to the wireframe file, resolved relative to the run's working
        directory.
    wireframe_type : str
        Wireframe format. Only ``obj`` is supported by the reference
        implementation.
    """

    type: Literal["polyhedron"] = "polyhedron"
    model_file: str
    wireframe_type: Literal["obj"] = "obj"


Crystal = Annotated[
    CuboidCrystal | SphericalCrystal | CylinderCrystal | PolyhedronCrystal,
    Field(discriminator="type"),
]
