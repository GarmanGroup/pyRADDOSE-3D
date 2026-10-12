"""Crystal geometry and the physics options attached to it.

A field has a default only where RADDOSE-3D has one, and it is RADDOSE-3D's,
so that leaving a setting out of a YAML file means the same as leaving it out
of a ``.txt`` input. The shape's size and the composition are required by the
Java and are required here.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from pyraddose_3d.models._base import RD3DModel
from pyraddose_3d.models.composition import Composition

__all__ = [
    "Container",
    "Crystal",
    "CuboidCrystal",
    "CylinderCrystal",
    "ElementalContainer",
    "EscapeOptions",
    "MixtureContainer",
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


class _ContainerBase(RD3DModel):
    """Fields every container has.

    Attributes
    ----------
    thickness : float
        Thickness the beam crosses, in micrometres.
    density : float
        Density in g/cm^3.
    """

    thickness: Annotated[float, Field(gt=0.0)]
    density: Annotated[float, Field(gt=0.0)]


class MixtureContainer(_ContainerBase):
    """A container of a compound or mixture NIST tabulates by name.

    ``ContainerMaterialType Mixture``. The mass attenuation coefficient is
    interpolated from NIST's table for the material, as
    ``ContainerMixture`` does.

    Attributes
    ----------
    material : str
        A NIST compound name: water, mylar, pmma, polyethylene, pyrex and
        44 others. Kapton is not among them; use an elemental container.
    """

    type: Literal["mixture"] = "mixture"
    material: str

    @field_validator("material")
    @classmethod
    def _tabulated(cls, material: str) -> str:
        from pyraddose_3d.backend.container import known_mixtures

        name = material.lower()
        if name not in known_mixtures():
            raise ValueError(
                f"NIST does not tabulate {material!r}; it has "
                f"{', '.join(sorted(known_mixtures()))}"
            )
        return name


class ElementalContainer(_ContainerBase):
    """A container given by its elemental formula.

    ``ContainerMaterialType Elemental``. Each element's coefficient comes
    from its own NIST table, combined by mass fraction, as
    ``ContainerElemental`` does.

    Attributes
    ----------
    elements : dict[str, float]
        Element symbol to number of atoms, e.g. ``{"C": 22, "H": 10,
        "N": 2, "O": 5}`` for Kapton.
    """

    type: Literal["elemental"] = "elemental"
    elements: dict[str, float]


Container = Annotated[
    MixtureContainer | ElementalContainer, Field(discriminator="type")
]


class _CrystalBase(RD3DModel):
    """Fields common to every crystal shape.

    Attributes
    ----------
    container : Container | None
        What the crystal is mounted in, which the beam crosses first.
        ``None`` is RADDOSE-3D's ``ContainerMaterialType None``: no
        attenuation.
    """

    pixels_per_micron: Annotated[float, Field(gt=0.0)] | None = None
    angle_p: float = 0.0
    angle_l: float = 0.0
    composition: Composition
    decay_model: Literal["simple", "linear", "leal", "bfactor"] = "simple"
    decay_parameters: tuple[float, float, float] | None = None
    escape: EscapeOptions = Field(default_factory=EscapeOptions)
    container: Container | None = None

    def _first_dimension(self) -> float | None:
        """The size RADDOSE-3D bases its default resolution on."""
        return None

    @property
    def base_pixels_per_micron(self) -> float:
        """The resolution asked for, or RADDOSE-3D's default before its cap.

        ``Crystal.java`` sets ``CRYSTAL_RESOLUTION_DEF = 10 / CRYSTAL_DIM_X``
        when no ``PixelsPerMicron`` is given -- ten voxels across the first
        dimension. A shape with no dimensions keeps the Java's static fallback
        of 0.5. The Java then caps a default at a million voxels over the
        crystal's bounding box, which needs the mesh, so that step is
        :func:`pyraddose_3d.solve.pixels_per_micron`'s; that is the
        resolution a calculation uses.

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


#: One to three numbers, as RADDOSE-3D accepts for a polyhedron.
_Placeholders = Annotated[tuple[float, ...], Field(min_length=1, max_length=3)]


class PolyhedronCrystal(_CrystalBase):
    """An arbitrary mesh loaded from a wireframe file.

    Attributes
    ----------
    model_file : str
        Path to a triangulated Wavefront ``.obj`` file, in micrometres,
        resolved relative to the run's working directory.
    wireframe_type : str
        Wireframe format. Only ``obj`` is supported by the reference
        implementation.
    dimensions : tuple[float, ...] | None
        Optional, and not the crystal's size -- the mesh is. RADDOSE-3D
        accepts them as placeholders, and still bases its default resolution
        on the first: ten voxels across it.
    """

    type: Literal["polyhedron"] = "polyhedron"
    model_file: str
    wireframe_type: Literal["obj"] = "obj"
    dimensions: (
        Annotated[tuple[float, ...], Field(min_length=1, max_length=3)] | None
    ) = None

    def _first_dimension(self) -> float | None:
        return None if self.dimensions is None else self.dimensions[0]


Crystal = Annotated[
    CuboidCrystal | SphericalCrystal | CylinderCrystal | PolyhedronCrystal,
    Field(discriminator="type"),
]
