"""Running an :class:`~pyraddose_3d.models.experiment.Experiment`.

This is the seam between the configuration layer and the numerical core.
Everything below :mod:`pyraddose_3d.backend` takes arrays and primitives and
has never heard of pydantic; everything here knows about both and translates
one into the other.

Keeping the seam in one module is what lets the backend be called directly
with tensors -- which is what differentiability and benchmarking need -- and
what stops configuration concerns leaking into the physics.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend import beam as backend_beam
from pyraddose_3d.backend._array import array_namespace, is_torch, to_namespace
from pyraddose_3d.backend.beam import pink_beam_energies
from pyraddose_3d.backend.coefficients import (
    AVERAGE_COEFFICIENTS,
    AVERAGE_DENSITY,
    MaterialCoefficients,
    density,
    material_coefficients,
    unit_cell_volume,
)
from pyraddose_3d.backend.composition import (
    Macromolecule,
    read_sequence_file,
    saxs_monomer_count,
    saxs_sequence_monomer_count,
    sequence_composition,
    small_molecule_composition,
    unit_cell_composition,
)
from pyraddose_3d.backend.container import (
    container_transmission,
    elemental_mass_attenuation,
    mixture_mass_attenuation,
)
from pyraddose_3d.backend.decay import leal_decay
from pyraddose_3d.backend.escape import (
    Escape,
    escape_factors,
    fluorescence_escape,
    photoelectron_escape,
    surrounding_composition,
    surrounding_escape,
)
from pyraddose_3d.backend.exposure import ExposureResult, expose, exposure_angles
from pyraddose_3d.backend.geometry import Mesh
from pyraddose_3d.backend.java_random import JavaRandom
from pyraddose_3d.backend.microed import MicroEDResult, electron_material
from pyraddose_3d.backend.microed import microed as run_microed
from pyraddose_3d.backend.structure_files import (
    composition_from_cif,
    composition_from_pdb,
)
from pyraddose_3d.backend.summary import WedgeSummary, summarise
from pyraddose_3d.backend.voxels import VoxelGrid, occupancy, voxel_grid
from pyraddose_3d.backend.wireframe import read_obj
from pyraddose_3d.models.crystal import SIMULATED

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyraddose_3d.backend.montecarlo.simple import SimpleRun, SimpleSetup
    from pyraddose_3d.models.experiment import Experiment

__all__ = [
    "FIXED",
    "TUNABLE",
    "Simulation",
    "apply_overrides",
    "crystal_mesh",
    "microed",
    "monte_carlo",
    "monte_carlo_setup",
    "pixels_per_micron",
    "simulate",
    "solve",
    "wedge_resolution",
]

#: Configuration paths whose value may be handed to :func:`solve` as a
#: tensor, and which then carry a usable gradient. ``*`` stands for a wedge
#: index, so ``wedges.0.exposure_time`` matches ``wedges.*.exposure_time``.
#:
#: Every one of these reaches the dose through arithmetic that is smooth
#: between steps.
#:
#: ``wedges.*.start`` earns its place by a whisker and is worth
#: understanding. ``exposure_angles`` returns
#: ``begin + sign * step * resolution``, so shifting the start shifts every
#: angle in the wedge, and that part of the gradient is real: at a start of
#: 1 degree it agrees with a central finite difference to 5e-9. But the
#: start also enters the integer image count, and *that* term is a step no
#: gradient can carry. At a start of 0 degrees the count falls from 46 to 45
#: once the angle moves by more than about 1e-11 radians -- whole numbers of
#: steps are given a 1e-9 margin -- so a finite difference there measures the
#: step and disagrees with autograd entirely. Autograd is the one telling
#: the truth -- it reports the slope at fixed image count -- but an
#: optimiser stepping across one of those boundaries will see the objective
#: jump. Shifting the *end* has only the step term, which is why it is in
#: :data:`FIXED` instead.
TUNABLE: frozenset[str] = frozenset(
    {
        "beam.flux",
        "beam.energy",
        "beam.fwhm",
        "beam.collimation.x",
        "beam.collimation.y",
        "wedges.*.start",
        "wedges.*.exposure_time",
        "wedges.*.rot_axis_beam_offset",
        "wedges.*.start_offset",
        "wedges.*.translate_per_degree",
    }
)

#: Paths that look tunable, are not, and would otherwise fail silently --
#: autograd returns a number for these, and the number is wrong. Rejected
#: with the reason rather than left as a footnote.
_GEOMETRY_IS_COUNTED = (
    "crystal geometry decides which voxels are inside the crystal, and that "
    "test is a boolean parity count. Autograd sees only the attenuation path "
    "through the mesh and is blind to voxels entering or leaving as the "
    "crystal changes size, so it returns a confident number that is missing "
    "a term -- measured 20% wrong against a finite difference for a cuboid's "
    "thickness. Optimise the beam and the wedge instead; the crystal is "
    "whatever came out of the drop"
)

FIXED: dict[str, str] = {
    "crystal.dimensions": _GEOMETRY_IS_COUNTED,
    "crystal.diameter": _GEOMETRY_IS_COUNTED,
    "crystal.height": _GEOMETRY_IS_COUNTED,
    "crystal.angle_p": _GEOMETRY_IS_COUNTED,
    "crystal.angle_l": _GEOMETRY_IS_COUNTED,
    "crystal.pixels_per_micron": (
        "the voxel grid shape is an integer count, so this is a step "
        "function with a zero derivative everywhere it is defined"
    ),
    "wedges.*.end": (
        "the end of a wedge sets how many images are taken and nothing "
        "else. exposure_angles returns begin + step * resolution, so the "
        "angles depend on the start and the step, never on the end; the end "
        "appears only inside an integer count. Ending at 88 or 89 degrees "
        "both give 45 images and identical doses, and 90 gives 46. The "
        "derivative is therefore zero wherever it exists. Optimise the "
        "exposure time, which is what actually trades dose against "
        "completeness at fixed rotation"
    ),
    "wedges.*.angular_resolution": (
        "the angular step sets the number of images, an integer count, so "
        "this is a step function like wedges.*.end"
    ),
}

#: ``math.radians`` and ``math.degrees`` spelled as multiplications, so they
#: apply to a tensor as well as a float. CPython computes them exactly this
#: way, so the float results are bit-identical.
_DEG_TO_RAD = math.pi / 180.0
_RAD_TO_DEG = 180.0 / math.pi

#: Vertices per circle of a cylinder, and the angular step between them. The
#: step is negative so the circle is traced anticlockwise, matching the
#: Blender export the original mesh came from.
_CYLINDER_VERTICES_PER_CIRCLE = 32

#: RADDOSE-3D's default angular step, in degrees.
DEFAULT_ANGULAR_RESOLUTION = 2.0

#: Decay constant of the linear diffraction decay model, in MGy. Owen et al.
#: (2006) put the half dose at 43 MGy; the model uses twice that as the dose
#: at which diffraction reaches zero.
_OWEN_2006_DOSE_LIMIT = 2 * 43.0


def _load_table(name: str) -> dict[str, list]:
    """Read a vendored mesh table."""
    text = resources.files("pyraddose_3d.data").joinpath(name).read_text("ascii")
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if fields[0] == "VERT":
            vertices.append([float(v) for v in fields[3:6]])
        elif fields[0] == "TRI":
            triangles.append([int(v) - 1 for v in fields[3:6]])
    return {"vertices": vertices, "triangles": triangles}


def _sphere_mesh(diameter: float) -> Mesh:
    """A 42-vertex icosphere, as ``CrystalSphericalNew`` builds one.

    The table is unit vectors halved, so scaling by the diameter gives a
    radius of half it.
    """
    table = _load_table("sphere_mesh.tsv")
    return Mesh(np.array(table["vertices"]) * diameter, np.array(table["triangles"]))


def _cylinder_mesh(diameter: float, height: float) -> Mesh:
    """A 64-vertex prism, as ``CrystalCylinder`` builds one.

    Two rings of 32, interleaved base and top, built about the x axis and
    then turned a quarter turn about z so the axis lies along y. The
    topology is vendored; the positions are parametric.
    """
    radius = diameter / 2
    step = -2 * math.pi / _CYLINDER_VERTICES_PER_CIRCLE

    vertices = []
    for index in range(_CYLINDER_VERTICES_PER_CIRCLE):
        angle = index * step
        # After the quarter turn about z, the circle lies in x-z and the
        # axis runs along y.
        across = -radius * math.cos(angle)
        along = radius * math.sin(angle)
        vertices.append([across, -height / 2, along])
        vertices.append([across, height / 2, along])

    table = _load_table("cylinder_topology.tsv")
    return Mesh(np.array(vertices), np.array(table["triangles"]))


def _apply_crystal_angles(vertices: Any, angle_p: float, angle_l: float) -> Any:
    """Turn a mesh by ``AngleP`` about z, then ``AngleL`` about x.

    Both are in degrees, matching the input file.
    """
    turn_p = math.radians(angle_p)
    turn_l = math.radians(angle_l)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    turned_x = x * math.cos(turn_p) + y * math.sin(turn_p)
    turned_y = -x * math.sin(turn_p) + y * math.cos(turn_p)
    return np.stack(
        [
            turned_x,
            turned_y * math.cos(turn_l) + z * math.sin(turn_l),
            -turned_y * math.sin(turn_l) + z * math.cos(turn_l),
        ],
        axis=-1,
    )


def crystal_mesh(crystal: Any) -> Mesh:
    """Build the triangle mesh for a crystal, with its angles applied.

    Parameters
    ----------
    crystal : Any
        One of the crystal models.

    Returns
    -------
    Mesh
        The mesh, in the lab frame.

    Raises
    ------
    pyraddose_3d.backend.wireframe.ObjError
        For a polyhedron whose file is not a triangle mesh.
    """
    if crystal.type == "cuboid":
        half = np.array(crystal.dimensions) / 2
        signs = np.array(
            [
                [-1, -1, 1],
                [-1, -1, -1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, 1],
                [1, -1, -1],
                [1, 1, -1],
                [1, 1, 1],
            ]
        )
        vertices = signs * half
        triangles = (
            np.array(
                [
                    [1, 3, 2],
                    [4, 3, 1],
                    [3, 6, 2],
                    [7, 6, 3],
                    [2, 5, 1],
                    [2, 6, 5],
                    [4, 8, 3],
                    [8, 7, 3],
                    [4, 1, 8],
                    [1, 5, 8],
                    [8, 5, 7],
                    [7, 5, 6],
                ]
            )
            - 1
        )
        mesh = Mesh(vertices, triangles)
    elif crystal.type == "spherical":
        mesh = _sphere_mesh(crystal.diameter)
    elif crystal.type == "cylinder":
        mesh = _cylinder_mesh(crystal.diameter, crystal.height)
    else:
        mesh = read_obj(crystal.model_file)

    return Mesh(
        _apply_crystal_angles(mesh.vertices, crystal.angle_p, crystal.angle_l),
        mesh.triangles,
    )


#: ``Crystal.CRYSTAL_RESOLUTION_DEF_VOXLIM``: a default resolution is lowered
#: until the crystal's bounding box holds no more voxels than this.
_DEFAULT_VOXEL_LIMIT = 1_000_000


def pixels_per_micron(crystal: Any, mesh: Mesh) -> float:
    """The resolution a calculation uses, in voxels per micrometre.

    As given, if it was. Otherwise ``Crystal.getDefaultLimitedResolution``:
    ten voxels across the first dimension, unless that would put more than a
    million voxels in the bounding box of the crystal -- after its P and L
    rotations -- in which case the resolution that puts exactly a million
    there. Only a thin plate or a needle reaches the limit: a 10 x 500 x
    500 um plate would otherwise need 2.5 million.

    Parameters
    ----------
    crystal : Any
        One of the crystal models.
    mesh : Mesh
        Its mesh, from :func:`crystal_mesh`.

    Returns
    -------
    float
        Voxels per micrometre.
    """
    if crystal.pixels_per_micron is not None:
        return float(crystal.pixels_per_micron)
    base = float(crystal.base_pixels_per_micron)
    vertices = np.asarray(to_namespace(np, mesh.vertices))
    x, y, z = (float(v) for v in vertices.max(axis=0) - vertices.min(axis=0))
    if (x * base) * (y * base) * (z * base) <= _DEFAULT_VOXEL_LIMIT:
        return base
    return math.pow(_DEFAULT_VOXEL_LIMIT / (x * y * z), 1 / 3)


def _cell_volume(composition: Any) -> Any:
    """Unit cell volume from a composition that names a cell.

    Returns the backend's ``Array``, not a ``float``: the cell dimensions
    may be tensors, and narrowing here would break the gradient.
    """
    angles = composition.unit_cell_angles or (90.0, 90.0, 90.0)
    return unit_cell_volume(*composition.unit_cell, *angles)


def crystal_material(composition: Any) -> tuple[Any, Any]:
    """Turn a composition model into cell contents and a density.

    Parameters
    ----------
    composition : Any
        One of the composition models.

    Returns
    -------
    tuple[Any, Any]
        The :class:`~pyraddose_3d.backend.composition.Composition` and its
        density in g/cm^3. For the ``average`` method there is no
        composition, and ``None`` is returned in its place.

    Raises
    ------
    NotImplementedError
        For a method that is not ported.
    """
    method = composition.method

    if method == "average":
        return None, AVERAGE_DENSITY

    if method == "unit_cell":
        built = unit_cell_composition(
            _cell_volume(composition),
            Macromolecule(
                monomers=composition.num_monomers,
                amino_acids=composition.num_residues,
                rna_nucleotides=composition.num_rna,
                dna_nucleotides=composition.num_dna,
                carbohydrate_residues=composition.num_carb,
            ),
            heavy_protein_atoms=composition.protein_heavy_atoms or None,
            solvent_concentrations=composition.solvent_heavy_conc or None,
            solvent_fraction=composition.solvent_fraction,
        )
    elif method == "small_molecule":
        built = small_molecule_composition(
            _cell_volume(composition),
            composition.atoms,
            monomers=composition.num_monomers,
        )
    elif method == "saxs":
        volume = _cell_volume(composition)
        macromolecule = Macromolecule(
            amino_acids=composition.num_residues,
            rna_nucleotides=composition.num_rna,
            dna_nucleotides=composition.num_dna,
            carbohydrate_residues=composition.num_carb,
        )
        monomers = saxs_monomer_count(
            volume, composition.protein_concentration, macromolecule
        )
        built = unit_cell_composition(
            volume,
            Macromolecule(
                monomers=monomers,
                amino_acids=composition.num_residues,
                rna_nucleotides=composition.num_rna,
                dna_nucleotides=composition.num_dna,
                carbohydrate_residues=composition.num_carb,
            ),
            heavy_protein_atoms=composition.protein_heavy_atoms or None,
            solvent_concentrations=composition.solvent_heavy_conc or None,
            solvent_fraction=composition.solvent_fraction,
        )
    elif method == "saxs_sequence":
        volume = _cell_volume(composition)
        chains = read_sequence_file(composition.sequence_file)
        built = sequence_composition(
            volume,
            chains,
            monomers=saxs_sequence_monomer_count(
                volume,
                composition.protein_concentration,
                chains,
                composition.num_carb,
            ),
            carbohydrate_residues=composition.num_carb,
            heavy_protein_atoms=composition.protein_heavy_atoms or None,
            solvent_concentrations=composition.solvent_heavy_conc or None,
            solvent_fraction=composition.solvent_fraction,
        )
    elif method == "sequence":
        built = sequence_composition(
            _cell_volume(composition),
            read_sequence_file(composition.sequence_file),
            monomers=composition.num_monomers,
            carbohydrate_residues=composition.num_carb,
            heavy_protein_atoms=composition.protein_heavy_atoms or None,
            solvent_concentrations=composition.solvent_heavy_conc or None,
            solvent_fraction=composition.solvent_fraction,
        )
    elif method == "pdb":
        built = composition_from_pdb(
            composition.pdb,
            solvent_concentrations=composition.solvent_heavy_conc or None,
        )
    elif method == "cif":
        built = composition_from_cif(composition.cif)
    else:
        raise NotImplementedError(f"composition method {method!r} is not ported")

    return built, density(built)


def wedge_resolution(start: float, end: float, requested: float | None) -> float:
    """The angular step RADDOSE-3D actually uses, in degrees.

    Not always the one asked for. ``Wedge.java`` keeps a requested step only
    if the wedge spans more than ten of them, in either direction, and
    otherwise uses 2 degrees::

        if (angularResolution != null && Math.abs(end - start) / angularResolution > 10)

    So a 30 degree wedge at 5 degrees -- six steps -- is simulated at 2, and so
    is any wedge with no step given. The Java prints "ANGULAR RESOLUTION TOO
    BIG - RESETTING TO DEFAULT" in both cases, including the second, where
    nothing was too big.

    Parameters
    ----------
    start, end : float
        The wedge's angles, in degrees.
    requested : float | None
        ``AngularResolution`` as given, if it was.

    Returns
    -------
    float
        The step in degrees.
    """
    if requested is not None and abs(end - start) / requested > 10:
        return requested
    return DEFAULT_ANGULAR_RESOLUTION


def _generalise(path: str) -> str:
    """Replace a wedge index with ``*``, so a path can be matched."""
    return re.sub(r"^wedges\.\d+\.", "wedges.*.", path)


def check_overrides(overrides: Mapping[str, Any]) -> None:
    """Reject a path that is not tunable, saying why.

    Parameters
    ----------
    overrides : Mapping[str, Any]
        Paths to values, as passed to :func:`solve`.

    Raises
    ------
    KeyError
        If a path is not in :data:`TUNABLE`. A path in :data:`FIXED` is
        reported with the reason it cannot be optimised; anything else is
        reported as unknown, with the tunable paths listed.
    """
    for path in overrides:
        general = _generalise(path)
        if general in TUNABLE:
            continue
        if general in FIXED:
            raise KeyError(f"{path!r} cannot be optimised: {FIXED[general]}")
        raise KeyError(
            f"{path!r} is not a tunable setting. Tunable: {', '.join(sorted(TUNABLE))}"
        )


def _setting(overrides: Mapping[str, Any], path: str, fallback: Any) -> Any:
    """The override for ``path`` if there is one, else the configured value."""
    return overrides[path] if path in overrides else fallback


def apply_overrides(
    experiment: Experiment, overrides: Mapping[str, Any] | None = None
) -> Experiment:
    """The experiment with its overrides written in: what is actually run.

    Overrides carry values as tensors so that a gradient can reach them.
    This writes their current values back into the configuration as plain
    numbers, so the result can be saved as YAML, run again without torch, or
    written out as RADDOSE-3D's files -- the strategy an optimiser found, in
    a form anything else can use.

    Parameters
    ----------
    experiment : Experiment
        The configuration.
    overrides : Mapping[str, Any], optional
        Paths to values, as passed to :func:`solve`.

    Returns
    -------
    Experiment
        A validated copy with each overridden setting replaced.

    Raises
    ------
    KeyError
        If a path is not tunable, names a wedge the experiment does not have,
        or tunes a collimation or width the beam does not have.
        :func:`simulate` would otherwise never read such an override, and
        ignore it without a word.
    pydantic.ValidationError
        If a value is not a valid setting -- a negative flux, say.
    """
    overrides = dict(overrides or {})
    check_overrides(overrides)
    if not overrides:
        return experiment

    config = experiment.model_dump()
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        node: Any = config
        for key in parents:
            if isinstance(node, list):
                if int(key) >= len(node):
                    raise KeyError(
                        f"{path!r}: the experiment has {len(node)} wedge(s), "
                        f"numbered from 0"
                    )
                node = node[int(key)]
            else:
                node = node[key]
            if node is None:
                raise KeyError(f"{path!r}: the beam has no collimation to tune")
        if isinstance(node, dict) and leaf not in node:
            raise KeyError(f"{path!r}: this {node.get('type')} beam has no {leaf}")
        plain = np.asarray(to_namespace(np, value), dtype=np.float64)
        node[leaf] = plain.item() if plain.ndim == 0 else tuple(plain.tolist())
    return type(experiment).model_validate(config)


def container_transmission_at(container: Any, energy: Any) -> Any:
    """The fraction of the beam a container lets through, at an energy.

    Parameters
    ----------
    container : Any
        A container model, or ``None``.
    energy : Any
        Photon energy in keV.

    Returns
    -------
    Any
        Between 0 and 1; exactly 1 when there is no container.
    """
    if container is None:
        return 1.0
    if container.type == "mixture":
        mu_rho = mixture_mass_attenuation(container.material, energy)
    else:
        mu_rho = elemental_mass_attenuation(container.elements, energy)
    return container_transmission(mu_rho, container.density, container.thickness)


def crystal_beam(
    beam: Any,
    overrides: Mapping[str, Any] | None = None,
    container: Any = None,
) -> Any:
    """Turn a beam model into a backend beam.

    Parameters
    ----------
    beam : Any
        One of the beam models.
    overrides : Mapping[str, Any], optional
        Tensor values for any of the ``beam.*`` paths in :data:`TUNABLE`.
    container : Any, optional
        The crystal's container, which the beam crosses first and which
        attenuates it, as ``BeamGaussian.applyContainerAttenuation`` does.

    Returns
    -------
    Any
        A backend beam. It holds tensors if any override did, and the
        exposure loop then produces a tensor dose.
    """
    overrides = overrides or {}
    collimation = beam.collimation
    circular = collimation is not None and collimation.shape == "circular"

    flux = _setting(overrides, "beam.flux", beam.flux)
    energy = _setting(overrides, "beam.energy", beam.energy)
    if container is not None:
        flux = flux * container_transmission_at(container, energy)
    aperture_x = None if collimation is None else collimation.x
    aperture_y = None if collimation is None else collimation.y
    if aperture_x is not None:
        aperture_x = _setting(overrides, "beam.collimation.x", aperture_x)
        aperture_y = _setting(overrides, "beam.collimation.y", aperture_y)

    if beam.type == "tophat":
        return backend_beam.TophatBeam(flux, energy, aperture_x, aperture_y, circular)
    if beam.type == "experimental":
        return backend_beam.ExperimentalBeam(
            flux,
            energy,
            backend_beam.read_pgm(beam.file),
            beam.pixel_size[0],
            beam.pixel_size[1],
            None if aperture_x is None else float(aperture_x),
            None if aperture_y is None else float(aperture_y),
            circular,
        )

    fwhm = _setting(overrides, "beam.fwhm", beam.fwhm)
    return backend_beam.GaussianBeam(
        flux,
        energy,
        fwhm[0],
        fwhm[1],
        aperture_x,
        aperture_y,
        circular,
    )


def decay_function(name: str, parameters: tuple[float, float, float] | None) -> Any:
    """The relative diffraction efficiency as a function of accumulated dose.

    Parameters
    ----------
    name : str
        One of ``simple``, ``linear``, ``leal`` or ``bfactor``.
    parameters : tuple[float, float, float] | None
        ``(gamma, b0, beta)`` for the B-factor and Leal models. When they are
        omitted RADDOSE-3D sets every one to 9 for the B-factor model and to
        0 for Leal's, which then does not decay at all.

    Returns
    -------
    Any
        A callable from dose in MGy to a factor between 0 and 1.
    """
    if name == "simple":
        return lambda dose: 1.0

    if name == "linear":

        def linear(dose: Any) -> Any:
            xp = _namespace(dose)
            return xp.clip(1 - dose / _OWEN_2006_DOSE_LIMIT, 0.0, None)

        return linear

    if name == "bfactor":
        gamma, b_zero, beta = parameters or (9.0, 9.0, 9.0)

        def bfactor(dose: Any) -> Any:
            xp = _namespace(dose)
            return xp.exp(-(dose * b_zero / beta) / 2 * (1 / gamma) ** 2)

        return bfactor

    # DDMLeal reads (gamma, b0, beta), and uses zeros -- no decay -- if any
    # is missing.
    gamma, b_zero, beta = parameters or (0.0, 0.0, 0.0)
    return leal_decay(gamma, b_zero, beta)


def _namespace(value: Any) -> Any:
    from pyraddose_3d.backend._array import array_namespace

    return array_namespace(value)


@dataclass(frozen=True)
class Simulation:
    """Everything a run produced, not just the reported numbers.

    :func:`solve` returns only the summaries, which is what an optimiser
    wants. The output files and the conformance checks need the fields the
    summaries were reduced from -- which voxels are inside the crystal,
    where they are, and what each received.

    Attributes
    ----------
    experiment : Experiment
        What was run: the configuration with any overrides written in, as
        :func:`apply_overrides` gives it.
    mesh : Mesh
        The crystal, with its angles applied. Its bounding box is what
        RADDOSE-3D reports as the crystal's size.
    coefficients : MaterialCoefficients
        Absorption, attenuation, elastic and inelastic coefficients at the
        beam energy, in inverse micrometres. For a pink beam, at the last
        energy sampled, as RADDOSE-3D reports them.
    grid : VoxelGrid
        The voxel grid, in the lab frame before any rotation.
    occupied : Array
        ``(N,)`` boolean, which voxels are inside the crystal.
    density : Array
        Crystal density in g/cm^3.
    angles : list[list[Any]]
        The rotation angles of each wedge, in radians.
    exposures : list[ExposureResult]
        One per wedge. ``dose`` is cumulative over all wedges so far;
        ``fluence``, ``elastic`` and ``absorbed_energy`` are that wedge's
        alone, as in RADDOSE-3D's per-wedge observer.
    summaries : list[WedgeSummary]
        One per wedge -- what :func:`solve` returns.
    seed : int | None
        The seed escape modelling drew its random numbers from: the
        experiment's, or one chosen at random if it gave none. ``None``
        without escape.
    """

    experiment: Experiment
    mesh: Mesh
    coefficients: MaterialCoefficients
    grid: VoxelGrid
    occupied: Any
    density: Any
    angles: list[list[Any]]
    exposures: list[ExposureResult]
    summaries: list[WedgeSummary]
    seed: int | None = None


def microed(experiment: Experiment) -> MicroEDResult:
    """Run RADDOSE-3D's MicroED engine: ``Subprogram EMED``, or ``EMSP``.

    A deterministic calculation, ported from ``MicroED.CalculateEM``: the dose
    to a thin crystal from its stopping power, slice by slice; how many
    electrons scatter elastically and inelastically; and a grid search for the
    voltage and thickness that would give the most diffraction information per
    unit dose. See :mod:`pyraddose_3d.backend.microed`.

    The crystal's extent along each axis, after ``angle_p`` and ``angle_l``,
    sets its size; z is along the beam, so it is the thickness. MicroED
    ignores the wedges, escape, the container and the decay model.

    Parameters
    ----------
    experiment : Experiment
        A configuration whose crystal has ``subprogram: microed`` and whose
        beam is an electron beam.

    Returns
    -------
    MicroEDResult
        Everything RADDOSE-3D reports.

    Raises
    ------
    ValueError
        If the experiment is not a MicroED one, or if its composition is the
        average one, which has no elements to scatter from.
    """
    crystal = experiment.crystal
    beam = experiment.beam
    if crystal.subprogram != "microed" or beam.type != "electron":
        raise ValueError(
            f"this experiment runs RADDOSE-3D's {_ENGINES[crystal.subprogram]}, "
            f"not its MicroED calculation"
        )
    built, crystal_density = crystal_material(crystal.composition)
    if built is None:
        raise ValueError(
            "MicroED needs the crystal's elements, and the average composition "
            "has none; describe the composition another way"
        )
    material = electron_material(built, crystal_density)
    vertices = np.asarray(crystal_mesh(crystal).vertices)
    x, y, z = (
        1000 * (float(vertices[:, axis].max()) - float(vertices[:, axis].min()))
        for axis in range(3)
    )
    if crystal.type == "cylinder":
        # MicroED's constructor takes 0.001 nm off a cylinder's thickness --
        # in RADDOSE-3D, only when the input gave two dimensions, the
        # documented form; a third, otherwise ignored, makes it a cuboid.
        z -= 0.001
    return run_microed(
        material,
        dimensions_nm=(x, y, z),
        aperture=(beam.collimation.x, beam.collimation.y),
        circular=beam.collimation.shape == "circular",
        energy=beam.energy,
        exposure=beam.exposure,
    )


def solve(
    experiment: Experiment,
    overrides: Mapping[str, Any] | None = None,
) -> list[WedgeSummary]:
    """Run an experiment and return one summary per wedge.

    Parameters
    ----------
    experiment : Experiment
        The validated configuration.
    overrides : Mapping[str, Any], optional
        Values to use in place of the configured ones, keyed by
        configuration path -- ``"beam.flux"``, ``"wedges.0.exposure_time"``.
        Only paths in :data:`TUNABLE` are accepted. Pass tensors here and
        the whole calculation runs in torch and the summaries come back as
        tensors, so a strategy can be optimised by gradient descent::

            fwhm = torch.tensor([20.0, 20.0], dtype=torch.float64, requires_grad=True)
            summary = solve(experiment, {"beam.fwhm": fwhm})[0]
            summary.max_dose.backward()

    Returns
    -------
    list[WedgeSummary]
        One entry per wedge, in order, each holding the sixteen reported
        numbers.

    Raises
    ------
    KeyError
        If an override names a path that is not tunable. Crystal geometry is
        rejected explicitly: autograd returns a number for it and the number
        is wrong, because the occupancy test is a boolean parity count.
    NotImplementedError
        If escape modelling is asked for with tensor overrides: it is Monte
        Carlo, and has no gradient.

    Notes
    -----
    With no overrides this is pure NumPy, which is what the conformance
    corpus runs and what a CCP4 install without torch gets. The crystal mesh
    and the voxel grid stay NumPy either way: they are fixed geometry whose
    occupancy is counted rather than measured, so there is no gradient there
    worth carrying. :func:`~pyraddose_3d.backend._array.to_namespace`
    reconciles the two at the boundary.
    """
    return simulate(experiment, overrides).summaries


def _beam_minimum_dimension(beam: Any, built: Any) -> float:
    """``Beam.beamMinumumDimension``: the narrowest of the beam's widths.

    From the model, so that a tensor override does not reach it; an
    experimental beam's comes from its image, which ``built`` has read.
    """
    if beam.type == "experimental":
        return float(built.minimum_dimension)
    collimation = beam.collimation
    if beam.type == "tophat":
        assert collimation is not None
        return float(min(collimation.x, collimation.y))
    narrowest = float(min(beam.fwhm[0], beam.fwhm[1]))
    if collimation is not None and min(collimation.x, collimation.y) != 0:
        return float(min(min(collimation.x, collimation.y), narrowest))
    return narrowest


def _escape(
    experiment: Experiment,
    built: Any,
    crystal_density: Any,
    mesh: Mesh,
    grid: VoxelGrid,
    occupied: Any,
    beam: Any,
    energies: Any,
    overrides: Mapping[str, Any],
    seed: int,
) -> Escape:
    """Set up photoelectron and fluorescence escape, as ``Crystal.expose`` does."""
    crystal = experiment.crystal
    options = crystal.escape
    if any(is_torch(v) for v in overrides.values()):
        raise NotImplementedError(
            "escape modelling chooses each photoelectron's direction at random, "
            "so it has no gradient; run it without tensor overrides"
        )
    # The crystal model refuses escape with the average composition.
    assert built is not None

    beam_energy = float(to_namespace(np, beam.energy))
    density = float(to_namespace(np, crystal_density))
    counts = np.asarray(to_namespace(np, built.counts), dtype=np.float64)

    def photoelectric_at(energy: float) -> float:
        coefficients = material_coefficients(built, energy)
        return float(to_namespace(np, coefficients.absorption))

    factors = escape_factors(built.elements, counts, beam_energy, photoelectric_at)
    vertical = options.goniometer_axis == 90.0
    ppm = grid.pixels_per_micron
    vertices = np.asarray(to_namespace(np, mesh.vertices))
    low, high = vertices.min(axis=0), vertices.max(axis=0)

    photoelectron = bins = None
    if options.photoelectron:
        photoelectron, bins = photoelectron_escape(
            factors, beam_energy, density, ppm, options.pe_resolution, vertical
        )
    fluorescence = None
    if options.fluorescence:
        fluorescence = fluorescence_escape(factors, high - low, options.fl_resolution)

    surrounding = surrounding_absorption = None
    if options.surrounding and photoelectron is not None:
        assert bins is not None
        oil_density = options.surrounding_density
        oil_elements = options.surrounding_elements
        container = crystal.container
        if crystal.composition.method in ("saxs", "saxs_sequence"):
            # The SAXS routes take the oil from the container when it is not
            # given, as the grammar does.
            if oil_density is None and container is not None:
                oil_density = container.density
            if (
                not oil_elements
                and container is not None
                and container.type == "elemental"
            ):
                oil_elements = container.elements
        cell, surrounding_density = surrounding_composition(
            float(to_namespace(np, built.cell_volume)),
            options.surrounding_heavy_conc,
            options.density_based,
            oil_elements,
            oil_density or 0.0,
        )
        cell_factors = escape_factors(
            cell.elements,
            np.asarray(cell.counts, dtype=np.float64),
            beam_energy,
            photoelectric_at,
            surrounding=True,
        )
        surrounding = surrounding_escape(
            cell_factors, beam_energy, surrounding_density, density, bins,
            low, high, ppm, _beam_minimum_dimension(experiment.beam, beam), vertical,
        )  # fmt: skip
        at = [beam_energy] if energies is None else list(np.asarray(energies))
        surrounding_absorption = np.array(
            [float(material_coefficients(cell, e).absorption) for e in at]
        )

    return Escape(
        JavaRandom(seed),
        grid.shape,
        np.asarray(to_namespace(np, occupied)),
        ppm,
        density,
        beam_energy,
        photoelectron,
        fluorescence,
        surrounding,
        surrounding_absorption,
    )


def monte_carlo_setup(experiment: Experiment) -> SimpleSetup:
    """What RADDOSE-3D's Monte Carlo and XFEL subprograms read of an experiment.

    The last wedge: each wedge reruns the simulation, and RADDOSE-3D's files
    keep the last's.

    Parameters
    ----------
    experiment : Experiment
        A crystal with ``subprogram: montecarlo``, ``gos`` or ``xfel``.

    Returns
    -------
    SimpleSetup
        For :func:`pyraddose_3d.backend.montecarlo.simple.simulate_simple`,
        or for the XFEL subprogram
        :func:`pyraddose_3d.backend.montecarlo.xfel.simulate_xfel`.

    Raises
    ------
    NotImplementedError
        For the XFEL subprogram with a surrounding, or with a wedge it would
        rotate or offset: not ported (``UPSTREAM_FIXES.md``).
    """
    from pyraddose_3d.backend.montecarlo.photons import BeamShape
    from pyraddose_3d.backend.montecarlo.simple import SimpleSetup, Surrounding
    from pyraddose_3d.models.beam import GaussianBeam as GaussianBeamModel
    from pyraddose_3d.models.beam import TophatBeam as TophatBeamModel

    crystal, beam = experiment.crystal, experiment.beam
    if crystal.subprogram not in ("montecarlo", "gos", "xfel"):
        raise ValueError("the Monte Carlo is Subprogram MONTECARLO, GOS or XFEL")
    xfel = crystal.subprogram == "xfel"
    # The model allows only these, with a flux for the Monte Carlo and a
    # pulse energy for the XFEL subprogram.
    assert isinstance(beam, (GaussianBeamModel, TophatBeamModel))
    assert beam.pulse_energy is not None if xfel else beam.flux is not None
    mesh = crystal_mesh(crystal)
    built, density_ = crystal_material(crystal.composition)
    assert built is not None
    photons = crystal.simulation.photons
    energies = None
    if beam.energy_fwhm is not None:
        # One energy per photon, ascending: the first photons of a run, and
        # so of a rotation, are the softest.
        energies = np.asarray(
            pink_beam_energies(float(beam.energy), beam.energy_fwhm, photons)
        )
    surrounding = None
    options = crystal.escape
    if options.surrounding:
        cell, cell_density = surrounding_composition(
            float(to_namespace(np, built.cell_volume)),
            options.surrounding_heavy_conc,
            options.density_based,
            options.surrounding_elements,
            options.surrounding_density or 0.0,
        )
        surrounding = Surrounding(
            composition=cell,
            density=float(cell_density),
            material=electron_material(cell, cell_density),
            thickness_um=crystal.simulation.surrounding_thickness or (0.0, 0.0, 0.0),
        )
    _refuse_untabulated(built, surrounding, beam.energy, energies)
    wedge = experiment.wedges[-1]
    resolution = wedge_resolution(wedge.start, wedge.end, wedge.angular_resolution)
    ox, oy, oz = wedge.start_offset or (0.0, 0.0, 0.0)
    tx, ty, tz = wedge.translate_per_degree or (0.0, 0.0, 0.0)
    if xfel:
        _refuse_unported_xfel(
            surrounding is not None,
            math.radians(wedge.start),
            math.radians(wedge.end),
            math.radians(resolution),
            (ox, oy, oz),
            (math.degrees(tx), math.degrees(ty), math.degrees(tz)),
        )
    return SimpleSetup(
        mesh=Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles)),
        pixels_per_micron=pixels_per_micron(crystal, mesh),
        composition=built,
        density=float(density_),
        material=electron_material(built, density_),
        beam=BeamShape.from_model(beam),
        photon_energy=float(beam.energy),
        flux=float(beam.flux or 0.0),
        exposure=float(wedge.exposure_time),
        start_angle=math.radians(wedge.start),
        end_angle=math.radians(wedge.end),
        angular_resolution=math.radians(resolution),
        start_offset=(ox, oy, oz),
        translation_per_radian=(math.degrees(tx), math.degrees(ty), math.degrees(tz)),
        photons=photons,
        runs=crystal.simulation.runs,
        rotated_polarisation=(crystal.escape.goniometer_axis == 90.0)
        != (crystal.simulation.polarisation_direction == 90.0),
        energies=energies,
        surrounding=surrounding,
        gos=crystal.subprogram == "gos",
        xfel=xfel,
        pulse_energy=float(beam.pulse_energy or 0.0),
    )


def _refuse_unported_xfel(
    surrounding: bool,
    start: float,
    end: float,
    resolution: float,
    offset: tuple[float, float, float],
    translation_per_radian: tuple[float, float, float],
) -> None:
    """The XFEL subprogram's inputs the port does not run.

    Its surrounding is a model of its own, unlike the Monte Carlo's. Its
    rotation turns only the voxels the dose is tallied in, by whole radians
    of the wedge, and its offset moves only them: the crystal stays where it
    is. Angles in radians, lengths in um.
    """
    if surrounding:
        raise NotImplementedError(
            "RADDOSE-3D's XFEL subprogram models the surrounding its own way, "
            "which pyRADDOSE-3D does not run (UPSTREAM_FIXES.md); leave the "
            "surrounding out, or use Subprogram GOS"
        )
    still = abs(start - end) < resolution or end == 0
    if not still and math.trunc(end - start) != 0:
        raise NotImplementedError(
            "RADDOSE-3D's XFEL subprogram turns the voxels the dose is "
            "tallied in, but not the crystal, by whole radians of a wedge of "
            "1 radian (57.3 degrees) or more, which pyRADDOSE-3D does not run "
            "(UPSTREAM_FIXES.md); an XFEL pulse is a still"
        )
    moved = [
        o + t * (0.0 - start)
        for o, t in zip(offset, translation_per_radian, strict=True)
    ]
    if any(moved):
        raise NotImplementedError(
            "RADDOSE-3D's XFEL subprogram moves the voxels the dose is tallied "
            "in by the wedge's offset, but not the crystal, which pyRADDOSE-3D "
            "does not run (UPSTREAM_FIXES.md); leave the offset out"
        )


def _refuse_untabulated(
    built: Any, surrounding: Any, energy: float, energies: Any
) -> None:
    """``SubprogramInputs.check``: ELSEPA's tables stop at Z = 52 above 20 keV."""
    from pyraddose_3d.backend.montecarlo.angles import HIGH_ENERGY_KEV, MAX_Z

    highest = float(np.max(energies)) if energies is not None else float(energy)
    if highest <= HIGH_ENERGY_KEV:
        return
    for where, composition in (("crystal", built), ("surrounding", surrounding)):
        if composition is None:
            continue
        cell = getattr(composition, "composition", composition)
        for element in cell.elements:
            if element.atomic_number > MAX_Z[True]:
                raise ValueError(
                    f"The Monte Carlo subprogram has electron scattering tables "
                    f"above 20 keV only up to Z = {MAX_Z[True]}, and the {where} "
                    f"contains {element.name} (Z = {element.atomic_number}) in a "
                    f"beam reaching {highest:.3f} keV."
                )


def monte_carlo(experiment: Experiment) -> list[SimpleRun]:
    """Run RADDOSE-3D's Monte Carlo, ``Subprogram MONTECARLO``, ``GOS`` or ``XFEL``.

    Compared with RADDOSE-3D statistically, by
    ``tests/conformance/test_montecarlo_files.py``.

    Parameters
    ----------
    experiment : Experiment
        A crystal with ``subprogram: montecarlo``, ``gos`` or ``xfel``. Its
        ``seed`` seeds NumPy's generator; the port does not reproduce
        RADDOSE-3D's draws, only their distribution.

    Returns
    -------
    list[SimpleRun]
        One per run.
    """
    from pyraddose_3d.backend.montecarlo.simple import simulate_simple
    from pyraddose_3d.backend.montecarlo.xfel import simulate_xfel

    setup = monte_carlo_setup(experiment)
    rng = np.random.default_rng(experiment.seed)
    if setup.xfel:
        return simulate_xfel(setup, rng)
    return simulate_simple(setup, rng)


#: How messages name RADDOSE-3D's engines.
_ENGINES = {
    "rd3d": "MX calculation (call solve or simulate)",
    "microed": "MicroED calculation",
    "montecarlo": "Monte Carlo subprogram (Subprogram MONTECARLO)",
    "gos": "Monte Carlo subprogram (Subprogram GOS)",
    "xfel": "XFEL subprogram",
}


def simulate(
    experiment: Experiment,
    overrides: Mapping[str, Any] | None = None,
    *,
    record_images: bool = False,
) -> Simulation:
    """Run an experiment and keep everything it produced.

    The same calculation as :func:`solve`, returning the per-voxel fields
    alongside the summaries.

    Parameters
    ----------
    experiment : Experiment
        The validated configuration.
    overrides : Mapping[str, Any], optional
        As for :func:`solve`.
    record_images : bool, optional
        Also keep every voxel's dose, fluence and decay after every image.
        Needed for per-image outputs; off by default because it costs a
        field per image.

    Returns
    -------
    Simulation
        The grid, the per-wedge exposures and the summaries.

    Raises
    ------
    KeyError
        If an override names a path that is not tunable.
    ValueError
        If the experiment runs RADDOSE-3D's MicroED engine, or its Monte
        Carlo or XFEL subprogram, which have entry points of their own.
    NotImplementedError
        If escape modelling is asked for with tensor overrides.
    """
    xray = experiment.beam
    if xray.type == "electron":
        raise ValueError(
            "this experiment runs RADDOSE-3D's MicroED engine; call "
            "pyraddose_3d.solve.microed"
        )
    if experiment.crystal.subprogram in SIMULATED:
        raise ValueError(
            f"this experiment runs RADDOSE-3D's "
            f"{_ENGINES[experiment.crystal.subprogram]}; call "
            "pyraddose_3d.solve.monte_carlo"
        )
    overrides = dict(overrides or {})
    as_run = apply_overrides(experiment, overrides)

    crystal = experiment.crystal

    mesh = crystal_mesh(crystal)
    grid = voxel_grid(mesh, pixels_per_micron(crystal, mesh))
    coordinates = grid.coordinates()
    occupied = occupancy(mesh, coordinates)

    built, crystal_density = crystal_material(crystal.composition)
    beam = crystal_beam(xray, overrides, crystal.container)

    # A pink beam is sampled at 200 energies about its mean, and the crystal
    # absorbs and scatters differently at each.
    energies = None
    if xray.energy_fwhm is not None:
        energies = pink_beam_energies(beam.energy, xray.energy_fwhm)
    if built is None:
        coefficients = AVERAGE_COEFFICIENTS
        if energies is not None:
            # CoefCalcAverage's coefficients do not depend on energy.
            coefficients = MaterialCoefficients(
                *(
                    np.full(len(energies), value)
                    for value in dataclasses.astuple(coefficients)
                )
            )
    else:
        # The beam's energy, override included: how strongly the crystal
        # absorbs and scatters depends on it as much as the beam does.
        coefficients = material_coefficients(
            built, beam.energy if energies is None else energies
        )
    # What RADDOSE-3D reports: the coefficients at the beam's energy, the
    # mean of a pink beam's.
    reported = coefficients
    if energies is not None:
        reported = (
            AVERAGE_COEFFICIENTS
            if built is None
            else material_coefficients(built, beam.energy)
        )

    decay = decay_function(crystal.decay_model, crystal.decay_parameters)
    seed = experiment.seed
    escape = None
    if crystal.escape.photoelectron or crystal.escape.fluorescence:
        if seed is None:
            # RADDOSE-3D seeds from the clock and prints the seed it chose.
            seed = int(np.random.default_rng().integers(0, 2**47))
        escape = _escape(
            experiment, built, crystal_density, mesh, grid, occupied,
            beam, energies, overrides, seed,
        )  # fmt: skip

    summaries = []
    exposures = []
    wedge_angles = []
    # RADDOSE-3D holds dose on the crystal, so wedge two starts where
    # wedge one left off and its summary covers the whole history.
    carried = None
    carried_seen = None
    for number, wedge in enumerate(experiment.wedges, start=1):
        index = number - 1
        first = _setting(overrides, f"wedges.{index}.start", wedge.start)
        last = _setting(overrides, f"wedges.{index}.end", wedge.end)
        # The step is a discrete choice, so it is taken on detached values.
        resolution = math.radians(
            wedge_resolution(
                float(to_namespace(np, first)),
                float(to_namespace(np, last)),
                wedge.angular_resolution,
            )
        )
        angles = exposure_angles(first * _DEG_TO_RAD, last * _DEG_TO_RAD, resolution)

        offset = _setting(overrides, f"wedges.{index}.start_offset", wedge.start_offset)
        per_degree = _setting(
            overrides,
            f"wedges.{index}.translate_per_degree",
            wedge.translate_per_degree,
        )
        seconds = _setting(
            overrides, f"wedges.{index}.exposure_time", wedge.exposure_time
        )
        off_axis = _setting(
            overrides,
            f"wedges.{index}.rot_axis_beam_offset",
            wedge.rot_axis_beam_offset or 0.0,
        )

        xp = array_namespace(
            *angles, offset, per_degree, seconds, off_axis, beam.flux, beam.energy
        )
        offset = to_namespace(xp, offset if offset is not None else (0.0, 0.0, 0.0))
        per_degree = to_namespace(
            xp, per_degree if per_degree is not None else (0.0, 0.0, 0.0)
        )
        # The translation accumulates with the angle turned since the start
        # of the wedge, in degrees. Measured from the configured start rather
        # than from the first angle, which differ when the wedge begins past
        # a full turn -- exposure_angles reduces the angles and not this.
        turned = to_namespace(
            xp, [(angle - first * _DEG_TO_RAD) * _RAD_TO_DEG for angle in angles]
        )
        translations = offset + turned[:, None] * per_degree

        result = expose(
            mesh,
            grid,
            occupied,
            beam,
            coefficients,
            crystal_density,
            angles,
            seconds,
            translations=translations,
            off_axis=off_axis,
            decay_model=decay,
            initial_dose=carried,
            last_observed=carried_seen,
            record_images=record_images,
            energies=energies,
            escape=escape,
        )
        carried = result.dose
        carried_seen = result.last_observed
        exposures.append(result)
        wedge_angles.append(list(angles))
        summaries.append(
            summarise(result, grid.volume, crystal_density, wedge_number=number)
        )

    return Simulation(
        experiment=as_run,
        mesh=mesh,
        coefficients=reported,
        grid=grid,
        occupied=occupied,
        density=crystal_density,
        angles=wedge_angles,
        exposures=exposures,
        summaries=summaries,
        seed=seed,
    )
