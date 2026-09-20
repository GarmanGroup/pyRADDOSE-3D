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

import math
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

from pyraddose_3d.backend import beam as backend_beam
from pyraddose_3d.backend.coefficients import (
    AVERAGE_COEFFICIENTS,
    AVERAGE_DENSITY,
    density,
    material_coefficients,
    unit_cell_volume,
)
from pyraddose_3d.backend.composition import (
    Macromolecule,
    saxs_monomer_count,
    sequence_composition,
    small_molecule_composition,
    unit_cell_composition,
)
from pyraddose_3d.backend.exposure import expose, exposure_angles
from pyraddose_3d.backend.geometry import Mesh
from pyraddose_3d.backend.structure_files import (
    composition_from_cif,
    composition_from_pdb,
)
from pyraddose_3d.backend.summary import WedgeSummary, summarise
from pyraddose_3d.backend.voxels import occupancy, voxel_grid

if TYPE_CHECKING:
    from pyraddose_3d.models.experiment import Experiment

__all__ = ["crystal_mesh", "solve"]

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
    NotImplementedError
        For a polyhedron read from a file, which is not ported.
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
        raise NotImplementedError(
            f"crystal type {crystal.type!r} is not ported; a polyhedron read "
            f"from a file needs an .obj reader"
        )

    return Mesh(
        _apply_crystal_angles(mesh.vertices, crystal.angle_p, crystal.angle_l),
        mesh.triangles,
    )


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
            ),
            heavy_protein_atoms=composition.protein_heavy_atoms or None,
            solvent_concentrations=composition.solvent_heavy_conc or None,
            solvent_fraction=composition.solvent_fraction,
        )
    elif method == "sequence":
        built = sequence_composition(
            _cell_volume(composition),
            _read_sequence(composition.sequence_file),
            monomers=composition.num_monomers,
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


def _read_sequence(path: str) -> str:
    """Read a FASTA file, dropping its comment lines."""
    from pathlib import Path

    return "".join(
        line.strip()
        for line in Path(path).read_text("ascii").splitlines()
        if not line.startswith(">")
    )


def crystal_beam(beam: Any) -> Any:
    """Turn a beam model into a backend beam."""
    collimation = beam.collimation
    circular = collimation is not None and collimation.shape == "circular"

    if beam.type == "tophat":
        return backend_beam.TophatBeam(
            beam.flux, beam.energy, collimation.x, collimation.y, circular
        )
    return backend_beam.GaussianBeam(
        beam.flux,
        beam.energy,
        beam.fwhm[0],
        beam.fwhm[1],
        None if collimation is None else collimation.x,
        None if collimation is None else collimation.y,
        circular,
    )


def decay_function(name: str, parameters: tuple[float, float, float] | None) -> Any:
    """The relative diffraction efficiency as a function of accumulated dose.

    Parameters
    ----------
    name : str
        One of ``simple``, ``linear``, ``leal`` or ``bfactor``.
    parameters : tuple[float, float, float] | None
        ``(gamma, b0, beta)`` for the B-factor model. RADDOSE-3D defaults
        every one of them to 9 when they are omitted.

    Returns
    -------
    Any
        A callable from dose in MGy to a factor between 0 and 1.

    Raises
    ------
    NotImplementedError
        For the Leal model, which is not ported.
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

    raise NotImplementedError(
        f"decay model {name!r} is not ported; Leal et al. needs its "
        f"integrated-intensity machinery"
    )


def _namespace(value: Any) -> Any:
    from pyraddose_3d.backend._array import array_namespace

    return array_namespace(value)


def solve(experiment: Experiment) -> list[WedgeSummary]:
    """Run an experiment and return one summary per wedge.

    Parameters
    ----------
    experiment : Experiment
        The validated configuration.

    Returns
    -------
    list[WedgeSummary]
        One entry per wedge, in order, each holding the sixteen reported
        numbers.

    Raises
    ------
    NotImplementedError
        If the configuration asks for something not ported -- escape
        modelling, the Leal decay model, or a polyhedron from a file.
    """
    crystal = experiment.crystal
    if crystal.escape.any_enabled:
        raise NotImplementedError(
            "photoelectron and fluorescence escape are Monte Carlo and are "
            "not ported; those cases are tier B"
        )

    mesh = crystal_mesh(crystal)
    grid = voxel_grid(mesh, crystal.pixels_per_micron)
    coordinates = grid.coordinates()
    occupied = occupancy(mesh, coordinates)

    built, crystal_density = crystal_material(crystal.composition)
    beam = crystal_beam(experiment.beam)

    if built is None:
        coefficients = AVERAGE_COEFFICIENTS
    else:
        coefficients = material_coefficients(built, experiment.beam.energy)

    decay = decay_function(crystal.decay_model, crystal.decay_parameters)

    summaries = []
    # RADDOSE-3D holds dose on the crystal, so wedge two starts where
    # wedge one left off and its summary covers the whole history.
    carried = None
    for number, wedge in enumerate(experiment.wedges, start=1):
        resolution = math.radians(
            wedge.angular_resolution
            if wedge.angular_resolution is not None
            else DEFAULT_ANGULAR_RESOLUTION
        )
        angles = exposure_angles(
            math.radians(wedge.start), math.radians(wedge.end), resolution
        )

        start = np.array(wedge.start_offset or (0.0, 0.0, 0.0))
        per_degree = np.array(wedge.translate_per_degree or (0.0, 0.0, 0.0))
        # The translation accumulates with the angle turned since the start
        # of the wedge, in degrees.
        turned = np.degrees(np.array(angles) - math.radians(wedge.start))
        translations = start + turned[:, None] * per_degree

        result = expose(
            mesh,
            grid,
            occupied,
            beam,
            coefficients,
            crystal_density,
            angles,
            wedge.exposure_time,
            translations=translations,
            off_axis=wedge.rot_axis_beam_offset or 0.0,
            decay_model=decay,
            initial_dose=carried,
        )
        carried = result.dose
        summaries.append(
            summarise(result, grid.volume, crystal_density, wedge_number=number)
        )

    return summaries
