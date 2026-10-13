"""RADDOSE-3D's MicroED engine: the parts the conformance cases cannot see.

The cases in ``tests/conformance/test_microed_files.py`` compare whole runs with
the Java. These pin the pieces underneath -- the ELSEPA grid, the trapezium
rule's point count, the refusals -- and the engine's place in the models.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import pytest

from pyraddose_3d.backend import microed as backend
from pyraddose_3d.backend.beam import KEV_TO_JOULES
from pyraddose_3d.models.composition import UnitCellComposition
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.outputs import write_microed_outputs
from pyraddose_3d.solve import crystal_material, microed, simulate

if TYPE_CHECKING:
    from pathlib import Path

BASELINE = {
    "unit_cell": [78.02, 78.02, 78.02],
    "num_monomers": 24,
    "num_residues": 51,
    "protein_heavy_atoms": {"Zn": 0.333, "S": 6.0},
    "solvent_heavy_conc": {"P": 425.0},
    "solvent_fraction": 0.64,
}


def material(**changes: object) -> backend.ElectronMaterial:
    composition, density = crystal_material(
        UnitCellComposition(**{**BASELINE, **changes})
    )
    assert composition is not None
    return backend.electron_material(composition, density)


def experiment(**crystal: object) -> Experiment:
    return Experiment(
        crystal={
            "type": "cuboid",
            "dimensions": [2.0, 2.0, 0.3],
            "subprogram": "microed",
            "composition": {"method": "unit_cell", **BASELINE},
            **crystal,
        },
        beam={
            "type": "electron",
            "energy": 300.0,
            "exposure": 1.0,
            "collimation": {"shape": "rectangular", "x": 4.0, "y": 4.0},
        },
        wedges=[{"start": 0.0, "end": 0.0, "exposure_time": 1.0}],
    )


# --- ELSEPA ----------------------------------------------------------------


def test_elsepa_energies_are_the_javas_map_keys() -> None:
    """``(double)(i - 2) * 0.05``: 0.15 is not exactly 0.15, and must not be."""
    energies = backend._elsepa_energies()
    assert len(energies) == 319
    assert energies[0] == 0.05
    assert energies[2] == 3 * 0.05 == 0.15000000000000002
    assert energies[19:21] == (1.0, 2.0)
    assert energies[-1] == 300.0


def test_elastic_cross_section_is_linear_between_grid_points() -> None:
    table = backend._elsepa()[6]
    i = table.energies.index(100.0)
    low, high = table.cross_sections[i], table.cross_sections[i + 1]
    assert table.at(100.0) == low
    assert table.at(100.5) == pytest.approx((low + high) / 2, rel=1e-15)
    assert table.at(300.5) == 0.0
    assert table.at(0.04) == 0.0


def test_hydrogen_keeps_the_tables_own_atomic_weight() -> None:
    """1.01 in ``fullelsepa.csv``, against 1.008 in the element table."""
    assert backend._elsepa()[1].atomic_weight == 1.01


# --- The GOS model ---------------------------------------------------------


def _java_distant_integral(energy: float, binding: float, wk: float) -> float:
    """``integrateDist`` and ``getpDisW``, transcribed loop for loop."""
    uk = binding * 1000
    wak = backend._wak(energy, wk, uk)
    w_dis = ((3 * wak - 2 * uk) / 1000) * KEV_TO_JOULES
    step = w_dis / 1000
    # getpDisW repeats integrateDist's arithmetic, by way of eV.
    p_uk = binding * 1000
    p_wak = backend._wak(energy, wk, p_uk)
    p_w_dis = ((3 * p_wak - 2 * p_uk) / 1000) * KEV_TO_JOULES
    p_uk_j = (p_uk / 1000) * KEV_TO_JOULES
    w = binding * KEV_TO_JOULES
    total = previous = 0.0
    count = 0
    while w <= w_dis:
        density = 0.0
        if p_uk_j <= w < p_w_dis:
            density = (2 / math.pow(p_w_dis - p_uk_j, 2)) * (p_w_dis - w)
        y = 0.0 if w == 0 else math.pow(w, -1) * density
        if count > 0:
            total += step * ((y + previous) / 2)
        count += 1
        w += step
        previous = y
    return total


@pytest.mark.parametrize("energy", [10.0, 120.0, 300.0, 2000.0])
def test_the_vectorised_distant_integral_is_the_javas_loop(energy: float) -> None:
    """Same points, same sums, for every subshell of every element."""
    m = material(protein_heavy_atoms={"Zn": 0.333, "S": 6.0, "Se": 1.0})
    for atom in m.atoms:
        _, inner = atom.valence
        for index in range(inner + 1):
            wk = backend._wk(m, m.sternheimer_a, atom, index)
            binding = atom.subshell_binding(index)
            assert backend._distant_integral(
                energy, binding, wk
            ) == _java_distant_integral(energy, binding, wk), (atom.symbol, index)


def test_the_baseline_matches_the_javas_mean_free_paths() -> None:
    """``Elastic Lambda`` and ``getGOSInel`` from the Java.

    The elastic path agrees to every digit. The inelastic one to 2 ulp: the
    Java sums elements in a ``HashSet``'s order.
    """
    m = material()
    assert backend.elastic_mean_free_path(m, 300.0) == 516.2419669247315
    assert backend.inelastic_mean_free_path(m, 300.0) == pytest.approx(
        253.15032210625571, rel=1e-15
    )


# --- Binding energies -------------------------------------------------------


def _run(
    m: backend.ElectronMaterial, thickness: float = 300.0
) -> backend.MicroEDResult:
    return backend.microed(
        m,
        dimensions_nm=(2000.0, 2000.0, thickness),
        aperture=(4.0, 4.0),
        circular=False,
        energy=300.0,
        exposure=1.0,
    )


def test_every_element_has_the_binding_energies_the_gos_model_reads() -> None:
    """Na, Ca, Sc, Ti, V, Cr, Ga and Ge had zeros, and Al a blank, until
    RADDOSE-3D #40 filled them from the X-ray Data Booklet and, for Al 3s
    and Cr 3d, Carlson."""
    table = backend._shell_table()
    atoms = tuple(
        backend._Atom(
            symbol=str(z),
            z=z,
            atomic_weight=1.0,
            count=1.0,
            shells=shells,
            elastic=backend._elsepa()[1],
        )
        for z, shells in table.items()
    )
    every = dataclasses.replace(material(), atoms=atoms)
    assert len(atoms) > 80
    assert backend.lacking_bindings(every) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"solvent_heavy_conc": {"P": 425.0, "Na": 150.0}},
        {"protein_heavy_atoms": {"Al": 1.0, "S": 6.0}},
    ],
)
def test_sodium_and_aluminium_have_an_inelastic_path(
    changes: dict[str, object],
) -> None:
    """Sodium gave NaN, aluminium a crash in RADDOSE-3D."""
    result = _run(material(**changes))
    assert result.lacking_bindings == ()
    assert math.isfinite(result.inelastic_mean_free_path)
    assert math.isfinite(result.information_coefficient)
    assert result.best_energy < 1990.0


def test_an_element_lacking_binding_energies_gives_nan() -> None:
    """The guard, for a table that lacks one. None does now, so zinc is
    made to: its L2 binding energy zeroed."""
    baseline = material()
    atoms = tuple(
        dataclasses.replace(
            atom,
            shells=dataclasses.replace(
                atom.shells,
                subshells=(*atom.shells.subshells[:2], 0.0, *atom.shells.subshells[3:]),
            ),
        )
        if atom.symbol == "ZN"
        else atom
        for atom in baseline.atoms
    )
    lacking = dataclasses.replace(baseline, atoms=atoms)
    assert backend.lacking_bindings(lacking) == ("ZN",)
    with pytest.warns(UserWarning, match="no binding energy for one or more"):
        result = _run(lacking)
    expected = _run(baseline)
    for value in (
        result.inelastic_mean_free_path,
        result.inelastic,
        result.productive,
        result.information_coefficient,
        result.best_energy,
        result.best_thickness,
    ):
        assert math.isnan(value)
    assert result.dose == expected.dose
    assert result.elastic == expected.elastic


# --- Slices ----------------------------------------------------------------


def test_a_crystal_thinner_than_a_slice_has_one() -> None:
    result = microed(experiment(dimensions=[2.0, 2.0, 0.004]))
    assert len(result.slice_doses) == 1
    assert result.slice_width == pytest.approx(4.0)
    assert result.dose == result.slice_doses[0] > 0


def test_slices_share_the_thickness() -> None:
    result = _run(material(), thickness=299.0)
    assert len(result.slice_doses) == 59
    assert result.slice_width == 299.0 / 59


# --- The engine in the models ----------------------------------------------


def test_microed_ignores_the_wedge() -> None:
    first = microed(experiment())
    data = experiment().model_dump()
    data["wedges"] = [{"start": 0.0, "end": 90.0, "exposure_time": 100.0}]
    assert microed(Experiment(**data)) == first


def test_microed_needs_an_electron_beam_and_only_microed_takes_one() -> None:
    data = experiment().model_dump()
    with pytest.raises(ValueError, match="MicroED needs an electron beam"):
        Experiment(
            **{
                **data,
                "beam": {
                    "type": "gaussian",
                    "flux": 1e12,
                    "fwhm": [2, 2],
                    "energy": 300,
                },
            }
        )
    data["crystal"]["subprogram"] = "rd3d"
    with pytest.raises(ValueError, match="an electron beam is for the MicroED"):
        Experiment(**data)


def test_each_engine_has_its_own_entry_point() -> None:
    with pytest.raises(ValueError, match=r"call pyraddose_3d\.solve\.microed"):
        simulate(experiment())
    mx = Experiment.from_yaml("tests/conformance/cases/000-baseline/config.yaml")
    with pytest.raises(ValueError, match="runs the MX calculation"):
        microed(mx)


def test_the_files_take_raddose_3ds_fixed_names(tmp_path: Path) -> None:
    written = write_microed_outputs(microed(experiment()), tmp_path)
    assert [p.name for p in written] == [
        "outputMicroED.CSV",
        "outputMicroED_slices.csv",
    ]
    assert written[0].read_text().startswith("Beam_en, Dose, Elastic,")
