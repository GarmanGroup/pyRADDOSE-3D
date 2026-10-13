"""RADDOSE-3D's MicroED engine: the parts the conformance cases cannot see.

The cases in ``tests/conformance/test_microed_files.py`` compare whole runs with
the Java. These pin the pieces underneath -- the ELSEPA grid, the trapezium
rule's point count, the refusals -- and the engine's place in the models.
"""

from __future__ import annotations

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


def _java_distant_integral(energy: float, uk: float, wk: float, shell: float) -> float:
    """``integrateDist`` and ``getpDisW``, transcribed loop for loop."""
    wak = backend._wak(energy, wk, uk)
    w_dis = ((3 * wak - 2 * uk) / 1000) * KEV_TO_JOULES
    step = w_dis / 1000
    shell_uk = shell * 1000
    shell_wak = backend._wak(energy, wk, shell_uk)
    shell_w_dis = ((3 * shell_wak - 2 * shell_uk) / 1000) * KEV_TO_JOULES
    shell_uk_j = (shell_uk / 1000) * KEV_TO_JOULES
    w = shell * KEV_TO_JOULES
    total = previous = 0.0
    count = 0
    while w <= w_dis:
        density = 0.0
        if shell_uk_j <= w < shell_w_dis:
            density = (2 / math.pow(shell_w_dis - shell_uk_j, 2)) * (shell_w_dis - w)
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
            uk = atom.subshell_binding(index) * 1000
            shell = atom.shell_binding(index)
            assert backend._distant_integral(
                energy, uk, wk, shell
            ) == _java_distant_integral(energy, uk, wk, shell), (atom.symbol, index)


def test_the_baseline_matches_the_javas_printed_mean_free_paths() -> None:
    """``Elastic Lambda`` and ``GOS Inelastic Lambda`` from a Java run.

    The elastic path agrees to every digit. The inelastic one differs by
    3e-9: one zinc subshell's integral takes one point fewer than the Java's,
    whose resonance energy is 2 ulp lower because it sums elements in a
    ``HashSet``'s order.
    """
    m = material()
    assert backend.elastic_mean_free_path(m, 300.0) == 516.2419669247315
    assert backend.inelastic_mean_free_path(m, 300.0) == pytest.approx(
        234.1956088809326, rel=1e-8
    )


# --- What RADDOSE-3D gets wrong --------------------------------------------


def test_sodium_gives_a_nan_inelastic_path_with_a_warning() -> None:
    """Its outermost subshell's binding energy is zero in the tables."""
    m = material(solvent_heavy_conc={"P": 425.0, "Na": 150.0})
    with pytest.warns(UserWarning, match="NA has a zero binding energy"):
        result = backend.microed(
            m,
            dimensions_nm=(2000.0, 2000.0, 300.0),
            aperture=(4.0, 4.0),
            circular=False,
            energy=300.0,
            exposure=1.0,
        )
    assert math.isnan(result.inelastic_mean_free_path)
    assert math.isnan(result.information_coefficient)
    assert result.dose == pytest.approx(3.678951, rel=1e-6)
    # The grid searches never stop on NaN, as in RADDOSE-3D.
    assert (result.best_energy, result.best_thickness) == (1990.0, 1995.0)


def test_aluminium_is_refused() -> None:
    """Its M1 binding energy is blank, and RADDOSE-3D crashes reading it."""
    with pytest.raises(ValueError, match="crashes reading it"):
        material(protein_heavy_atoms={"Al": 1.0})


def test_a_crystal_thinner_than_a_slice_is_refused() -> None:
    with pytest.raises(ValueError, match="thinner than RADDOSE-3D's 5 nm slice"):
        microed(experiment(dimensions=[2.0, 2.0, 0.004]))


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
