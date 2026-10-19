"""The Monte Carlo subprograms' atomic tables, as RADDOSE-3D loads them."""

from __future__ import annotations

import numpy as np
import pytest

from pyraddose_3d.backend.montecarlo import tables


def test_every_relaxing_vacancy_has_both_tables() -> None:
    """``getNumAugerShells`` names the vacancies the Java loads files for."""
    auger = tables.auger_transitions()
    fluorescence = tables.fluorescence_transitions()
    for z in tables.AUGER_ELEMENTS:
        for shell in range(tables.auger_shells(z) + 1):
            assert (z, shell) in auger, (z, shell)
            assert (z, shell) in fluorescence, (z, shell)
    assert len(auger) == 92


def test_carbon_k_shell_as_tabulated() -> None:
    """``auger_linewidths/6-0.csv``: KL1L1 first, linewidth 0.061 eV."""
    t = tables.auger_transitions()[(6, 0)]
    assert t.linewidth_ev[0] == 0.061
    assert t.probability[0] == 0.413609
    assert t.energy_kev[0] == 0.25589
    assert t.exit_shell is not None
    assert (t.exit_shell[0], t.drop_shell[0]) == (1, 1)


@pytest.mark.parametrize("kind", ["auger", "fluorescence"])
def test_cumulative_probabilities_are_the_javas(kind: str) -> None:
    """A running sum scaled by ``1 / sum``, zero in the unused slots."""
    loaded = (
        tables.auger_transitions()
        if kind == "auger"
        else tables.fluorescence_transitions()
    )
    for t in loaded.values():
        assert t.cumulative.shape == (65,)
        running = 0.0
        for i in range(t.rows):
            running += t.probability[i]
            assert t.cumulative[i] == running * (1 / t.total)
        assert running == t.total
        assert not t.cumulative[t.rows :].any()
        assert not t.probability[t.rows :].any()
        assert t.cumulative[t.rows - 1] == pytest.approx(1.0, abs=1e-15)


def test_fluorescence_has_no_exit_shell() -> None:
    assert tables.fluorescence_transitions()[(16, 0)].exit_shell is None


def test_overvoltage_tables_are_indexed_as_the_java() -> None:
    """``lowOvervoltages[Z][shell]`` and ``highOvervoltages[Z][shell]``."""
    low = tables.overvoltage_low()
    high = tables.overvoltage_high()
    assert low.shape == (95, 9, 6)
    assert high.shape == (95, 9, 7)
    # Hydrogen's K shell: 13.61 eV.
    assert low[1, 0, 0] == 13.61
    assert high[1, 0, 0] == 1.513


@pytest.mark.parametrize("kind", ["auger", "fluorescence"])
def test_packed_tables_hold_the_dicts_entries(kind: str) -> None:
    """Indexed by (Z, vacancy, slot), the packed arrays read as the tables."""
    loaded, packed = (
        (tables.auger_transitions(), tables.packed_auger())
        if kind == "auger"
        else (tables.fluorescence_transitions(), tables.packed_fluorescence())
    )
    assert packed.drop_shell.dtype == packed.exit_shell.dtype == np.int64
    for (z, vacancy), t in loaded.items():
        for name in ("cumulative", "energy_kev", "linewidth_ev", "drop_shell"):
            assert np.array_equal(getattr(packed, name)[z, vacancy], getattr(t, name))
        if t.exit_shell is not None:
            assert np.array_equal(packed.exit_shell[z, vacancy], t.exit_shell)
    assert not packed.cumulative.flags.writeable
