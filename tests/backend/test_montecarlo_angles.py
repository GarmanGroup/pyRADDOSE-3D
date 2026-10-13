"""ELSEPA's electron angular tables, from their two data packages.

These tests need both packages, built: ``uv sync`` installs them, and
``tests/backend/data/build_elsepa_packages.py`` builds the high-energy one's
tables. Without them the tests fail rather than skip, so a missing table is
never mistaken for a passing suite.
"""

from __future__ import annotations

import hashlib
import lzma
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyraddose_3d.backend.montecarlo import angles

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(name: str) -> dict[int, tuple[str, int]]:
    text = (resources.files(name) / "tables" / "MANIFEST.tsv").read_text("ascii")
    rows = [line.split("\t") for line in text.splitlines()[2:]]
    return {int(z): (sha, int(size)) for z, sha, size in rows}


@pytest.mark.parametrize("high_energy", [False, True])
def test_every_table_is_raddose_3ds_unchanged(high_energy: bool) -> None:
    """Each decompresses to the file its manifest hashed, Z = 1 up to the last."""
    _, name = angles._PACKAGES[high_energy]
    manifest = _manifest(name)
    assert sorted(manifest) == list(range(1, angles.MAX_Z[high_energy] + 1))
    tables = resources.files(name) / "tables"
    for z, (sha, size) in manifest.items():
        data = lzma.decompress((tables / f"{z}.csv.xz").read_bytes())
        assert (hashlib.sha256(data).hexdigest(), len(data)) == (sha, size), z


def test_a_table_as_tabulated() -> None:
    """``below_20000/6.csv``: carbon from 0.05 to 20 keV, a column per degree."""
    table = angles.angular_table(6, high_energy=False)
    assert table.probabilities.shape == (39, 181)
    assert table.energies_kev[0] == 0.05
    assert table.energies_kev[-1] == 20.0
    assert list(table.probabilities[0, :3]) == [12.5629, 12.54672, 12.49835]
    assert list(table.probabilities[-1, :2]) == [23.51937, 17.01842]
    np.testing.assert_array_equal(table.angles_deg, np.arange(181.0))


def test_repeated_energies_keep_the_later_row_in_energy_order() -> None:
    """Hydrogen above 20 keV lists 166 to 300 keV twice; a TreeMap keeps one."""
    table = angles.angular_table(1, high_energy=True)
    np.testing.assert_array_equal(table.energies_kev, np.arange(21.0, 301.0))
    assert table.probabilities.shape == (280, 606)
    assert table.probabilities[0, 0] == 1.085534
    assert table.probabilities[166 - 21, 0] == 1.755447
    assert table.angles_deg[:3].tolist() == [0.0, 0.0001, 0.000125]


@pytest.mark.parametrize(("z", "high_energy"), [(0, False), (95, False), (53, True)])
def test_elements_without_a_table_are_refused(z: int, high_energy: bool) -> None:
    """RADDOSE-3D refuses a run above 20 keV with an element beyond Z = 52."""
    with pytest.raises(ValueError, match=f"none for Z = {z}"):
        angles.angular_table(z, high_energy)


def test_a_missing_package_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        angles._PACKAGES, False, ("pyraddose-3d-elsepa-nowhere", "elsepa_nowhere")
    )
    angles.angular_table.cache_clear()
    with pytest.raises(
        angles.MissingTablesError, match="package pyraddose-3d-elsepa-nowhere"
    ):
        angles.angular_table(6, high_energy=False)
    angles.angular_table.cache_clear()


def test_an_unbuilt_package_says_how_to_build_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The high-energy package is committed without its tables."""
    (tmp_path / "tables").mkdir()
    monkeypatch.setattr(angles.resources, "files", lambda name: tmp_path)
    angles.angular_table.cache_clear()
    with pytest.raises(angles.MissingTablesError, match="build_elsepa_packages"):
        angles.angular_table(6, high_energy=True)
    angles.angular_table.cache_clear()
