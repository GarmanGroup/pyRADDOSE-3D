"""Splits RADDOSE-3D's ``MuCalcConstants.txt`` into named, topical tables.

Run once to produce the files in ``src/pyraddose_3d/data/``. Kept in the
repository so the derivation is auditable rather than folklore, in the same
spirit as ``DumpMeshes.java`` next door.

The input is 87 elements by 80 tab-separated columns with no header, no
column names in the file, and nothing to stop a row being the wrong length
(gadolinium's had 82 fields). Seventy-three of the columns are named only in
``ElementDatabase.DatabaseFields``, a Java enum in a different repository;
the other seven are named nowhere at all.

Usage
-----
::

    python tests/backend/data/build_element_tables.py \\
        ~/git/RADDOSE-3D/constants/MuCalcConstants.txt \\
        src/pyraddose_3d/data

Number formatting
-----------------
Values are written with :func:`repr`, which is the shortest string that
round-trips to the same double. ``1.434560E+01`` becomes ``14.3456`` and
reads back as the identical float. Blank fields stay blank: the original
distinguishes "absent" from "zero", and the loader relies on it -- a missing
M edge is ``None`` while a missing M coefficient is ``0.0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Column index of the atomic number and the chemical symbol, which every
#: table repeats so that each one can be read on its own.
_Z = 0
_SYMBOL = 1

#: Output file -> (column heading, index in MuCalcConstants.txt).
#:
#: Names come from ``ElementDatabase.DatabaseFields`` where it has one.
#: Columns 21, 22, 32, 33, 34 and 35 are named nowhere in the Java and were
#: identified from the values: 21 and 22 are the density and the atomic mass,
#: and 32 to 35 are the Ka, Kb, La and Lb fluorescence line energies, which
#: agree with the X-ray Data Booklet to the tabulated precision (iron Ka
#: 6.403 against 6.404, gold La 9.711 against 9.713). Column 38 is named
#: nowhere and read by nothing; see ``edge_ratios.tsv``.
TABLES: dict[str, list[tuple[str, int]]] = {
    "elements.tsv": [
        ("atomic_weight", 23),
        ("atomic_mass_1e24_g", 22),
        ("density_g_cm3", 21),
    ],
    "absorption_edges.tsv": [
        ("k_kev", 2),
        ("l1_kev", 3),
        ("l2_kev", 36),
        ("l3_kev", 37),
        ("m1_kev", 4),
        ("m2_kev", 49),
        ("m3_kev", 50),
        ("m4_kev", 51),
        ("m5_kev", 52),
    ],
    "fluorescence.tsv": [
        ("line_k_alpha_kev", 32),
        ("line_k_beta_kev", 33),
        ("line_l_alpha_kev", 34),
        ("line_l_beta_kev", 35),
        ("average_k_kev", 47),
        ("average_l_kev", 48),
        ("yield_k", 39),
        ("yield_l1", 40),
        ("yield_l2", 41),
        ("yield_l3", 42),
        ("yield_m1", 75),
        ("yield_m2", 76),
        ("yield_m3", 77),
        ("yield_m4", 78),
        ("yield_m5", 79),
    ],
    "edge_ratios.tsv": [
        ("k", 43),
        ("l1", 44),
        ("l2", 45),
        ("l3", 46),
        ("m1", 53),
        ("m2", 54),
        ("m3", 55),
        ("m4", 56),
        ("m5", 57),
        ("unknown_col38", 38),
    ],
    "electron_ionisation.tsv": [
        ("e_min_low_kev", 58),
        ("e_max_low_kev", 59),
        ("b_k_low", 60),
        ("c_k_low", 61),
        ("e_min_high_kev", 62),
        ("b_k_high", 63),
        ("c_k_high", 64),
        ("mean_ionisation_potential", 71),
        ("mean_ionisation_potential_l1", 72),
        ("mean_ionisation_potential_m1", 73),
        ("mean_ionisation_potential_n1", 74),
    ],
    "elsepa_elastic.tsv": [
        ("ev_50", 65),
        ("ev_100", 66),
        ("ev_150", 67),
        ("ev_200", 68),
        ("ev_250", 69),
        ("ev_300", 70),
    ],
}

#: ``cross_section_coefficients.tsv`` is written long rather than wide: six
#: four-term polynomials per element would be twenty-four columns, and one
#: row per element per shell is both narrower and closer to how the code uses
#: them. Shell -> index of the first of the four coefficients.
POLYNOMIALS: dict[str, int] = {
    "K": 5,
    "L": 9,
    "M": 13,
    "N": 17,
    "coherent": 24,
    "incoherent": 28,
}

#: Prefixed to each output file, so the file says what it is without anyone
#: having to find this script.
HEADERS: dict[str, str] = {
    "elements.tsv": """\
# Identity and bulk properties of the 87 elements RADDOSE-3D knows about.
#
# atomic_weight         unified atomic mass units
# atomic_mass_1e24_g    the same mass in units of 1e-24 g, as tabulated
# density_g_cm3         density of the element in its standard state
""",
    "absorption_edges.tsv": """\
# X-ray absorption edge energies in keV.
#
# Blank means the element has no such edge, which is not the same as zero:
# the loader reads a blank M edge as absent and a blank coefficient as zero.
# The 15 lightest elements have no M edge.
""",
    "cross_section_coefficients.tsv": """\
# Pathikrit Bandyopadhyay's parameterisation of the McMaster tables.
#
# Each row is four coefficients of a cubic in log(E), E in keV. The
# cross-section in Barns/Atom is
#
#     exp(c0 + c1*log(E) + c2*log(E)^2 + c3*log(E)^3)
#
# One row per element per shell. K, L, M and N are photoelectric absorption
# above the corresponding edge; coherent and incoherent are Rayleigh and
# Compton scattering. A shell the element has no parameterisation for is
# absent from this table rather than present and zero.
""",
    "fluorescence.tsv": """\
# Fluorescence line energies in keV and fluorescence yields.
#
# line_*        the Ka, Kb, La and Lb characteristic line energies
# average_*     the shell's weighted mean emission energy
# yield_*       probability that a vacancy in that shell fills radiatively
#
# Used by the Monte Carlo fluorescence-escape paths, which are not ported.
""",
    "edge_ratios.tsv": """\
# Absorption edge jump ratios: the factor by which the cross-section rises
# on crossing each edge.
#
# unknown_col38 is column 38 of MuCalcConstants.txt. It is named nowhere in
# ElementDatabase.DatabaseFields and read by no code path in RADDOSE-3D. It
# is zero below Z = 30 and then falls monotonically -- Zn 5.684, Se 4.587,
# Mo 3.675, Au 2.439, U 2.292. That shape is suggestive of an L edge jump
# ratio, but it is a guess, so the column keeps a neutral name and its
# values are carried through untouched.
""",
    "electron_ionisation.tsv": """\
# Electron-impact K-shell ionisation, and mean ionisation potentials.
#
# e_min_low/e_max_low/e_min_high   validity ranges in keV
# b_k_*, c_k_*                     fitted constants for each range
# mean_ionisation_potential*       overall, then the L1, M1 and N1 shells
#
# Used by the Monte Carlo electron-transport paths, which are not ported.
""",
    "elsepa_elastic.tsv": """\
# ELSEPA elastic scattering cross-sections, by electron energy in eV.
#
# Zero for most elements: the table was populated only where the Monte Carlo
# paths needed it.
#
# Used by the Monte Carlo paths, which are not ported.
""",
}


def _format(field: str) -> str:
    """Render one value, preserving a blank and round-tripping a number.

    Parameters
    ----------
    field : str
        The raw field from ``MuCalcConstants.txt``.

    Returns
    -------
    str
        The empty string for a blank, else the shortest string that parses
        back to the identical double.
    """
    stripped = field.strip()
    return "" if not stripped else repr(float(stripped))


def read_source(path: Path) -> list[list[str]]:
    """Read ``MuCalcConstants.txt`` into rows of exactly 80 fields.

    Parameters
    ----------
    path : Path
        The Java's constants file.

    Returns
    -------
    list[list[str]]
        One list of 80 fields per element, in file order.

    Raises
    ------
    ValueError
        If a row has fewer than 80 fields. Rows with more are accepted only
        when the surplus is blank -- gadolinium's row has two stray trailing
        tabs, which is exactly the kind of thing a header would have caught.
    """
    rows = []
    for number, line in enumerate(path.read_text("ascii").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 80 or any(f.strip() for f in fields[80:]):
            raise ValueError(
                f"{path}:{number} has {len(fields)} fields, expected 80, and "
                f"the surplus is not blank"
            )
        rows.append(fields[:80])
    return rows


def write_tables(rows: list[list[str]], out_dir: Path) -> dict[str, int]:
    """Write every topical table.

    Parameters
    ----------
    rows : list[list[str]]
        Output of :func:`read_source`.
    out_dir : Path
        Directory to write into.

    Returns
    -------
    dict[str, int]
        File name to number of data rows written.
    """
    written = {}

    for name, columns in TABLES.items():
        lines = [HEADERS[name], "\t".join(["z", "symbol", *(c for c, _ in columns)])]
        for row in rows:
            lines.append(
                "\t".join(
                    [
                        row[_Z],
                        row[_SYMBOL],
                        *(_format(row[index]) for _, index in columns),
                    ]
                )
            )
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="ascii")
        written[name] = len(rows)

    name = "cross_section_coefficients.tsv"
    lines = [HEADERS[name], "\t".join(["z", "symbol", "shell", "c0", "c1", "c2", "c3"])]
    count = 0
    for row in rows:
        for shell, first in POLYNOMIALS.items():
            raw = row[first : first + 4]
            # A shell with no parameterisation is left out rather than
            # written as four blanks; the loader supplies the zeros the Java
            # would have, and says so.
            if not any(f.strip() for f in raw):
                continue
            lines.append(
                "\t".join([row[_Z], row[_SYMBOL], shell, *(_format(f) for f in raw)])
            )
            count += 1
    (out_dir / name).write_text("\n".join(lines) + "\n", encoding="ascii")
    written[name] = count

    return written


def main() -> None:
    """Convert the file named on the command line."""
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)

    source, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    rows = read_source(source)
    for name, count in write_tables(rows, out_dir).items():
        print(f"  {name:36s} {count:4d} rows")
    print(f"\n{len(rows)} elements from {source}")


if __name__ == "__main__":
    main()
