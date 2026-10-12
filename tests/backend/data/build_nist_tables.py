"""Vendor NIST's X-ray mass attenuation tables the way RADDOSE-3D reads them.

RADDOSE-3D's sample containers fetch Hubbell and Seltzer's tables from
physics.nist.gov at run time -- one page per element for ``ContainerElemental``
and one per named compound for ``ContainerMixture``. This downloads the same
pages once, and extracts each with the *same logic as the Java class that
reads it*, so that the stored rows are by construction what the Java
interpolates from:

``ContainerElemental`` (``ElemTab/zNN.html``)
    Reads the lines of the ``<PRE>`` block that contain scientific notation,
    splits each on whitespace, takes the first token containing "E" as the
    energy in MeV and the next as mu/rho.
``ContainerMixture`` (``ComTab/<name>.html``)
    Collects every ``<TD>`` cell holding a number in scientific notation, in
    page order, and reads them in threes: energy in MeV, mu/rho, mu_en/rho.

Row order is preserved, including the repeated energy at each absorption
edge, because the Java's walk through the table depends on it.

The data are a work of the US government and in the public domain:
J. H. Hubbell and S. M. Seltzer, *Tables of X-Ray Mass Attenuation
Coefficients and Mass Energy-Absorption Coefficients*, NIST Standard
Reference Database 126, doi:10.18434/T4D01F.

Usage
-----
::

    python tests/backend/data/build_nist_tables.py src/pyraddose_3d/data
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://physics.nist.gov/PhysRefData/XrayMassCoef"

#: The 48 compounds and mixtures NIST tabulates, as named in their URLs.
COMPOUNDS = (
    "a150 adipose air alanine b100 bakelite blood bone brain breast c552 calcium "
    "ceric cesium concrete concreteba eye fluoride fricke gadolinium gafchromic "
    "gallium glass kodak lithium lithiumflu lung magnesium mercuric muscle mylar "
    "nylonfilm ovary photoemul pmma polyethylene polystyrene polyvinyl pyrex "
    "teflon telluride temethane tepropane testis tissue tissue4 vinyl water"
).split()

#: ContainerMixture's patterns, verbatim.
_MIXTURE_CELL = re.compile(r"<TD>([0-9]\.[0-9]+E[+-][0-9]+)</TD>", re.IGNORECASE)
#: ContainerElemental's patterns, verbatim.
_PRE_OPEN = re.compile(r"<PRE>")
_PRE_CLOSE = re.compile(r"</PRE>")
_SCIENTIFIC = re.compile(r"[0-9]E[-+][0-9]")


def fetch(url: str) -> str:
    """Download one page, politely."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("latin-1")
    time.sleep(0.2)
    return text


def elemental_rows(html: str) -> list[tuple[str, str]]:
    """``ContainerElemental.extractMassAttenuationCoef``'s parse, minus the walk.

    The Java checks for ``<PRE>`` before ``</PRE>`` on each line and acts on
    the first that matches, so a line holding both opens the block.
    """
    rows = []
    reading = False
    for line in html.splitlines():
        if _PRE_OPEN.search(line):
            reading = True
        elif _PRE_CLOSE.search(line):
            reading = False
        if reading and _SCIENTIFIC.search(line):
            tokens = re.split(r"\s+", line)
            energy_at = next(i for i, token in enumerate(tokens) if "E" in token)
            rows.append((tokens[energy_at], tokens[energy_at + 1]))
    return rows


def mixture_rows(html: str) -> list[tuple[str, str]]:
    """``ContainerMixture.extractMassAttenuationCoef``'s parse, minus the walk."""
    cells = _MIXTURE_CELL.findall(html)
    return [(cells[i], cells[i + 1]) for i in range(0, len(cells) - 2, 3)]


HEADER = """\
# NIST X-ray mass attenuation coefficients, as RADDOSE-3D reads them.
#
# J. H. Hubbell and S. M. Seltzer, Tables of X-Ray Mass Attenuation
# Coefficients and Mass Energy-Absorption Coefficients, NIST Standard
# Reference Database 126, doi:10.18434/T4D01F. Public domain.
#
# {what}
#
# energy_mev is the photon energy in MeV and mu_rho the mass attenuation
# coefficient in cm^2/g. Rows are in the page's order; an absorption edge
# appears as two rows at the same energy, below and then above it, and the
# order matters to how RADDOSE-3D interpolates.
#
# Produced by tests/backend/data/build_nist_tables.py.
"""


def main() -> None:
    """Download every page and write the two tables."""
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    out = Path(sys.argv[1])

    lines = [
        HEADER.format(
            what="One table per element, from ElemTab/zNN.html, extracted as "
            "ContainerElemental extracts it."
        ),
        "z\tenergy_mev\tmu_rho",
    ]
    for z in range(1, 93):
        rows = elemental_rows(fetch(f"{BASE}/ElemTab/z{z:02d}.html"))
        lines.extend(f"{z}\t{energy}\t{mu}" for energy, mu in rows)
        print(f"  element {z:2d}: {len(rows)} rows")
    (out / "nist_elements.tsv").write_text("\n".join(lines) + "\n", "ascii")

    lines = [
        HEADER.format(
            what="One table per compound or mixture, from ComTab/<name>.html, "
            "extracted as ContainerMixture extracts it."
        ),
        "material\tenergy_mev\tmu_rho",
    ]
    for name in COMPOUNDS:
        rows = mixture_rows(fetch(f"{BASE}/ComTab/{name}.html"))
        lines.extend(f"{name}\t{energy}\t{mu}" for energy, mu in rows)
        print(f"  {name:12s}: {len(rows)} rows")
    (out / "nist_mixtures.tsv").write_text("\n".join(lines) + "\n", "ascii")


if __name__ == "__main__":
    main()
