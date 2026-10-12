"""Extract the BEST intensity table from RADDOSE-3D's DDMLeal.java.

``DDMLeal`` integrates over 300 rows of average intensity against
resolution, from the BEST program (Popov and Bourenkov 2003), held as a
literal Java array. This copies the rows verbatim -- the same digits the
Java compiles -- into ``src/pyraddose_3d/data/best_intensity.tsv``.

Usage
-----
::

    python tests/backend/data/build_best_table.py /path/to/RADDOSE-3D \\
        src/pyraddose_3d/data/best_intensity.tsv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROW = re.compile(r"\{\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\}")

HEADER = """\
# Average intensity against resolution, from the BEST program (Popov, A. N.
# and Bourenkov, G. P. (2003). Acta Cryst. D59, 1145-1153), as RADDOSE-3D's
# DDMLeal.java holds it. h2 is 1/d^2 in inverse square Angstroms; J is the
# average intensity. Copied verbatim by tests/backend/data/build_best_table.py.
h2\tJ
"""


def main() -> None:
    """Read DDMLeal.java and write the table."""
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    java = Path(sys.argv[1]) / "src" / "se" / "raddo" / "raddose3D" / "DDMLeal.java"
    text = java.read_text(encoding="utf-8")
    block = text[text.index("BEST_DATA = new double[][]") : text.index("};")]
    rows = ROW.findall(block)
    Path(sys.argv[2]).write_text(
        HEADER + "".join(f"{h2}\t{j}\n" for h2, j in rows), encoding="ascii"
    )
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
