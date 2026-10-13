# Conformance corpus

The port specification. Each case pairs a canonical YAML configuration with
reference output from the Java implementation, so "the Python port behaves the
same" is a testable claim rather than an aspiration.

## Layout

    MANIFEST.yaml         corpus-wide: reference version and commit, JVM,
                          seed, and each tolerance with how it was measured
    _cases.py             the single definition of every case
    generate.py           regenerates cases/ from the Java implementation
    _array_stats.py       reduces voxel-level outputs to statistics
    _java_files.py        reads RADDOSE-3D's output files
    test_conformance.py   the numbers: summaries and array statistics
    test_output_files.py  the files: the port's output files against the Java's
    cases/<id>/
      config.yaml         canonical input (generated, committed, reviewable)
      case.yaml           tier, what it exercises, seed, reference version
      expected/
        summary.json      per-wedge summary from the reference
        array_stats.json  statistics of the voxel-level outputs
        Summary.txt       the reference's own files, kept whole: compared
        Summary.csv         line by line with the port's
        DoseState.R         (its timestamp blanked)
        first_lines.json  the opening line of each per-voxel file

Both sides of every comparison go through the same reduction and the same
parser: `generate.py` applies them to the Java's files, the tests to the
port's arrays and to the port's own files. A reduction implemented twice
would drift, and the drift would look exactly like a port error.

## Why YAML is canonical

The `.txt` handed to the Java implementation is *emitted* from `config.yaml`
(`Experiment.to_raddose_txt`), not written by hand. The two inputs therefore
cannot drift apart, and a conformance failure is unambiguously a difference in
behaviour rather than a difference in input.

The port also reads `.txt` files (`Experiment.from_raddose_txt`), so existing
inputs keep working. Every case survives the round trip YAML to `.txt` and
back -- all but the seed, which RADDOSE-3D takes from its command line --
which tests the reader against the emitter on 73 configurations.

## The two tiers

**Tier A — reproducible.** Compared field by field. Every case is tier A,
including the eight with photoelectron or fluorescence escape. Escape is Monte
Carlo, but `java.util.Random` is exactly specified, and the port draws from it
in the Java's order -- one fluorescence draw, then the photoelectron draws, for
each lit voxel in turn -- so with the case's seed it reproduces the Java's dose
value by value rather than only its statistics.

MicroED's fourteen cases (120-133) are tier A too, and simpler: the calculation
draws no random numbers. They compare RADDOSE-3D's two MicroED files and what
it prints, which gives the mean free paths to full precision
(`test_microed_files.py`, tolerances in `MANIFEST.yaml`).

**Tier B — statistical.** For a calculation the port cannot reproduce draw for
draw, compared loosely. No case uses it at present; the Monte Carlo and XFEL
subprograms might, whose draws depend on each electron's history.

## Regenerating

    export RADDOSE3D_DIR=/path/to/RADDOSE-3D   # must be built: `ant build`
    python tests/conformance/generate.py

Run deliberately, never in CI. The committed reference values *are* the
corpus; regenerating them is a claim that behaviour changed on purpose, so
review the diff.

Note the reference is not bit-reproducible across JVMs — the same Java code
gives 12.614139 on macOS/JDK 25, 12.614146 on Linux/JDK 8 and 12.614135 on
Linux/JDK 21. The JVM used is recorded in `MANIFEST.yaml`, and the tolerances
are set well above that noise.

## Adding a case

Add it to `_cases.py` and regenerate. Cases are one-factor-at-a-time from a
baseline, plus interaction cases chosen because they correspond to defects
actually found in the reference implementation. A new case should produce a
result distinct from every existing one — otherwise it is not exercising what
it claims to. `test_conformance.py` checks the corpus's own integrity
(seeds, wedge counts, emitter determinism) independently of the port.
