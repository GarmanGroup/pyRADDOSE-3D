# Conformance corpus

The port specification. Each case pairs a canonical YAML configuration with
reference output from the Java implementation, so "the Python port behaves the
same" is a testable claim rather than an aspiration.

## Layout

    MANIFEST.yaml        corpus-wide: reference version, JVM, seed, tolerances
    _cases.py            the single definition of every case
    generate.py          regenerates cases/ from the Java implementation
    test_conformance.py  the pytest side
    cases/<id>/
      config.yaml        canonical input (generated, committed, reviewable)
      case.yaml          tier, what it exercises, seed, reference version
      expected/
        summary.json     per-wedge summary from the reference
        array_stats.json statistics of the voxel-level outputs

## Why YAML is canonical

The `.txt` handed to the Java implementation is *emitted* from `config.yaml`
(`Experiment.to_raddose_txt`), not written by hand. The two inputs therefore
cannot drift apart, and a conformance failure is unambiguously a difference in
behaviour rather than a difference in input. It also avoids reimplementing the
legacy ANTLR grammar, which this port deliberately does not do.

## The two tiers

**Tier A — deterministic.** No escape modelling, so the calculation is
reproducible without matching the reference's random number stream. Compared
field by field. This is the differentiable core and where nearly all the value
is.

**Tier B — Monte Carlo.** Photoelectron or fluorescence escape is enabled.
`java.util.Random` is exactly specified and so *could* be reproduced in Python,
but that is not sufficient: a vectorised implementation does not consume random
draws in the same order as the reference's scalar loops. These cases therefore
only ever get loose, statistical agreement, and exist to catch gross error.

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
(tiering, wedge counts, emitter determinism) independently of the port.
