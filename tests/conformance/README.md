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
    _tier_b.py            tier B's statistics, and what it compares
    test_conformance.py   the numbers: summaries and array statistics
    test_output_files.py  the files: the port's output files against the Java's
    test_microed_files.py MicroED's files and report
    test_montecarlo_files.py  tier B: the Monte Carlo and XFEL subprograms
    test_tier_b_statistics.py the statistics, checked by simulation
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
        runs.json         tier B: each run's every written and printed value

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
which tests the reader against the emitter on 76 configurations.

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

**Tier B — statistical.** The Monte Carlo and XFEL subprograms follow each
electron with random numbers whose order depends on its history, so the port
cannot reproduce them draw for draw, and is not written to: it follows many
electrons at once. Each case runs 20 times on each side (`Runs 20`), and the
means of five quantities per subprogram -- the dose to the whole crystal and
to its exposed region, RADDOSE-3D's own dose estimate, the elastic
scattering and the ionisations per atom -- are compared with Welch's t-test
(`_tier_b.py`, `test_montecarlo_files.py`).

- **False alarms.** Across the corpus the chance of any is 1e-3, by
  Bonferroni's correction. The seeds are fixed, so a false alarm is not a
  flaky test: it can only appear when a change alters how random numbers are
  used.
- **Power.** `MANIFEST.yaml` gives, per case and quantity, the bias its test
  catches four times in five: about 1.9 times the run-to-run spread recorded
  in `case.yaml`. Each case's photons and energy are chosen for that spread,
  which comes from how many photons are absorbed.
- **Calibration.** Before the port is compared, the reference is compared
  with itself: its first ten runs against its last ten must pass, and must
  mostly fail once one half is shifted by the claimed detectable bias.
  `test_tier_b_statistics.py` checks the false-alarm rate and the power by
  simulation.
- **Cost.** The port takes about 25 minutes over the Monte Carlo cases, so
  those comparisons are marked `slow`, and run with
  `uv run pytest -m slow tests/conformance/test_montecarlo_files.py`. Every
  case also runs briefly in the default suite, which checks that the port
  writes the files and printout RADDOSE-3D does. Regenerating the references
  takes the Java longer: an hour for GOS and XFEL each, and ten for the
  cylinder, whose 124 triangles it tests one by one.

The Monte Carlo cases are case 140, a 20 um crystal, and 141 to 149, of 3 to
8 um, which photoelectrons escape from, so that how they are followed shows
in the dose. Each varies one thing from 141: the beam above 20 keV, iodine
and sodium, a circular top-hat, a rotation, a cylinder, a pink beam, a
surrounding, and the polarisation.

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
