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

MicroED's fifteen cases (120-134) are tier A too, and simpler: the calculation
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
- **Cost.** The port takes about an hour over the 28 Monte Carlo, GOS and
  XFEL cases, so those comparisons are marked `slow`, and run with
  `uv run pytest -m slow tests/conformance/test_montecarlo_files.py`. Every
  case also runs briefly in the default suite, which checks that the port
  writes the files and printout RADDOSE-3D does. Regenerating the references
  takes the Java longer: half an hour to an hour a Monte Carlo case, and for
  the cylinder, whose 124 triangles it tests one by one, an hour at
  RADDOSE-3D #48 and ten before it. A GOS or XFEL case takes ten to thirty
  minutes with #51, which stopped the Java recomputing the composition's
  sums at each of the thousand points of a cross-section integral; before
  it, a GOS run of case 161 took nine hours on a loaded machine. The Monte
  Carlo, GOS and XFEL cases are generated from the top of RADDOSE-3D #54 to
  #62, a stack of PRs since merged: from #59 at `c1566d2`; the seven that #60
  changes or adds from #60 at `cff0a59`; 182 from #61 at `29347d6`; 187
  from #62 at `e0529dc`; and 151 and 185, whose exposed region #66
  moves, from #66 at `28b90f4`.

The Monte Carlo cases are case 140, a 20 um crystal, and 141 to 153, of 3 to
8 um, which photoelectrons escape from, so that how they are followed shows
in the dose. Each varies one thing from 141: the beam above 20 keV, iodine
and sodium, a circular top-hat, a rotation, a cylinder, a pink beam, a
surrounding, and the polarisation. 150 and 151 test the wedge, on 145's
crystal: turned back from 90 degrees to 0, offset, which must match 145, and
held still at 60 degrees in a beam narrower than the crystal, where it
absorbs 42% less than at 0, and the exposed region is the slab the beam
crosses rather than the column it would at 0. 152 and 153 test the surrounding: a cylinder in a
thin one, whose shell must be as thick all round, and a pink beam in one of
no given thickness, which the most energetic photon caps. 154 is platinum at
15 keV, whose missing L1 binding energy stopped RADDOSE-3D's run.

GOS mode has five cases: 160, case 140's crystal; 161, 141's 5 um cube,
which photoelectrons and their secondaries escape; 162, 148's surrounding,
where electrons are followed without the GOS model; and 163, selenium above
its K edge, whose vacancies relax through its L and M shells; and 164, 1 M
iodide, whose electrons the model had bound by iodine's K edge. XFEL has
eight, which test its clock: 180, a 30 fs pulse; 181, a 1 fs pulse, in
0.01 fs bins, before whose cutoff only a quarter of the dose arrives; 182, a
pink beam, its photons' energies in no order through the pulse; 183, 163's selenium in a 10 fs
pulse, whose fluorescence carries its energy away; 184, 162's salt
surrounding; 185, 151's crystal still at 60 degrees and offset 1 um, which
RADDOSE-3D's own XFEL left at 0; 186, a 0.4 fs pulse, which had no
length when it was rounded to whole femtoseconds; and 187, a Gaussian pulse
of 2 fs FWHM, its photons within two FWHMs about its peak.

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
