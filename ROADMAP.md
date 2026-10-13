# pyRADDOSE-3D road-map

Where the port is, and what is left. File and line references throughout are so a reader can
check a claim rather than take it on trust.

## Where the port stands

11,000 lines across `backend/` (physics, never imports pydantic), `models/` (pydantic, YAML and
RADDOSE-3D's `.txt`), `solve.py` (the seam) and `outputs.py` (RADDOSE-3D's files). 1,258 tests,
none expected to fail. All 56 conformance cases match the Java within measured tolerances — on
the sixteen summary numbers, on the dose field and per-image series behind them, and on the
output files themselves — photoelectron and fluorescence escape included, value by value. The
MX path is complete: every input RADDOSE-3D's MX calculation accepts is ported, except two that
cannot be (Milestone 4). `solve()` is differentiable end to end apart from escape, which is
Monte Carlo — ten tunable settings carry gradients matching central finite differences to
~1e-10, at 0.15 s per optimiser step.

It reads RADDOSE-3D's input files and writes its output files, so it can stand in for the Java;
`programs/dose` runs it, `programs/optimise` chooses a beam energy by gradient, and six executed
notebooks show it. What is left is optimising several settings at once, publication, and — if
anyone wants them — the other three subprograms.

## Three findings that set the priorities

1. **RADDOSE-3D is AGPL-3.0** (`LICENSE.txt`, added in a commit titled "AGPL v3.0"), and this
   is a derivative work of it. pyRADDOSE-3D declared BSD-3-Clause, which it could not. Now
   AGPL-3.0-only. See Milestone 0.
2. **`array_stats.json` was generated for all 32 cases and never compared**, so conformance was
   sixteen averaged numbers per case and the dose *field* was unchecked. Fixed in Milestone 1.
3. **The input models were missing the sample container**, which multiplies beam flux by
   `1 - attenuationFraction` on the ordinary MX path (`BeamGaussian.java:270`), and
   `MaxResolution` and `NumCarb`. All three added in Milestone 2.

---

# Milestone 0 — Licence — *licence change done, relicensing outstanding*

**Why.** Until this is settled, keeping the repository local is the only lawful state rather
than a preference.

AGPL-3.0 is strong copyleft, and this port is a derivative by any reading: it reproduces the
algorithms line by line, vendors RADDOSE-3D's element tables, and says so itself.

**Done.** `LICENSE` is the FSF's AGPL-3.0 text, `pyproject.toml` declares `AGPL-3.0-only` with
the matching classifier, the README has a Licence section, and `NOTICE` records what is derived
from where — algorithms, element data, the residue and mesh tables, and the conformance corpus.

`-only` rather than `-or-later` is deliberate: RADDOSE-3D grants no "or any later version"
anywhere outside the licence's own boilerplate appendix, and a derivative cannot grant more
than it received.

This makes the repository *publishable*; whether and when to publish stays a separate decision.
Doing it before the first push rather than after matters because BSD-3 could not have been
walked back once out, whereas AGPL can be relaxed later if the relicensing below succeeds.

**Separately, and starting now:** open the relicensing conversation with the RADDOSE-3D
copyright holders — Gerstel, Brooks-Bartlett, Ginn, Bury, Walsh, Zeldin and others; there is no
CLA, so it needs all of them. This is on the critical path for CCP4 and nothing else, and it is
the only item measured in months.

**Why CCP4 needs it.** [CCP4's contribution route](https://www.ccp4.ac.uk/contributing-to-ccp4/)
requires the work to fit the STFC-CCP4 licence mode and a **Contributor Agreement Document
signed by the IPR holder**. CCP4 sells commercial licences, so a copyleft dependency is a
substantive obstacle, not paperwork. The route itself is human — present at Working Group 2,
contact the core team with the dependency list, WG2 and core recommend, CCP4-Exec decides. No
submission form; the published contact is `ccp4@ccp4.ac.uk`. Note Java RADDOSE-3D has **no
existing CCP4 relationship** — absent from the v9 program index, and the repository mentions
CCP4 nowhere — so this would be a first submission.

**Still outstanding:** putting the relicensing question to the contributors.

# Milestone 1 — Tighten the specification — *done*

**Why.** Cheapest large gain available, and it had to come before any new physics because it is
what will catch errors in that physics.

**What was wrong.** `generate.py` flattened *every* numeric field of each Java CSV into one list
before taking statistics, so three of the five entries were a meaningless mash — `DoseState`
reported a minimum of −10 (a coordinate) and a maximum of 7.3e6 (a fluence) — and nothing read
any of them anyway.

**What exists now.** `tests/conformance/_array_stats.py` holds one set of reducers that both
sides go through: `generate.py` feeds it the Java's CSVs, `test_arrays_match_reference` feeds
it the port's arrays. A reduction implemented twice would drift, and the drift would look like a
port error.

| entry | what is compared |
|---|---|
| `DoseState` | per-column distribution of dose, fluence and elastic yield; dose-weighted centroid and spread along each axis |
| `VoxelDose`, `VoxelFluences` | per image: sum, maximum, number of exposed voxels |
| `DWDs` | per image: angle, DWD, exposed volume |
| `RDE` | per image: fluence-weighted mean and running minimum of the decay model's efficiency |

All counts are exact across the corpus — voxels, images, and exposed voxels in every image.

**Two things the plan above had wrong, found by reading the Java writers rather than trusting
a survey of them:**

- `VoxelDose` and `VoxelFluences` are *per image*, not final, so they were not computable from
  `ExposureResult`, which only accumulates. `expose(record_images=True)` now keeps every voxel's
  dose, fluence and decay after every image as an `ImageRecord`, off by default because it costs
  a field per image. Milestone 2's writers need exactly this.
- `RDE.csv` does not need `MaxResolution`. It records the *decay model's* efficiency, which the
  port already computed, so it is compared now. `MaxResolution` feeds only the five RDE columns
  of `DWDs.csv` — `exp(-DWD/De)` — which were the one thing deferred, listed in
  `_array_stats.DEFERRED`. The comparison fails if the port starts producing a deferred
  quantity, so the deferral could not outlive the gap. Emptied in Milestone 2.

To reach the arrays at all, `solve()` became a thin wrapper over `simulate()`, which returns a
`Simulation` — the grid, which voxels are occupied, the per-wedge exposures and angles, and the
summaries. `solve()`'s behaviour and signature are unchanged.

**Tolerances**, per statistic family, recorded with their measured basis in `MANIFEST.yaml`.
Same rule as the summary: the worst disagreement with the Java across the corpus, with
headroom, justified because it is *smaller* than the port's disagreement with itself when every
rotation angle moves by one ulp. Statistics that average over the crystal agree to 5e-5 and
share the summary's 1e-4.

The rest are limited by voxels whose centres lie exactly on a crystal face. At some angles
their path length through the crystal flips between zero and the full chord on the last bit
of a cosine — measured directly: a face voxel's depth at 66° is 14.75 µm one ulp either side
and exactly zero *at* the angle. Java's `Math.cos` is not bit-reproducible, so the two
implementations land on different sides at different angles: 144 voxel-images one way and 152
the other in the worst case, a coin flip rather than a bias. How much a flip matters scales with
absorption — per-image fluence maximum agrees to 2.5e-3 for protein at 12.1 keV and 3.6e-2 for
the antimony CIF crystal. The control: the stills, which do not rotate, and the AngleP 45 case,
which has no face on a voxel plane, agree to about 1e-8 on every statistic.

**Shown to catch what the summary cannot.** Reversing the rotation direction mirrors the dose
field. In 25 of 26 cases that also changes the totals, because the half-open occupancy test
leaves the voxel set slightly lopsided. The cylinder's voxel set is genuinely symmetric about
the axis, so all sixteen summary numbers stay within tolerance — and the dose centroid flips
from +0.00498 µm to −0.00498 µm. Pinned as
`test_arrays_see_a_mirrored_dose_field_the_summary_cannot`.

One caveat on that demonstration: the centroid tolerance, 5e-3 µm, is set by the antimony case,
so the cylinder's flip of 0.01 µm is caught with only a factor of two to spare, although the
cylinder itself agrees with the Java to 3e-5 µm. Per-case tolerances would tighten this; they
were not worth the machinery yet.

# Milestone 2 — Make it usable — *done, CI unverified*

**Why.** Nobody outside this machine could run the port.

**Done when** someone in the group can clone, edit a YAML — or bring an existing `.txt` — run a
script and get the output files they expect, and the notebooks run green in CI. All of that
holds locally; the CI half waits for the repository to have a remote.

### Input: YAML, and RADDOSE-3D's `.txt` unchanged

`Experiment.from_file` reads either. The `.txt` reader follows `Inputfile.g` rather than a
reading of the manual: case-insensitive, not line-based, every keyword alias, the three comment
styles, and the lexer's habit of *skipping* a character it does not recognise, with a message,
rather than failing — a warning here. Every corpus case survives YAML → `.txt` → YAML
(all but the seed, which RADDOSE-3D takes from its command line), and eight of the thirteen
example inputs RADDOSE-3D ships are read. Of the other five, three ask for the MicroED or Monte
Carlo engines and two are malformed, and each is refused saying why.

Defaults now mirror the Java's, and what the Java requires is required. That changed the models:
a crystal's `PixelsPerMicron` defaults to ten voxels across its first dimension, no
`Collimation` means no aperture, `AngularResolution` is honoured only above ten images, and the
made-up defaults the models had carried are gone. Each default has a conformance case that
leaves the setting out of the `.txt`, so the case checks the Java's default rather than restating
it.

Added, each with a case: **`MaxResolution`**, completing `DWDs.csv`; **`NumCarb`**; and
**containers**. For the containers, NIST's mass attenuation tables are vendored rather than
fetched at run time as the Java does, extracted with the Java's own parsing so the rows are the
ones it interpolates between — so they match to 2e-6 rather than being a recorded divergence,
which our own element tables would have been. The interpolation is reproduced exactly, including
its being linear rather than log-log and taking the value *above* an absorption edge for a beam
exactly at one.

### Output: the same eight files

`write_outputs` writes what RADDOSE-3D writes by default, under the same names: `Summary.txt`,
`Summary.csv`, `DoseState.csv`, `DoseState.R`, `RDE.csv`, `DWDs.csv`, `VoxelDose.csv` and
`VoxelFluences.csv`. Two departures from the plan above: `DoseState.R` is ported rather than
replaced with a plotting helper, since "the same files" was the point and it is a fixed
template around the dose grid; and the per-image files need `record_images=True`, which the run
script sets.

Java prints numbers its own way, and Python's formatting gets the last digit wrong
surprisingly often: `String.format` rounds the shortest decimal half-up, Python rounds the
binary value half-even, so `1.005` prints `1.01` in one and `1.00` in the other.
`_java_format` reproduces `%f`, `%e`, `Double.toString` and `Float.toString`, checked against
4,356 strings JDK 25 produced (`tests/data/java_format.tsv`).

Against every case, `tests/conformance/test_output_files.py` compares `Summary.txt`,
`Summary.csv` and `DoseState.R` with the Java's line by line — text identical, every number laid
out identically, values within the manifest's tolerances — and for the two stills `Summary.txt`
is byte-identical. The other files are read back with the parser applied to the Java's and
reduced to the corpus statistics. Two tolerances joined the manifest, both measured by the usual
rule: the dose histogram, in percentage points, and `DoseState.R` per voxel.

### Running it: `programs/dose`

As planned, Leopard-EM's layout and no CLI: an editable `run_dose.py`, a commented
`dose_example_config.yaml`, and `dose_example_input.txt`, which is RADDOSE-3D's own
`MyInput.txt`, unmodified. A test checks the YAML and the `.txt` load to the same experiment.

The CLI question stays open on the same terms: RADDOSE-3D users type
`java -jar raddose3d.jar -i input.txt`, and CCP4 may want a command-line program. Now that the
reader and the writers exist, a CLI is ~30 lines of argparse over them. Defer it until something
demands it.

### Notebooks and documentation

Five notebooks in `docs/examples` — getting started, crystals and beams, composition, decay
models, gradients — each run as a test with `pytest --nbmake` and committed without outputs.
The site is mkdocs-material with mkdocs-jupyter, which executes them at build time, and
mkdocstrings for the API; pages that are already READMEs are included, not copied.

Not zensical, which Leopard-EM uses: zensical 0.0.70 does not support mkdocs-jupyter, ignores it
without a warning and publishes the notebooks as raw `.ipynb` downloads.

### Housekeeping

- **Windows** is in the CI matrix — unverified until there is a remote. The library names an
  encoding wherever it reads a file, the obvious Windows trap.
- **Python floor** unchanged at 3.11, targeting CCP4 2026.
- **scipy** unchanged: still a dependency for `erf` alone.

# Milestone 3 — Deliver the differentiability — *energy done; several settings at once open*

**Why.** The capability exists; the result does not. This is what justifies the port over
running the Java.

**Done when:** a notebook demonstrates a better strategy found by gradient descent than by
scanning, with the compute budget stated. Met for one setting, the beam energy; the case that
most needs gradients, several settings at once, is still to do.

### First, a bug that made every energy gradient wrong

Overriding `beam.energy` changed the photon energy in the beam but left the crystal's
absorption and scattering coefficients at the configured energy, so the dose simply scaled with
energy and no absorption edge appeared. The gradient tests could not see it: a finite
difference taken through the override agrees with a gradient taken through it, both describing
the same half-changed crystal. Fixed, and now guarded the way it should have been from the
start — for every tunable setting, an override must give exactly what editing the
configuration gives (`tests/test_overrides.py`). The comparison needed the configuration with
the override written in, which became `apply_overrides()`: the experiment as run, with tensors
turned back into numbers, which is also how an optimised strategy is saved.

### Choosing the energy

With the fix, diffraction efficiency behaves as the physics says: it rises with energy, because
elastic scattering falls more slowly than photoabsorption, drops at each absorption edge, and
turns over near 36 keV as Compton scattering takes over. On its own that says "as high as the
beamline goes". What makes it a real optimisation is the detector: a thin silicon sensor lets
hard photons through, so *detected* efficiency peaks — near 11.5 keV for 450 µm of silicon,
14.5 keV for 1 mm, and with the crystal for CdTe.

- **`backend/detector.py`** — the fraction a sensor absorbs, from the same McMaster
  cross-sections the crystal uses; within 3% of NIST below 10 keV, where photoabsorption is all
  of it. Not part of RADDOSE-3D, and an upper bound on detector efficiency.
- **`optimise.py`** — the thin layer. Absorption edges are where a gradient fails: the slope
  points up on both sides of the Se K edge while efficiency drops by a third, so an optimiser
  that steps over it climbs on into worse territory. The edges are known in advance, so
  `maximise_energy` splits the range at every edge of the crystal, its container and the
  sensor, and in each piece bisects on the gradient's sign — each run halves the piece, with
  no step size to tune. Adam was tried first and stalled: its memory of large early gradients
  shrank its steps to nothing once it had overshot. `EDGE_CLEARANCE` keeps a strategy off the
  model's sharp edges, which real ones are not.
- **`programs/optimise/run_optimise_energy.py`** — the run script.
- **`docs/examples/06_optimising_energy.ipynb`** — the worked example, compute stated. For
  insulin with a silicon sensor, 15 runs against a scan's 71, at fifty times the precision,
  in about a quarter of the time. For selenomethionine, 18 runs against a 1 keV scan's 36, and
  a 4% better energy, because the scan never sampled the stretch just below the Se edge.

A gradient run with torch turned out to cost about what a plain NumPy run does, so the
gradient is close to free.

### Still to do

- **Several settings at once** — energy, beam size, exposure and translation together, under a
  dose budget. That is where gradients beat scans by orders of magnitude rather than a factor
  of a few, and where Adam-style optimisers, with their trouble on edges, will need the same
  piecewise care.
- **Photoelectron escape** changes the energy answer for small crystals: escaping
  photoelectrons take dose with them, more so at higher energies. Escape is ported now
  (Milestone 4), but it is Monte Carlo and has no gradient, so the search cannot include it.
  The energy example should be re-checked by running its chosen energies with escape on.
- **A published comparison.** The energy question is a real one with a literature; setting the
  example against a published result would make it the paper figure.
- **Anomalous signal** is outside diffraction efficiency, so the energy search is for native
  data. A SAD or MAD objective would need a model of anomalous differences.

### Honest scope

- `TUNABLE` and `FIXED` encode which settings carry a gradient and why the others do not. The
  integer image count and the voxel occupancy parity test are steps an optimiser cannot see;
  the gradients notebook says so, and absorption edges are now a third, handled.
- The two backends are not bit-identical. Torch's `cos` and `sin` can differ from NumPy's in
  the last bit, which flips face voxels like any other one-ulp change: the baseline's summary
  moves by 2e-4 between them, and even a still by 1e-3 in its maximum dose — a still has face
  voxels too, and NumPy's `cos(π/4)` happens to match Java's where torch's does not. Within the
  port's measured self-sensitivity, but above tier A's 1e-4, so the conformance claim is
  NumPy's. It also means a gradient checked against a NumPy finite difference agrees only to
  ~1e-3; against a torch finite difference, to 1e-10. The energy example re-runs its answer on
  NumPy, where it agrees to 3e-4.

# Milestone 4 — Completeness of the MX path — *done*

**Why.** To finish what the port claims to cover: anything RADDOSE-3D's MX calculation accepts,
the port runs and gets the Java's answer for.

**Done when** every MX input is ported or refused for a reason that is not "not ported yet", and
every conformance case passes. Both hold: 50 of 50 cases, no xfails.

### What was ported

| feature | Java | corpus |
|---|---|---|
| Leal et al. decay model | `DDMLeal` | `041` |
| crystals from `.obj` files, and the default resolution's million-voxel cap | `ImportWireframeObj`, `getDefaultLimitedResolution` | `013`, `014` |
| pink beams (`EnergyFWHM`) | `SampleNormalEnergyDistribution`, `Crystal.expose` | `028`, `029`, `105` |
| photoelectron and fluorescence escape, and the surrounding | `Crystal.java`, `CrystalPolyhedron.java` | `100`–`108` |
| SAXS from a sequence (`AbsCoefCalc SAXSSEQ`) | `CoefCalcFromSequenceSAXS` | `038` |
| measured beam profiles (`Type ExperimentalPGM`) | `BeamExperimental`, `BeamExperimentalpgm` | `080`, `081`, `108` |

Two inputs are refused for good: `AbsCoefCalc RDV2`/`RDV3` run the original Fortran RADDOSE as
an external program and scrape its output, and `Type Experimental` cannot be built from an input
file by RADDOSE-3D either — `BeamExperimental` has no property constructor, so only
`ExperimentalPGM` works there.

### Escape is tier A, not tier B — the plan above was wrong

The plan said escape could only be compared statistically, because "a vectorised
implementation cannot consume random draws in the Java's order". It can.
`java.util.Random` is specified exactly, and `backend/java_random.py` reproduces it. What makes
the vectorised version possible is that the Java's draws are predictable in advance. Each lit
voxel, in index order, takes a fluorescence draw if it has fluorescence to release, then a fixed
number of photoelectron draws, all bounded `nextInt`s. The generator can therefore jump ahead
to produce every draw for an angle at once. The rare draw that `nextInt` rejects is replayed in
order.

So with RADDOSE-3D's seed the port gives RADDOSE-3D's escape result value by value: the nine
escape cases agree to between 3e-7 and 2e-4 on the summary, the largest on quantised
quantities. Tier B is now unused.

Escape is the one part of the MX path with no gradient. The direction of each photoelectron is
drawn at random, so `solve()` raises if escape is asked for with tensor overrides. Pink beams
stay differentiable.

### Port bugs found on the way

- **The default resolution was only half ported.** RADDOSE-3D caps the rotated crystal's bounding
  box at a million voxels, and the port never applied the cap. Only a thin plate or a needle
  reaches it.
- **Sequence files lost their chain types.** The reader dropped FASTA headers and read every
  residue as protein, so a chain marked `ISDNA` or `ISRNA` was silently wrong. A, C, G and T are
  valid amino acid codes, so nothing failed. Case `037` now covers protein, DNA and RNA chains in
  one file, and reading everything as protein fails it.
- **SAXS required a unit cell.** RADDOSE-3D defaults to a 1000 Å box of solution, and both SAXS
  routes now do too.

### The Java's quirks: reproduced, then fixed in both

Faithfulness means reproducing what RADDOSE-3D does, not what it meant to do, so the port first
reproduced each of these and marked it in its source:

- fluorescence deposited along +z only;
- the crystal's photoelectron tracks used the surrounding's draw table;
- photoelectron escape alone lost the atom's relaxation energy, except the K-shell Auger part;
- the surrounding's attenuation was clamped onto the unrotated crystal's box;
- pink beams gave the surrounding 200 times its dose, reported the per-image volume ×200, and
  kept only the last energy's fluences;
- the L-shell fluorescence of every element from Mg to Co went to voxel (0, 0, 0).

All are now fixed in RADDOSE-3D, in PRs #36–#38, not yet merged, with the port following
each fix and the corpus regenerated from it. `UPSTREAM_FIXES.md` has the detail, measurements and
status.

# Milestone 5 — Publication and CCP4

Gated on Milestone 0's relicensing, not on code. Paper on the differentiable strategy
optimisation; CCP4 submission via Working Group 2 once the licence permits.

# Milestone 6 — The other three subprograms

**Why.** RADDOSE-3D is not only the MX dose calculation. `Subprogram <name>` on the crystal
block switches the whole run to a different physics engine, and all three are unported:

| subprogram | keyword | source | what it models |
|---|---|---|---|
| Monte Carlo | `MONTECARLO`, `GOS` | `MC.java`, 4,809 lines | full electron transport, per-run, `outputMC.CSV` |
| XFEL | `XFEL` | `XFEL.java`, 4,388 lines | pulsed sources, ionisation per atom, `outputXFEL.CSV` |
| MicroED | `EMSP`, `EMED` | `MicroED.java`, 4,856 lines | electron diffraction, `outputMicroED.CSV` + slices |

Dispatch is `Crystal.expose` (lines 746–770) into `startXFEL` / `startMicroED` / `startMC`,
each constructed in `CrystalPolyhedron` from the vertices, indices and voxel coordinates. The
default is `RD3D`, the MX path already ported.

**Why this is a project rather than a task.** ~14,000 lines, **zero Java test coverage**, so
there is no specification beyond the source itself — the same argument that made the
conformance corpus the first thing built for the MX path applies here with more force. The
sequence has to be: characterise the Java first, then port.

**Sketch of the approach**, in order:

1. **Decide whether they are wanted at all.** MicroED and XFEL serve different communities from
   MX. This is a scope decision for the group, not a technical one, and it should be made
   before any code — porting 14,000 lines nobody asked for would be the single largest way to
   waste this effort.
2. **Build a corpus** for whichever is chosen, the way `tests/conformance` was built for MX:
   cases first, generated from the Java. Escape showed that Monte Carlo need not mean
   statistical comparison. If the draws can be predicted, reproduce `java.util.Random` and
   compare value by value (tier A). Fall back to statistics over N seeds (tier B) only where
   they cannot.
3. **Fix the Java's own coverage gap** as a by-product. Characterising these to write a corpus
   will find defects — it did for every MX module — and those are worth upstreaming whether or
   not the port proceeds.
4. **Port**, reusing `backend/_array.py`, the element tables and the geometry that already
   exist. The crystal, mesh, voxel grid and composition layers are shared; it is the transport
   physics that is new.
5. **Differentiability is explicitly out of scope** here, as it is for escape: a randomly drawn
   track has no gradient.

**Known defects already logged against these**, to be fixed as part of any port:
`MicroED.getSolventFraction()` returns 0 when explicitly set; `getBeamArea()` and
`getExposure()` are stubs in `BeamGaussian`/`BeamExperimental`.

Milestones 5 and 6 are not ordered: 6 could start at any point once someone wants it, and does
not block 5.

---

## Upstream: Java defects

`UPSTREAM_FIXES.md` explains each defect the port found, measures it, and records its fix and
status. All are fixed in three stacked RADDOSE-3D PRs, #36, #37 and #38; the port follows each,
with the corpus regenerated from it. Items 1 and 2 there are physics changes and need the group's
sign-off before #38 leaves draft. Item 1 changes doses materially -- photoelectron escape alone
lost 15% of a SeMet crystal's dose above the Se K edge -- while item 2, measured, moves them by
under 0.1%.

Still open:

- `MicroED.getSolventFraction()` returns 0 when explicitly set, and `getBeamArea()`/
  `getExposure()` are stubs in `BeamGaussian`/`BeamExperimental` -- Milestone 6.
- Lexer errors are not refused. A character the grammar does not know is skipped with a message
  ("no viable alternative at character 'ß'") and the run carries on -- `input-b3x14.txt`, which
  RADDOSE-3D ships, has a stray `ß`. A gap in #34, which refuses input the *parser* cannot read.
- Container runs fetch NIST's tables over the network on every run, and NIST refuses Java 8's
  requests, which then fail with a `NullPointerException`. The port vendors the tables; the Java
  should too.

## Verification, throughout

- Every step keeps the conformance corpus bit-identical unless it deliberately changes a number,
  and a changed expectation is a reviewable diff. Baseline is 56/56. The worst deviation is
  2.0e-4 (`102-escape-pe-fl`, `dose_threshold`, a quantised quantity), and 6.5e-5 for the rest;
  `pytest tests/conformance` prints the per-case table.
- New output writers are checked against the Java's own files for the same case through
  `generate.py`, not against hand-written expectations.
- `solve()` with no overrides stays pure NumPy — `test_no_overrides_stays_on_numpy` and the
  architecture test enforce it, and CCP4 sizing depends on it.
- Mutation-check every new conformance comparison the way the existing ones were: perturb a
  constant, confirm the relevant cases fail, and show the new comparison catches something the
  old ones miss — as the array statistics did with a mirrored dose field, now pinned as a test.
- Notebooks run in CI, so an example that stops working fails the build rather than misleading a
  reader.
