# pyRADDOSE-3D road-map

Where the port is, and what is left. File and line references throughout are so a reader can
check a claim rather than take it on trust.

## Where the port stands

5,300 lines across `backend/` (physics, never imports pydantic), `models/` (pydantic, YAML
only) and `solve.py` (the seam). 544 tests, 12 xfail. 26 of 32 conformance cases match the Java
within measured tolerances — on the sixteen summary numbers *and* on the dose field and
per-image series behind them; the six that do not are the five Monte Carlo escape cases (tier B
by design) and `041-ddm-leal`. `solve()` is differentiable end to end — ten tunable settings carry
gradients matching central finite differences to ~1e-10, at 0.15 s per optimiser step.

The physics is done. Everything below is about making it usable, provable and published.

## Three findings that set the priorities

1. **RADDOSE-3D is AGPL-3.0** (`LICENSE.txt`, added in a commit titled "AGPL v3.0"), and this
   is a derivative work of it. pyRADDOSE-3D declared BSD-3-Clause, which it could not. Now
   AGPL-3.0-only. See Milestone 0.
2. **`array_stats.json` was generated for all 32 cases and never compared**, so conformance was
   sixteen averaged numbers per case and the dose *field* was unchecked. Fixed in Milestone 1.
3. **The input models are missing the sample container**, which multiplies beam flux by
   `1 - attenuationFraction` on the ordinary MX path (`BeamGaussian.java:270`). Also missing:
   `MaxResolution` (feeds the RDE columns of `DWDs.csv`) and `NumCarb`.

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
  of `DWDs.csv` — `exp(-DWD/De)` — which are the one thing deferred, listed in
  `_array_stats.DEFERRED`. The comparison fails if the port starts producing a deferred
  quantity, so the deferral cannot outlive the gap.

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

# Milestone 2 — Make it usable

**Why.** Nobody outside this machine can run the port.

### No CLI — run scripts and YAML, per Leopard-EM

Follow `programs/<name>/` as Leopard-EM does it: a short **editable** `run_<name>.py` with a
marked parameters block at the top, a worked `<name>_example_config.yaml`, and a per-program
`README.md`. No `[project.scripts]`, no argparse.

That is the right call for this audience. Scientists run a handful of configured jobs, often via
SLURM; an editable script is trivially extended, and the optimisation loop in Milestone 3 is
naturally a script rather than a flag. It also matches the house style the rest of the port
already follows.

The counter-argument is worth recording rather than dismissed: RADDOSE-3D users today type
`java -jar raddose3d.jar -i input.txt`, and CCP4 may want a command-line program. If either
turns out to matter, a CLI is ~30 lines of argparse over the same `solve()` and the same
writers — so **the writers are the real work, and they are needed either way**. Defer the CLI
until something actually demands it.

Two programs to start: `programs/dose/` (run an experiment, write the outputs) and, in
Milestone 3, `programs/optimise/`.

### Output writers

The Java turns nine on by default. These four are what people actually read:

| writer | file | notes |
|---|---|---|
| `SummaryText` | `output-Summary.txt` **and stdout** | human-readable block plus a 9-bin dose histogram; this is what a user sees |
| `SummaryCSV` | `output-Summary.csv` | the sixteen numbers `WedgeSummary` already holds — nearly free |
| `FinalDoseStateCSV` | `output-DoseState.csv` | `x,y,z,dose,fluence,elastic` per in-crystal voxel; `ExposureResult` has all six |
| `DWDs` | `output-DWDs.csv` | per image; DWD and volume are already compared, the five RDE columns need `MaxResolution` |

`VoxelDose`/`VoxelFluences` can be written from `simulate(record_images=True)`, which Milestone 1
added for the comparison. `FinalDoseStateR` emits an R/rgl script; a Python port should offer a plotting
helper instead.

### Input coverage gaps

- **`MaxResolution`** on the wedge → the five RDE columns of `DWDs.csv`. The whole calculation
  is `q = {2π/maxRes, 2π, π, 2π/3, π/2}`, `De = 81.3 / q**1.7`, `RDE = exp(-DWD/De)`
  (`ExposureSummary.java:123-125`). About 20 lines; it completes the `DWDs` writer and empties
  `_array_stats.DEFERRED`. (`RDE.csv` is a different quantity — the decay model's efficiency —
  and is already compared.)
- **`Container`** → a scalar flux attenuation. `ContainerTransparent` is free;
  `ContainerElemental`/`ContainerMixture` need mass attenuation coefficients the Java scrapes
  from NIST at runtime. *Decision deferred to this milestone:* our own element tables could
  supply them offline — better engineering, but it would not match the Java numerically, so it
  is a deliberate divergence to be recorded in the corpus, not a silent one.
- **`NumCarb`** → carbohydrate residues in the composition.

### Notebooks that are also tests

Beyond the Leopard-EM pattern — their `40S_demo.ipynb` is not run in CI, and documentation that
is never executed rots quietly. Ours should be executed.

One notebook per capability, kept small enough to run in CI (a baseline solve is 0.15 s):

1. **Getting started** — YAML in, dose out, reading the summary.
2. **Crystal and beam geometry** — cuboid/sphere/cylinder, Gaussian vs tophat, collimation, with
   plots of the dose field.
3. **Composition routes** — unit cell, sequence, PDB, CIF, SAXS, and what solvent content does.
4. **Decay models** — simple, linear, B-factor, and what each does to the DWD series.
5. **Gradients and optimisation** — Milestone 3's example.

Run them in CI with `pytest --nbmake` (or jupytext-paired scripts, if diffs matter more than
convenience — worth choosing when we get there). Add a `docs/` tree at the same time; Leopard-EM
uses zensical/mkdocs-material and that is a reasonable default.

### Housekeeping while here

- **Windows in CI.** CCP4 requires all three platforms; CI runs ubuntu and macos only.
- **Python floor.** CCP4 9 ships Python 3.9, CCP4 2026 ships 3.11. Keep `>=3.11` and target
  CCP4 2026, since relicensing gates CCP4 anyway. Cost of changing our mind is one import:
  `typing.Self` in `models/_base.py:5` is the only 3.11-only construct in the tree.
- **scipy** is a dependency for `erf` alone (`pyproject.toml:48`). At ~3% of a 3–4 GB CCP4
  install the size is defensible, but one function is a thin reason for the heaviest dependency
  in the tree.

**Done when:** someone in the group can clone, edit a YAML, run a script and get the output
files they expect — and the notebooks run green in CI.

# Milestone 3 — Deliver the differentiability

**Why.** The capability exists; the result does not. This is what justifies the port over
running the Java.

- A thin objective layer over `solve()` — minimise peak dose subject to a diffraction-efficiency
  floor, or maximise total diffraction under a dose budget. Keep it thin; torch has the
  optimisers.
- `programs/optimise/` in the same run-script style.
- One worked example against a real published data-collection problem, showing gradient descent
  reaching a better strategy than a parameter scan for the same compute. This is the notebook
  and the paper figure.
- Honest scope in the docs. `TUNABLE` and `FIXED` already encode which settings carry a gradient
  and why the others do not. Both step discontinuities found so far — the integer image count
  and the voxel occupancy parity test — mean an optimiser can see its objective jump; worth a
  short note on choosing step sizes that do not straddle them.

**Done when:** a notebook demonstrates a better strategy found by gradient descent than by
scanning, with the compute budget stated.

# Milestone 4 — Completeness of the MX path

Finishes what the port already claims to cover. In value order, none urgent.

- **`DDMLeal`** (496 lines) — closes the last deterministic conformance case, `041-ddm-leal`.
- **`.obj` polyhedron reader** — the only unported crystal type. `ModelFile` and `WireframeType`
  already exist in the models; `crystal_mesh` raises for them.
- **Photoelectron and fluorescence escape** — the machinery inside `Crystal.java`
  (`addDoseAfterPE`, `addDoseAfterFL`, `escapeFactor`) that the five tier-B conformance cases
  exercise. Monte Carlo, so comparable only statistically: match the generator, then compare
  means over N seeds against the observed standard error, as the corpus already anticipates.
  Closing this takes conformance from 26/32 to 31/32.

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
2. **Build a tier-B corpus** for whichever is chosen, the way `tests/conformance` was built for
   MX: cases first, generated from the Java, statistical comparison over N seeds.
3. **Fix the Java's own coverage gap** as a by-product. Characterising these to write a corpus
   will find defects — it did for every MX module — and those are worth upstreaming whether or
   not the port proceeds.
4. **Port**, reusing `backend/_array.py`, the element tables and the geometry that already
   exist. The crystal, mesh, voxel grid and composition layers are shared; it is the transport
   physics that is new.
5. **Differentiability is explicitly out of scope** here, consistent with the decision taken
   for escape modelling: these are Monte Carlo, and a vectorised implementation cannot consume
   random draws in the Java's order.

**Known defects already logged against these**, to be fixed as part of any port:
`MicroED.getSolventFraction()` returns 0 when explicitly set; `getBeamArea()` and
`getExposure()` are stubs in `BeamGaussian`/`BeamExperimental`.

Milestones 4, 5 and 6 are not strictly ordered — 6 in particular could start at any point once
someone wants it, and does not block the others.

---

## Upstream: three small Java defects to report

Found while surveying, unrelated to the port:

- `OutputFactory.java:59-61` tests `"voxeldose"` twice — a copy-paste defect, so the
  `voxelfluences` key never matches and `-o voxelfluences:...` silently does the wrong thing.
  One-line fix.
- `RD3D.printCommandlineHelp()` advertises module names (`Summary`, `DoseStateCSV`,
  `SummaryCSV:SummaryCSV.csv`) that do not resolve to real classes.
- `MicroED.getSolventFraction()` returns 0 when explicitly set, and `getBeamArea()`/
  `getExposure()` are stubs in `BeamGaussian`/`BeamExperimental` — from the earlier audit, still
  open.

## Verification, throughout

- Every step keeps the conformance corpus bit-identical unless it deliberately changes a number,
  and a changed expectation is a reviewable diff. Baseline is 26/32, worst deviation 1.94e-3
  (`011-crystal-cylinder`, `tad`); `pytest tests/conformance` prints the per-case table.
- New output writers are checked against the Java's own files for the same case through
  `generate.py`, not against hand-written expectations.
- `solve()` with no overrides stays pure NumPy — `test_no_overrides_stays_on_numpy` and the
  architecture test enforce it, and CCP4 sizing depends on it.
- Mutation-check every new conformance comparison the way the existing ones were: perturb a
  constant, confirm the relevant cases fail, and show the new comparison catches something the
  old ones miss — as the array statistics did with a mirrored dose field, now pinned as a test.
- Notebooks run in CI, so an example that stops working fails the build rather than misleading a
  reader.
