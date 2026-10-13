# pyRADDOSE-3D road-map

Where the port is, what is left, and how it stays small. Stock taken on 2026-10-13. File and
line references are there so a reader can check a claim rather than take it on trust.

What was done to get here, and why, is in `HISTORY.md`. The defects the port found in
RADDOSE-3D, and their fixes, are in `UPSTREAM_FIXES.md`.

## Where the port stands

**It does what RADDOSE-3D's MX calculation and its MicroED engine do, value for value.**

- 12,700 lines of Python in `src/` and 10,600 in `tests/`. 1,422 tests, none expected to fail.
- Every input RADDOSE-3D's MX calculation accepts is ported, except two that cannot be:
  `AbsCoefCalc RDV2`/`RDV3` run the original Fortran RADDOSE as an external program, and
  `Type Experimental` cannot be built from an input file by RADDOSE-3D either.
- 73 conformance cases, generated from RADDOSE-3D, match it within measured tolerances. They
  cover the sixteen summary numbers, the dose field and per-image series behind them, and the
  output files themselves. Photoelectron and fluorescence escape are included, value by value.
- **MicroED** (`Subprogram EMED`) is ported: the dose to a thin crystal in an electron beam,
  slice by slice, the scattering counts, and the grid search for the best voltage and
  thickness. Its fourteen cases agree with RADDOSE-3D to the last printed digit or two, now that
  RADDOSE-3D has its MicroED fixes (#40). It needs no new dependency and 0.37 MB of
  data.
- It reads RADDOSE-3D's `.txt` input files as well as YAML, and writes the same eight output
  files RADDOSE-3D writes.
- `solve()` is differentiable end to end, apart from escape, which is Monte Carlo. Ten tunable
  settings carry gradients. `programs/optimise` chooses a beam energy by gradient.
- Two run scripts, `programs/dose` and `programs/optimise`, and six notebooks, each run as a
  test.

**What it is not yet**, and the milestone that changes it:

- **Light.** Two-thirds of the install is scipy, used for one function (Milestone 5).
- **Tested beyond one Mac.** The repository has never been pushed, so its CI has never run
  (Milestone 5).
- **Able to run every RADDOSE-3D input.** It cannot download a PDB entry by its code, still reads
  three file formats by hand, and lacks RADDOSE-3D's Monte Carlo and XFEL subprograms
  (Milestones 6 and 7).
- **Optimising several settings at once**, which is the case that most needs gradients
  (Milestone 8).
- **In CCP4.** That needs the relicensing, and answers from CCP4 (Milestone 9).

## How heavy it is

Measured in a fresh environment holding only the runtime dependencies (Python 3.12, macOS on
Apple silicon).

**Install: 107 MB, two-thirds of it scipy.**

| package | installed | used for |
|---|---|---|
| scipy | 72 MB | `erf` and `erfinv`, nothing else |
| numpy | 22 MB | everything |
| pydantic and pydantic-core | 6.5 MB | the input models |
| gemmi | 6.1 MB | PDB, mmCIF, CIF and sequence files |
| pyyaml | 0.6 MB | YAML input |
| pyRADDOSE-3D itself | 1.1 MB | 670 KB of it data; the wheel is 363 KB |
| torch, optional | 552 MB | gradients only |

RADDOSE-3D's release jar is 53 MB, and needs a JVM.

**Start-up.** Importing the solver takes about 130 ms warm. 55 ms of that is
`importlib.metadata`, looking up the package's version number. The first run that needs `erf`
or `erfinv` imports scipy's special functions, about 200 ms more, and the input models add
about 110 ms. On a fresh install, the first import of scipy compiled its bytecode and took 14
seconds, once.

**Running.** The port against RADDOSE-3D on the same inputs. Java times are wall-clock and
include starting the JVM; the port's are warm and exclude its start-up.

| case | RADDOSE-3D | port | RADDOSE-3D memory | port memory |
|---|---|---|---|---|
| baseline: protein, 90° wedge | 0.55 s | 0.24 s | 114 MB | 70 MB |
| pink beam | 3.5 s | 0.41 s | 311 MB | 73 MB |
| photoelectron escape | 1.3 s | 0.24 s | 557 MB | 205 MB |
| escape with a surrounding | 7.8 s | 7.3 s | 563 MB | 314 MB |
| escape with a pink beam | 29.8 s | 11.0 s | 630 MB | 314 MB at most |

## Keeping it light

Every addition still to come has a cost in weight, so the rules are set before the additions:

1. **A runtime dependency has to earn its place.** It is either something nothing else can
   do, or a library CCP4 asks for, and it carries a reason in `pyproject.toml`. A test will pin
   the list, so adding one is a visible, reviewed change rather than a line slipped in.
2. **Heavy or rarely needed libraries import lazily**, inside the function that needs them, as
   torch and scipy do now. `import pyraddose_3d.solve` must not import torch, scipy or
   pydantic; a test will check it in a fresh interpreter.
3. **The core package's data stays under 1 MB**, also pinned by a test. Bulk data goes in a
   separate, optional package.
4. **Optional capabilities are extras**: torch for gradients now, and any subprogram's data
   later.

### Milestone 5's changes: about 70% off the install, 40% off start-up

| change | saves | notes |
|---|---|---|
| Replace scipy with the standard library | 72 MB, 67% of the install, and ~200 ms | `math.erf` agrees with `scipy.special.erf` to 3e-16, and `statistics.NormalDist().inv_cdf` with `sqrt(2) * erfinv(2p - 1)` to 8e-16, on the pink-beam grid. Torch keeps `torch.erf` for gradients. The corpus must stay unchanged. |
| Write the version at build time | 55 ms, about 40% of the import | `hatch-vcs` can write `_version.py`, so `__init__` stops calling `importlib.metadata`. |
| Move `types-PyYAML` to the dev group | a stub package | It is for mypy, not for running. |
| Fix the description | — | `pyproject.toml` still says "pyTorch port"; torch is optional. |
| The guard tests above | — | The dependency list, lazy imports and the data budget. |

### What the work still to come would add

| change | adds | notes |
|---|---|---|
| PGM beam images to `netpbmfile` | 96 KB | Pure Python. Imported lazily. |
| `.obj` meshes to `trimesh` | 4.3 MB | NumPy only. `meshio` is smaller itself, but pulls in `rich` and `pygments`, about 9 MB in all. Imported lazily. |
| Vendored tables to `csv` or NumPy | nothing | Both are already here. |
| `.txt` reader to a Lark grammar, if CCP4 asks | 420 KB | Pure Python, no dependencies. |
| Downloading PDB entries by code, as the Java can | nothing | `gemmi.fetch` and the standard library. |
| A command-line program, if CCP4 wants one | nothing | `argparse`. |
| MicroED, done | 0.37 MB of data | `fullelsepa.csv`, 0.12 MB if gzipped. No dependency. |
| Monte Carlo and XFEL | 7.5 MB of data, 3 MB compressed; 95 MB more for hard X-rays | See below. |

**Monte Carlo and XFEL are what is left, and what could make the port heavy**, in two ways:

- **Data.** Both read ELSEPA's angular cross-sections, one table per element:
  `constants/below_20000/` for electrons under 20 keV (7.3 MB, 2.9 MB gzipped) and
  `constants/above_20000/` above it (95 MB, 23 MB with xz) -- plus small Auger and
  fluorescence line tables. Photoelectrons from an X-ray beam under 20 keV never need the large
  set. So the small set can go in an optional data package, and the large one in another, for
  hard X-rays only. Kept as compressed text and read with the standard library, the whole lot
  parses in 1.4 s, one element at a time as a run needs it. MicroED, which the analysis had
  feared needed the large set, reads it only from code RADDOSE-3D has commented out.
- **Speed.** RADDOSE-3D's Monte Carlo takes 19 s per 10^5 photons, XFEL 417 s and 1 GB, so
  about 3 and 70 minutes at their default 10^6. Matching the Java draw for draw needs each
  electron followed step by step, because its random draws depend on its history: secondaries
  are tracked depth-first, and some samples retry until accepted. Per step the Java recomputes
  a cross-section integral of about 10^5 operations, which NumPy does in one call, and much it
  could cache. Whether a step-by-step Python port comes near the Java's speed is unmeasured; a
  prototype of one transport step settles it. If it cannot, the choice is a vectorised port
  compared statistically, or an optional compiled accelerator -- and numba alone is over
  100 MB.

## Milestones ahead

Milestones 0 to 4, and MicroED, are done (`HISTORY.md`). Restated on 2026-10-13, after #40:

| milestone | in short | done when | waits on |
|---|---|---|---|
| 5. Light, and tested beyond one Mac | the weight changes; CI on three platforms | the install is about 35 MB without torch, and 73 of 73 cases pass in CI on Linux, macOS and Windows | a remote for the repository |
| 6. Every RADDOSE-3D input, read as CCP4 asks | PDB codes; the last hand-written readers; two known divergences | every input RADDOSE-3D accepts runs with its answer, or is refused for good, and nothing but perhaps `.txt` is read by hand | CCP4, on `.txt` |
| 7. Monte Carlo and XFEL | made reproducible in RADDOSE-3D (#42), ported as they are, then fixed in both | their tier-B cases pass, and their data sits in optional packages | the group, on the physics fixes |
| 8. Several settings at once | the rest of Milestone 3 | a worked example beats a parameter scan, against a published result | nobody |
| 9. Publication and CCP4 | the paper; the Working Group 2 submission | submitted | the relicensing |

**Order.** Milestone 5 comes first: it is small, and once it is done CI checks everything after
it. Milestone 6 finishes the claim that any input RADDOSE-3D accepts works here, for everything
already ported. Milestone 7 is the largest, and starts in RADDOSE-3D, so its first step can run
alongside 5 and 6. Milestone 8 can be taken up at any point, and Milestone 9 waits on people.

### Milestone 5 — Light, and tested beyond one Mac

- **The weight changes** in *Keeping it light*: scipy replaced by the standard library, the
  version written at build time, `types-PyYAML` moved to the dev group, the description fixed,
  and the guard tests for the dependency list, lazy imports and the data budget. Each keeps the
  corpus unchanged.
- **A remote**, even a private one, so that CI runs: Linux, macOS and Windows, and the notebooks.
- **Escape off macOS.** The escape code takes `cos` and `sin` from the platform's C library, which
  match the Java on macOS but may differ in the last bit elsewhere, and a last-bit difference can
  move a track into the next voxel (`UPSTREAM_FIXES.md`, *Found while fixing*). Linux CI shows
  whether it does.
- **Re-measure the tolerances.** The escape tolerances predate #38, which redrew the Monte Carlo
  realisations; every case still passes, but the margins are unmeasured. The dose-centroid
  tolerance is set by the antimony case and catches the cylinder's mirrored dose field with only
  a factor of two to spare, so it wants setting per case.

**Done when** the install is about 35 MB without torch, the import about 75 ms, both pinned by
tests, and 73 of 73 cases pass in CI on all three platforms.

### Milestone 6 — Every RADDOSE-3D input, read as CCP4 asks

For the MX calculation and MicroED: any input file that works in RADDOSE-3D works here, with
RADDOSE-3D's answer.

- **PDB entries by code.** RADDOSE-3D downloads `PDB 3I40` from RCSB; the port looks for a file
  called `3I40` and fails. Download with the standard library and read with Gemmi, as now. A test
  that downloads is not reproducible, so the corpus keeps using local files.
- **The last hand-written readers**, from CCP4's request: PGM beam images to `netpbmfile`
  (Pillow will not do: it rescales any maxval but 255 and 65535, and rounds), `.obj` meshes to
  `trimesh`, and the vendored tables to `csv` or NumPy. Each is checked against the reader it
  replaces, on the corpus and beyond, as Gemmi was. RADDOSE-3D's `.txt` input waits on CCP4: if
  it counts as a bespoke parser, it becomes a Lark grammar mirroring the Java's `Inputfile.g`.
- **Two known divergences**, fixed in RADDOSE-3D first as before, then matched:
  - a third `Dimensions` value on a cylinder makes RADDOSE-3D treat it as a cuboid in MicroED,
    which the port does not;
  - RADDOSE-3D skips a character its grammar does not know, with a message, and carries on.
    The port warns. `input-b3x14.txt`, which RADDOSE-3D ships, has a stray `ß`.

**Done when** every input RADDOSE-3D accepts runs here with its answer, or is one of the two
refused for good (`AbsCoefCalc RDV2`/`RDV3`, `Type Experimental`), and no file but perhaps
`.txt` is read by hand.

### Milestone 7 — Monte Carlo and XFEL

| subprogram | keyword | source | what it models |
|---|---|---|---|
| Monte Carlo | `MONTECARLO`, `GOS` | `MC.java`, 4,809 lines | electron transport photon by photon, `outputMC.CSV` |
| XFEL | `XFEL` | `XFEL.java`, 4,388 lines | the same, timed within a pulse; ionisation per atom, `outputXFEL.CSV` |

They are one program twice: 76 of MC's methods are identical in XFEL, and both lean on the GOS
physics MicroED already ported. One shared electron-transport core serves both.

**Decided:** the port is compared with RADDOSE-3D *statistically*, not draw for draw. Matching
draw for draw would mean following every electron in the Java's depth-first order with its
random numbers; comparing statistically lets the port vectorise across particles, which is
what can make it fast in Python. And the Java is fixed in stages around the port, not all
first:

1. **Make RADDOSE-3D reproducible and runnable, with no physics change**:
   [#42](https://github.com/GarmanGroup/RADDOSE-3D/pull/42), open.
   - Elements are chosen in atomic-number order: identity-hash order had meant the same seed gave
     a different run on a different JVM.
   - Every draw is seeded.
   - `SimPhotons 1e6` is accepted.
   - XFEL's arrays are sized from the pulse, which cuts its memory by two-thirds.
   - Unrunnable inputs are refused before the run.
   - Two small numeric fixes it ran into: a truncated charge sum in the stopping power, and
     XFEL's per-voxel occupancy cache, which biased small non-cuboid crystals by about 6%.
2. **Port foundations**:
   - the models and `.txt` keywords;
   - the Auger, fluorescence and overvoltage tables, about 160 KB, in the core;
   - ELSEPA's angular tables as optional data packages;
   - tier-B (statistical) conformance, calibrated: Java against Java and the port against
     itself must pass, and mutations of the claimed detectable size must fail.
3. **A timing gate**: the vectorised core for the simple Monte Carlo against the Java. Within
   about 3× of it, continue; otherwise reconsider.
4. **Port the current physics**, defects and all: the simple Monte Carlo, then GOS, then XFEL.
5. **Then fix the physics in both**, PR by PR with the group's sign-off, as for escape and
   MicroED. The defects -- "isotropic" directions with a uniform polar angle, uniform Compton
   angles, elastic angles without their sin θ weight, a wrong direction-vector formula, cascades
   that double-count hole energy, and more -- are listed in `UPSTREAM_FIXES.md`.

No gradients: a randomly drawn track has none, as with escape.

**Done when** their tier-B cases pass and the core package stays within its weight budget.

### Milestone 8 — Several settings at once

The rest of Milestone 3. The beam energy is optimised by gradient
(`docs/examples/06_optimising_energy.ipynb`). What is left:

- **Several settings at once**: energy, beam size, exposure and translation together, under a
  dose budget. This is where gradients beat scans by orders of magnitude rather than a factor of
  a few. Adam-style optimisers will need the same care at absorption edges that
  `maximise_energy` takes.
- **Re-check the energy example with escape on.** Escaping photoelectrons take dose with them,
  more so at high energy and in small crystals. Escape has no gradient, so the search cannot
  include it, but its chosen energies can be re-run with it.
- **A published comparison**, against a published result on choosing an energy. That would make
  it the paper's figure.
- **Anomalous signal.** The energy search is for native data. A SAD or MAD objective needs a
  model of anomalous differences.

The limits stay as `HISTORY.md` records them. `TUNABLE` and `FIXED` say which settings carry a
gradient and why. The integer image count and voxel occupancy are steps an optimiser cannot
see. And the torch and NumPy backends are not bit-identical: they agree to about 2e-4 on the
summary, so the conformance claim is NumPy's.

**Done when** a worked example finds a better strategy by gradient than a parameter scan does
for the same compute, against a published result.

### Milestone 9 — Publication and CCP4

Gated on the relicensing, not on code. The paper is on differentiable strategy optimisation.
Submission to CCP4 goes through Working Group 2, once the licence permits.

## Waiting on people

None of these is code.

| decision | who | blocks |
|---|---|---|
| Relicensing: put the question to RADDOSE-3D's copyright holders -- Gerstel, Brooks-Bartlett, Ginn, Bury, Walsh, Zeldin and others. There is no CLA, so it needs all of them. Measured in months. | the group | Milestone 9 |
| A remote for the repository, even a private one | the author | Milestone 5's CI |
| Does the `.txt` input count as a bespoke parser? Which of `netpbmfile` and `trimesh` does CCP4 already ship, or accept? Does it want a command-line program? | CCP4 | Milestone 6's last item; a CLI |
| Sign-off on fixing Monte Carlo and XFEL's physics defects in RADDOSE-3D | the group | Milestone 7 |

The licence itself is settled: AGPL-3.0-only, as RADDOSE-3D is (`HISTORY.md`, Milestone 0).
CCP4's route needs the work to fit its licence model and a Contributor Agreement signed by the
copyright holders, so relicensing is on the critical path for CCP4 alone.

## Upstream: RADDOSE-3D

The port's fixes are merged into RADDOSE-3D as #36, #37, #38 and #40, all on 2026-10-13
(`UPSTREAM_FIXES.md` has each, with measurements). #41 adds MicroED's beam requirements and
output files to the user guide. The corpus follows `master` at `e2df6e4`.

**Still open in the Java**, each with the milestone that takes it up:

- Monte Carlo and XFEL's physics defects, and the GOS model's missing inner subshells from
  Z = 35 (Milestone 7).
- A third `Dimensions` value making a cylinder a cuboid, and lexer errors not being refused
  (Milestone 6).
- Container runs fetch NIST's tables over the network on every run, and NIST refuses Java 8's
  requests, which then fail with a `NullPointerException`. The port vendors the tables; the Java
  should too.
- An NMR or EM model given to `AbsCoefCalc EXP` gets a dose for a cell of water. With no
  `REMARK 290`, `CoefCalcFromPDB` has no symmetry operators, so no macromolecule. The port
  refuses such files.
- `MicroED.getSolventFraction()` returns 0 when explicitly set; it is read only by MicroED's
  disabled charge model.

## Verification, throughout

- Every change keeps the conformance corpus identical unless it deliberately changes a number,
  and a changed expectation is a reviewable diff. The baseline is 73 of 73. The worst deviation
  is 1.3e-4 (`063-wedge-full-rotation`, `dose_threshold`, a quantised quantity), and 5.0e-5 for
  the rest. `pytest tests/conformance` prints the per-case table.
- New readers and writers are checked against RADDOSE-3D's own output for the same input,
  through `generate.py`, never against hand-written expectations.
- `solve()` with no overrides stays pure NumPy. `test_no_overrides_stays_on_numpy` and the
  architecture test enforce it.
- Weight is measured, not assumed. The guard tests above hold the line, and the numbers in
  *How heavy it is* are re-measured before each release.
- Every new conformance comparison is mutation-checked: perturb a constant, and confirm the
  relevant cases fail.
- Notebooks run as tests, so an example that stops working fails the build rather than
  misleading a reader.
