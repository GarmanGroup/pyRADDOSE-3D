# pyRADDOSE-3D road-map

Where the port is, what is left, and how it stays small. Stock taken on 2026-10-15. File and
line references are there so a reader can check a claim rather than take it on trust.

What was done to get here, and why, is in `HISTORY.md`. The defects the port found in
RADDOSE-3D, and their fixes, are in `UPSTREAM_FIXES.md`.

## Where the port stands

**It does what RADDOSE-3D's MX calculation and its MicroED engine do, value for value, and what
its Monte Carlo and XFEL subprograms do, statistically.**

- 18,100 lines of Python in `src/` and 13,400 in `tests/`. 1,736 tests, 44 of them slow, none
  expected to fail.
- Every input RADDOSE-3D's MX calculation accepts is ported, except two that cannot be:
  `AbsCoefCalc RDV2`/`RDV3` run the original Fortran RADDOSE as an external program, and
  `Type Experimental` cannot be built from an input file by RADDOSE-3D either.
- 95 conformance cases, generated from RADDOSE-3D. 73 match it value for value within measured
  tolerances: the sixteen summary numbers, the dose field and per-image series behind them, and
  the output files themselves, photoelectron and fluorescence escape included. The other 22 are
  its Monte Carlo, GOS and XFEL subprograms, which are random, and are compared statistically:
  20 runs on each side, five quantities a case.
- **MicroED** (`Subprogram EMED`) is ported: the dose to a thin crystal in an electron beam,
  slice by slice, the scattering counts, and the grid search for the best voltage and
  thickness. Its fourteen cases agree with RADDOSE-3D to the last printed digit or two, now that
  RADDOSE-3D has its MicroED fixes (#40). It needs no new dependency and 0.37 MB of
  data.
- **The Monte Carlo and XFEL subprograms** (`Subprogram MONTECARLO`, `GOS`, `XFEL`) are ported:
  each photon and every electron it frees, by stopping power, through the GOS model's
  collisions, or through those within one X-ray pulse. Their 22 cases agree with RADDOSE-3D's
  runs: no quantity's mean differs by more than 2.9 standard errors, where the test allows about
  five. Following every electron of every run at once, the port is about ten times as fast as
  the Java on the simple Monte Carlo. ELSEPA's electron tables come as optional data packages.
- It reads RADDOSE-3D's `.txt` input files as well as YAML, and writes the same eight output
  files RADDOSE-3D writes.
- `solve()` is differentiable end to end, apart from escape, which is Monte Carlo. Ten tunable
  settings carry gradients. `programs/optimise` chooses a beam energy by gradient.
- Two run scripts, `programs/dose` and `programs/optimise`, and eight notebooks, each run as a
  test.

**What it is not yet**, and the milestone that changes it:

- **Tested beyond one Mac.** CI has run once, on GarmanGroup/pyRADDOSE-3D: macOS passed, and
  Linux and Windows failed on voxels lying on a crystal face. The fix awaits a second run
  (Milestone 5).
- **Reading every file as CCP4 asks.** PGM beam images, `.obj` meshes and the vendored tables
  are still read by hand, and stay so for now (Milestone 6).
- **Optimising several settings at once**, which is the case that most needs gradients
  (Milestone 8).
- **In CCP4.** That needs the relicensing, and answers from CCP4 (Milestone 9).

## How heavy it is

Measured in a fresh environment holding only the runtime dependencies (Python 3.12, macOS on
Apple silicon).

**Install: 107 MB, two-thirds of it scipy, which stays: CCP4 ships it already.**

| package | installed | used for |
|---|---|---|
| scipy | 72 MB | `erf`, `erfinv` and the normal distribution's quantile |
| numpy | 22 MB | everything |
| pydantic and pydantic-core | 6.5 MB | the input models |
| gemmi | 6.1 MB | PDB, mmCIF, CIF and sequence files |
| pyyaml | 0.6 MB | YAML input |
| pyRADDOSE-3D itself | 1.1 MB | 670 KB of it data; the wheel is 363 KB |
| torch, optional | 552 MB | gradients only |

RADDOSE-3D's release jar is 53 MB, and needs a JVM.

**Start-up.** Importing the package takes 3 ms: its version is written when it is built, where
looking it up in `importlib.metadata` took 55 ms. Importing the solver takes about 190 ms warm,
on a busy machine: NumPy about 65 ms, and the input models about 125 ms. The first run that
needs `erf` or `erfinv` imports scipy's special functions, about 200 ms more. On a fresh install,
the first import of scipy compiled its bytecode and took 14 seconds, once.

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
   `tests/test_weight.py` pins it.
2. **Heavy or rarely needed libraries import lazily**, inside the function that needs them, as
   torch and scipy do. `import pyraddose_3d.solve` must not import torch or scipy, which
   `tests/test_weight.py` checks in a fresh interpreter. It does import pydantic: whoever calls
   it has built the input models already.
3. **The core package's data stays under 1 MB**, pinned by the same tests: it is 828 KB. Bulk
   data goes in a separate, optional package.
4. **Optional capabilities are extras**: torch for gradients now, and any subprogram's data
   later.

### Milestone 5's changes

| change | saves | state |
|---|---|---|
| Replace scipy with the standard library | 72 MB, 67% of the install, and ~200 ms | **not done**, by decision: CCP4 ships scipy already |
| Write the version at build time | 55 ms, about 40% of the import | done: `hatch-vcs` writes `_version.py` |
| Move `types-PyYAML` to the dev group | a stub package | done: it is for mypy, not for running |
| Fix the description | — | done: torch is optional |
| The guard tests above | — | done: `tests/test_weight.py`, for the dependency list, the lazy imports of torch and scipy, the metadata lookup, and the data budget |

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
| Monte Carlo and XFEL, done | 2.3 MB of data in an optional package; 23 MB more, in another, for hard X-rays | See below. |

**Monte Carlo and XFEL were what could make the port heavy**, in two ways. Neither did:

- **Data.** Both read ELSEPA's angular cross-sections, one table per element:
  `constants/below_20000/` for electrons under 20 keV and `constants/above_20000/` above it
  (95 MB), plus small Auger and fluorescence line tables, which are in the core. The two sets are
  optional data packages, both committed: `packages/elsepa-low`, 2.3 MB, and
  `packages/elsepa-high`, 23 MB with xz. They are not on PyPI yet, so a pip install from GitHub
  names their subdirectories (README, *Installing*).
  Photoelectrons from an X-ray beam under 20 keV never need the large set.
- **Speed.** The port follows every electron of every run at once, and is compared with the Java
  statistically rather than draw for draw. On the simple Monte Carlo it is about ten times as
  fast as the Java. In the GOS model the Java recomputed a cross-section integral at every step
  of every electron, summing over the composition twice at each of its thousand points;
  [#51](https://github.com/GarmanGroup/RADDOSE-3D/pull/51) sums once per integral, with
  identical results, and makes a GOS run fourteen times faster: 18 s rather than 245 s for one
  run of case 160. The port, which tabulates the integrals once per run, takes 32 s for all
  twenty. An XFEL case takes the port about a minute for twenty runs of up to a million photons.

## Milestones ahead

Milestones 0 to 4, and MicroED, are done (`HISTORY.md`). Restated on 2026-10-13, after #40:

| milestone | in short | done when | waits on |
|---|---|---|---|
| 5. Light, and tested beyond one Mac | the small weight changes (done); CI on three platforms (first run, and its fixes, done) | the corpus passes in CI on Linux, macOS and Windows | a second CI run |
| 6. Every RADDOSE-3D input, read as CCP4 asks | PDB codes and two known divergences (done); the last hand-written readers (left for now) | every input RADDOSE-3D accepts runs with its answer, or is refused for good, and nothing but perhaps `.txt` is read by hand | CCP4, on `.txt`; the decision to change the readers |
| 7. Monte Carlo and XFEL | made reproducible in RADDOSE-3D (#42), ported as they are, then fixed in both | their tier-B cases pass, and their data sits in optional packages | the group, on the physics fixes |
| 8. Several settings at once | the rest of Milestone 3 | a worked example beats a parameter scan, against a published result | nobody |
| 9. Publication and CCP4 | the paper; the Working Group 2 submission | submitted | the relicensing |

**Order.** Milestone 5 comes first: it is small, and once it is done CI checks everything after
it. Milestone 6 finishes the claim that any input RADDOSE-3D accepts works here, for everything
already ported. Milestone 7 is the largest, and starts in RADDOSE-3D, so its first step can run
alongside 5 and 6. Milestone 8 can be taken up at any point, and Milestone 9 waits on people.

### Milestone 5 — Light, and tested beyond one Mac

- **The weight changes** in *Keeping it light* (done, on 2026-10-17): the version written at
  build time, `types-PyYAML` moved to the dev group, the description fixed, and the guard tests.
  Scipy stays, by decision: CCP4 ships it already.
- **A remote** (done, on 2026-10-17): GarmanGroup/pyRADDOSE-3D, private. Its first CI run
  passed on macOS and failed on Linux and Windows, all for one reason: a voxel whose centre lies
  on a crystal face is inside or outside by the last bit of the rotated coordinates, and the
  platforms' `sin` and `cos` differ in that bit. macOS on ARM rounds 4 of `test_exposure`'s 55
  angles away from the nearest double, Linux on x86 none, and Windows sin(45 degrees); Windows'
  reports moved by up to 4e-4. Every rotation now takes its cosine and sine correctly rounded
  (`geometry.cos_sin`), so the port gives the same numbers on every platform: macOS now gives
  Linux's, bit for bit. Against the Java, which rounds 23 of the corpus's 171 angles differently,
  the worst summary field moved from 6.5e-5 to 7.1e-5, inside the 1e-4; given the Java's own
  values, `test_summary.py` holds all five of its cases to 1e-11. MicroED's inelastic outputs got
  a measured tolerance of their own (`MANIFEST.yaml`), and check-manifest ignores the version
  file written at build time.
- **Escape off macOS.** The escape code takes the cosines and sines of its track directions from
  the platform's C library, which may differ from the Java's in the last bit, and a last-bit
  difference can move a track into the next voxel (`UPSTREAM_FIXES.md`, *Found while fixing*).
  Linux and Windows CI passed every escape case at the current tolerances; the rotation of the
  tracks now uses `cos_sin`.
- **Re-measure the tolerances.** The summary, array and MicroED tolerances were re-measured on
  2026-10-17 against correctly rounded rotations (`MANIFEST.yaml`). The dose-centroid tolerance is
  set by the antimony case and catches the cylinder's mirrored dose field with only a factor of
  two to spare, so it wants setting per case.

**Done when** the corpus passes in CI on all three platforms.

### Milestone 6 — Every RADDOSE-3D input, read as CCP4 asks

For the MX calculation and MicroED: any input file that works in RADDOSE-3D works here, with
RADDOSE-3D's answer.

- **PDB entries by code** (done): `PDB 3I40` is downloaded from RCSB's address for it, as
  RADDOSE-3D downloads it, once per process, with the standard library, and read with Gemmi as a
  local file is. Only a four-character code is downloaded; any other name is a path, so a local
  mmCIF file still works. Tests serve the entry locally; one marked `network`, deselected by
  default, downloads it.
- **The last hand-written readers**, from CCP4's request, left as they are for now by decision: PGM beam images to `netpbmfile`
  (Pillow will not do: it rescales any maxval but 255 and 65535, and rounds), `.obj` meshes to
  `trimesh`, and the vendored tables to `csv` or NumPy. Each is checked against the reader it
  replaces, on the corpus and beyond, as Gemmi was. RADDOSE-3D's `.txt` input waits on CCP4: if
  it counts as a bespoke parser, it becomes a Lark grammar mirroring the Java's `Inputfile.g`.
- **Two known divergences**, fixed in RADDOSE-3D first as before, then matched (done):
  - a third `Dimensions` value on a cylinder made RADDOSE-3D treat it as a cuboid in MicroED,
    which the port did not. [#64](https://github.com/GarmanGroup/RADDOSE-3D/pull/64) takes a
    cylinder's shape from its `Type`, and the port now warns of the third value, as RADDOSE-3D
    does;
  - RADDOSE-3D skipped a character its grammar does not know, with a message, and carried on,
    and the port warned. [#65](https://github.com/GarmanGroup/RADDOSE-3D/pull/65) refuses it, and
    so does the port, every one in one message; both read a byte-order mark and a final comment
    with no newline. `input-b3x14.txt`, an example input in the author's RADDOSE-3D directory
    but not in its repository, has a stray `ß`, and both now refuse it.

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
   [#42](https://github.com/GarmanGroup/RADDOSE-3D/pull/42), merged as `82ac582`.
   - Elements are chosen in atomic-number order: identity-hash order had meant the same seed gave
     a different run on a different JVM.
   - Every draw is seeded.
   - `SimPhotons 1e6` is accepted.
   - XFEL's arrays are sized from the pulse, which cuts its memory by two-thirds.
   - Unrunnable inputs are refused before the run.
   - Two small numeric fixes it ran into: a truncated charge sum in the stopping power, and
     XFEL's per-voxel occupancy cache, which biased small non-cuboid crystals by about 6%.
2. **Port foundations**:
   - the models and `.txt` keywords (done: the three subprograms and their settings are read
     and written, and running one is refused until it is ported);
   - the Auger, fluorescence and overvoltage tables, about 160 KB, in the core (done);
   - ELSEPA's angular tables as optional data packages (done: `packages/elsepa-low`, 2.3 MB,
     committed; `packages/elsepa-high`, 23 MB, built locally from a RADDOSE-3D checkout, and
     committed on 2026-10-17 so that an install from the repository has it);
   - tier-B (statistical) conformance, calibrated: Java against Java and the port against
     itself must pass, and mutations of the claimed detectable size must fail. (Done: 22
     cases, 20 runs each, five quantities each compared by Welch's t, 110 comparisons sharing a
     false-alarm rate of 1e-3; RADDOSE-3D's first ten runs agree with its last ten, and so do
     the port's. The doses' detectable bias is 4 to 7% for MONTECARLO, 5 to 7% for GOS and 6%
     for XFEL, but about 19% for the first case of each of these two, made when the Java took
     a second per absorbed photon.)
3. **A timing gate**: the vectorised core for the simple Monte Carlo against the Java. Within
   about 3× of it, continue; otherwise reconsider. (Passed: on case 140, the port takes 0.4–0.5 s
   a run of 100,000 photons and 3.4–4.5 s of 1,000,000, against RADDOSE-3D's 4.3–4.9 s and
   41–48 s, JVM start-up excluded: ten times faster. GOS's inelastic mean free path, which the
   Java recomputes every step, costs 0.66 ms per energy in the port's scalar MicroED code, so it
   can be tabulated once per run.)
4. **Port the current physics**, defects and all: the simple Monte Carlo, then GOS, then XFEL.
   (The simple Monte Carlo is ported: `backend/montecarlo/` follows every electron of every run
   at once, with a surrounding, a pink beam, and RADDOSE-3D's three files and printout, which
   `programs/dose/run_dose.py` writes. Case 140, a 20 um crystal, is weakly sensitive to how
   electrons are followed -- 12 keV photoelectrons stop well inside it, so a stopping power 20%
   low moves its dose by 3% -- so thirteen more cases are crystals of 3 to 8 um, which electrons
   escape from: there it moves the dose by 7 to 15%. They vary the beam's shape and energy, the
   crystal's shape and rotation, the polarisation, a surrounding and a pink beam, and include
   iodine and sodium for [#43](https://github.com/GarmanGroup/RADDOSE-3D/pull/43)'s fix. The
   comparisons with the port take forty minutes, and are `slow`; every case also runs
   briefly in the default suite. GOS is ported too: `gos_transport.py` follows each electron
   through the GOS model's collisions with the Java's per-electron state, its quirks included,
   frees secondaries and relaxes every vacancy in full, from tables of the subshells' cross
   sections built once per run. So is XFEL, the same transport with a clock: photons arrive
   through the pulse, steps take their flight time and relaxations their lifetimes, nothing is
   followed past the last photon's exit, and each voxel keeps what arrives before its cutoff.
   On case 180 the time cut keeps 62% of the dose on both sides. XFEL's surrounding, rotation
   and offset, which RADDOSE-3D's own copy got wrong and the port refused, run as the Monte
   Carlo's since #57, and cases 184 and 185 test them.)
5. **Then fix the physics in both**, PR by PR with the group's sign-off, as for escape and
   MicroED. (Begun: [#44](https://github.com/GarmanGroup/RADDOSE-3D/pull/44) to
   [#49](https://github.com/GarmanGroup/RADDOSE-3D/pull/49), merged, fix a rotated crystal's
   escape distances, circular beams, elastic deflections' sin θ, the directions electrons and
   photons take, MC's fluorescence energies with four smaller slips, and the surrounding. The
   port follows each in its own commit and agreed with RADDOSE-3D at each. The largest effect
   is #48's: at 25 keV a crystal with selenium loses 22% of its Monte Carlo dose. Cases 150 and
   151 test the wedges, 152 and 153 the surrounding. GOS's and XFEL's followed:
   [#51](https://github.com/GarmanGroup/RADDOSE-3D/pull/51) and
   [#52](https://github.com/GarmanGroup/RADDOSE-3D/pull/52), merged, then
   [#54](https://github.com/GarmanGroup/RADDOSE-3D/pull/54) to
   [#60](https://github.com/GarmanGroup/RADDOSE-3D/pull/60), open as a stack: cascades that
   counted a vacancy's energy twice, the mean free paths, simple mode's outputs, XFEL on the
   Monte Carlo's code, XFEL's pulse, lifetimes and elastic count, the polarisation, and heavy
   atoms' subshells. The largest effects: a selenium crystal's GOS dose falls 37% (#54), and
   1 M iodide lengthens MicroED's inelastic mean free path by 27% (#60). Nothing found porting
   them is left open. On them, [#61](https://github.com/GarmanGroup/RADDOSE-3D/pull/61) sends
   a pink XFEL pulse's photons in no order of energy, and
   [#62](https://github.com/GarmanGroup/RADDOSE-3D/pull/62) adds a Gaussian pulse.
   [#66](https://github.com/GarmanGroup/RADDOSE-3D/pull/66), open, fixes issue #17: the
   exposed region, and its volume, now follow an offset or a rotation, which moves case 151's
   exposed-region dose by +83% and 185's by +148%.)

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
| Does the `.txt` input count as a bespoke parser? Which of `netpbmfile` and `trimesh` does CCP4 already ship, or accept? Does it want a command-line program? | CCP4 | Milestone 6's last item; a CLI |
| Sign-off on fixing Monte Carlo and XFEL's physics defects in RADDOSE-3D | the group | Milestone 7 |

The licence itself is settled: AGPL-3.0-only, as RADDOSE-3D is (`HISTORY.md`, Milestone 0).
CCP4's route needs the work to fit its licence model and a Contributor Agreement signed by the
copyright holders, so relicensing is on the critical path for CCP4 alone.

## Upstream: RADDOSE-3D

The port's fixes are merged into RADDOSE-3D as #36, #37, #38, #40 and #42, all on 2026-10-13,
and #43 to #49, the Monte Carlo and XFEL's, on 2026-10-14 (`UPSTREAM_FIXES.md` has each, with
measurements). #41 adds MicroED's beam requirements and output files to the user guide. #51,
which makes the GOS model twelve times faster, and #52, which stops GOS and XFEL losing every
electron that collides with phosphorus's or selenium's valence electrons, were merged on
2026-10-15. #54 to #62, GOS's and XFEL's remaining fixes and XFEL's pulse, were merged on
2026-10-17, and `master` at `a79eb45` has the same source as the stack's top. The corpus follows
`master` at `85a6813`, but for its 28 Monte Carlo, GOS and XFEL cases and MicroED's 121 and 134,
which were generated from the stack's branches: #59 (`c1566d2`) for the 19 that the rest leave
unchanged, #60 (`cff0a59`), #61 (`29347d6`) and #62 (`e0529dc`) for the cases each changes or
adds. [#64](https://github.com/GarmanGroup/RADDOSE-3D/pull/64) and
[#65](https://github.com/GarmanGroup/RADDOSE-3D/pull/65), Milestone 6's two divergences, were
merged on 2026-10-17 as `882803b` and `8f1aef0`; neither moves a corpus number. Cases 151 and
185 come from [#66](https://github.com/GarmanGroup/RADDOSE-3D/pull/66) at `28b90f4`, open.

**Still open in the Java**, each with the milestone that takes it up:

- A third `Dimensions` value making a cylinder a cuboid, and lexer errors not being refused,
  fixed by #64 and #65, open (Milestone 6).
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
