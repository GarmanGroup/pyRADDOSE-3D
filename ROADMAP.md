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
  RADDOSE-3D has its MicroED fixes (#40, step 4). It needs no new dependency and 0.37 MB of
  data.
- It reads RADDOSE-3D's `.txt` input files as well as YAML, and writes the same eight output
  files RADDOSE-3D writes.
- `solve()` is differentiable end to end, apart from escape, which is Monte Carlo. Ten tunable
  settings carry gradients. `programs/optimise` chooses a beam energy by gradient.
- Two run scripts, `programs/dose` and `programs/optimise`, and six notebooks, each run as a
  test.

**What it is not yet:**

- **Published.** The repository has never been pushed, so its CI has never run: no Linux, no
  Windows, and no notebooks in CI. Everything has been tested on one Mac.
- **In CCP4.** That needs the relicensing, and answers from CCP4 (both under *Waiting on
  people*).
- **Optimising several settings at once**, which is the case that most needs gradients.
- **RADDOSE-3D's other two subprograms**, Monte Carlo and XFEL.

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

### Do now: about 70% off the install, 40% off start-up

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

## Next, in order

### 1. The weight changes above

Small and independent, with nobody to wait for. Each one keeps the conformance corpus
unchanged.

### 2. Finish CCP4's no-bespoke-parsers request

CCP4's developers have said they may not accept code that implements its own file parsers, and
they encourage Gemmi for coordinate and reflection data. RADDOSE-3D reads no reflection data.

**Done.** PDB, mmCIF, small-molecule CIF and sequence files are read with Gemmi. Checked against
the hand-written readers they replaced:

- every crystal structure in a random sample of 211 PDB entries gave the same composition;
- so did all 50 COD files;
- NMR and EM models are now refused.

Three conformance cases pin the PDB route to RADDOSE-3D itself, for NCS, nucleic acids and
ligands. The comparison tool is kept on the local branch `gemmi-readers-comparison`.

**Still read by hand:**

- **PGM beam images** (`beam.read_pgm`), to move to `netpbmfile`. Pillow will not do: it
  rescales any maxval but 255 and 65535 and rounds, which moves the profile.
- **`.obj` meshes** (`wireframe.read_obj`), to move to `trimesh`.
- **The data tables vendored from the Java**, read by splitting lines, to move to `csv` or
  NumPy. Not a file format in CCP4's sense, but cheap to settle.
- **RADDOSE-3D's `.txt` input** (`_raddose_txt.py`). No library reads it. Whether a program's
  own keyword input counts is for CCP4 to say. If it does, it becomes a Lark grammar mirroring
  the Java's `Inputfile.g`.

### 3. Waiting on people

These decide what happens after step 2, and none of them is code.

| decision | who | blocks |
|---|---|---|
| Relicensing: put the question to RADDOSE-3D's copyright holders -- Gerstel, Brooks-Bartlett, Ginn, Bury, Walsh, Zeldin and others. There is no CLA, so it needs all of them. Measured in months. | the group | CCP4, and nothing else |
| Does the `.txt` input count as a bespoke parser? Which of `netpbmfile` and `trimesh` does CCP4 already ship, or accept? Does it want a command-line program? | CCP4 | step 2's last item; the CLI |
| A remote for the repository, even a private one | the author | CI on Linux and Windows; the notebooks in CI |
| Sign-off on fixing Monte Carlo and XFEL's physics defects in RADDOSE-3D (step 6) | the group | porting them faithfully |

The licence itself is settled: AGPL-3.0-only, as RADDOSE-3D is (`HISTORY.md`, Milestone 0).
CCP4's route needs the work to fit its licence model and a Contributor Agreement signed by the
copyright holders, so relicensing is on the critical path for CCP4 alone.

### 4. Upstream: RADDOSE-3D

The fixes the port found are merged into RADDOSE-3D, as #36, #37, #38 and #40 on 2026-10-13:

- **#36**: input that was silently ignored, `Type Experimental`, `-o voxelfluences`, and the
  `PixelSize` docs;
- **#37**: pink beams, collimation on measured beams, and the wedge image count;
- **#38**: escape modelling. With photoelectron escape alone, its binding-energy fix raises the
  dose by 3.5% for protein and by 20% for SeMet above the Se K edge. Its fluorescence-direction
  fix moves doses by under 0.1%.

[#40](https://github.com/GarmanGroup/RADDOSE-3D/pull/40) is the MicroED fixes, merged as
`e2df6e4`. The corpus is regenerated from `master` at that commit. The MX numbers moved by at
most 3e-14, from the order a `HashSet` sums elements in. #40:

- ends the `NullPointerException` that every XFEL, MicroED and Monte Carlo run ended in;
- gives the electron physics the whole crystal for PDB, sequence and CIF compositions. It used to
  see one molecule's protein, the solvent alone, or nothing;
- makes the distant-collision integral use one shell model, which lengthens insulin's inelastic
  mean free path at 300 keV from 234.20 to 253.15 nm;
- gives Na, Al, Ca, Sc, Ti, V, Cr, Ga and Ge the binding energies the GOS model reads, which the
  tables lacked, from the X-ray Data Booklet and, for Al 3s and Cr 3d, Carlson. Al used to
  crash, and the others gave NaN and grid searches that reported 1990 kV and 1995 nm. Should a
  table still lack one, MicroED now warns and reports NaN without searching;
- gives a crystal thinner than 5 nm one slice, steps slice depths by the real width, refuses
  beams without an exposure, and removes Selenium, 9 MB of the 53 MB jar.

**Still open in the Java:**

- Lexer errors are not refused. A character the grammar does not know is skipped with a
  message, and the run carries on. `input-b3x14.txt`, which RADDOSE-3D ships, has a stray `ß`.
- Container runs fetch NIST's tables over the network on every run, and NIST refuses Java 8's
  requests, which then fail with a `NullPointerException`. The port vendors the tables; the Java
  should too.
- An NMR or EM model given to `AbsCoefCalc EXP` gets a dose for a cell of water. With no
  `REMARK 290`, `CoefCalcFromPDB` has no symmetry operators, so no macromolecule. The port
  refuses such files.
- MicroED, beyond #40: elements from Z = 35 have no inner subshells in the GOS model, and a
  third `Dimensions` value makes a cylinder a cuboid (`UPSTREAM_FIXES.md`).
- `MicroED.getSolventFraction()` returns 0 when explicitly set; it is read only by MicroED's
  disabled charge model.
- Monte Carlo and XFEL's physics defects, step 6.

### 5. Several settings at once (the rest of Milestone 3)

The beam energy is optimised by gradient (`docs/examples/06_optimising_energy.ipynb`). What is
left:

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

### 6. Monte Carlo and XFEL (the rest of Milestone 6)

All three subprograms are wanted. MicroED is done (`backend/microed.py`); RADDOSE-3D's MicroED
turned out to be a deterministic calculation, its Monte Carlo commented out. The other two
remain:

| subprogram | keyword | source | what it models |
|---|---|---|---|
| Monte Carlo | `MONTECARLO`, `GOS` | `MC.java`, 4,809 lines | electron transport photon by photon, `outputMC.CSV` |
| XFEL | `XFEL` | `XFEL.java`, 4,388 lines | the same, timed within a pulse; ionisation per atom, `outputXFEL.CSV` |

They are one program twice: 76 of MC's methods are identical in XFEL, and both lean on the GOS
physics MicroED already ported. One shared electron-transport core serves both. In order:

1. **Fix them in RADDOSE-3D first**, with the group's sign-off, as #38 was. Surveying them found
   physics defects -- "isotropic" directions drawn with the polar angle uniform, Compton angles
   uniform, the elastic angle sampled without its sin(theta) weight and one bin low,
   fluorescence looked up in the Auger table -- listed in `UPSTREAM_FIXES.md`. And before
   anything can match them value by value: the element a random number picks depends on a
   `HashSet`'s identity-hash order, which changes between JVMs, so RADDOSE-3D does not
   reproduce its own runs. It needs a fixed order.
2. **Prototype one transport step in NumPy and time it against the Java.** That decides how the
   port compares with RADDOSE-3D: draw for draw, or statistically.
3. **Settle the data**: the low-energy tables in one optional package, the high-energy ones in
   another.
4. **Build the corpus from the fixed Java**, then port. No gradients: a randomly drawn track has
   none, as with escape.

### 7. Publication and CCP4 (Milestone 5)

This step is gated on relicensing, not on code. The paper is on differentiable strategy
optimisation. Submission to CCP4 goes through Working Group 2, once the licence permits.

## Known gaps in the port's own testing

- **No CI has ever run.** The matrix includes Windows and the notebooks, but it needs a remote.
- **Off macOS, untested.** The escape code takes `cos` and `sin` from the platform's C library.
  Those match the Java on macOS but may differ in the last bit on Linux, and a last-bit
  difference can move a track into the next voxel (`UPSTREAM_FIXES.md`, *Found while fixing*).
- **Escape tolerances were not re-measured** after RADDOSE-3D's escape fixes redrew the Monte
  Carlo realisations. Every case still passes.
- **The dose-centroid tolerance is set by the antimony case**, so it catches the cylinder's
  mirrored dose field with only a factor of two to spare. Per-case tolerances would tighten it.

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
