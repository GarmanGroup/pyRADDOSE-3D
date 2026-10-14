# Fixing what the port found in RADDOSE-3D

Porting Milestone 4 turned up a set of defects in RADDOSE-3D's Java. It also left four places where
the port made an interim choice because the Java's own behaviour was wrong. This document explains
each one and sets out how to fix it in both codebases.

Everything here was checked against RADDOSE-3D `9ad4e9d` and pyRADDOSE-3D `0570760` on 2026-10-12.
`File.java:N` references are to `src/se/raddo/raddose3D/` in RADDOSE-3D. Port references are to
`src/pyraddose_3d/`. The numbers come from the port, which reproduces the Java value by value; where
a number matters, it was also checked against the Java itself.

Terms: *PE* is photoelectron, *FL* fluorescence, *the surrounding* the liquid modelled around the
crystal (`CalcSurrounding`), and *a pink beam* one with an energy spread (`EnergyFWHM`).

## Status

The RADDOSE-3D fixes went up as three stacked PRs, reordered for review so that the two physics
changes needing sign-off came last. All three were merged on 2026-10-13:

| PR | branch, onto | items | state |
|---|---|---|---|
| [#36](https://github.com/GarmanGroup/RADDOSE-3D/pull/36) | `jd_upstream_input_fixes`, onto `master` | 7–11, 13; `voxelfluences`; help text | merged as `b5a53b3` |
| [#37](https://github.com/GarmanGroup/RADDOSE-3D/pull/37) | `jd_upstream_beam_wedge_fixes`, onto #36 | 6, 12; the wedge image count | merged as `fca1682` |
| [#38](https://github.com/GarmanGroup/RADDOSE-3D/pull/38) | `jd_upstream_escape`, onto #37 | 1, 5, 2, 3, 4; item 3's slowdown undone | merged as `167fb7a` |

MicroED's defects followed, found porting it (*MicroED*, below):

| PR | branch, onto | items | state |
|---|---|---|---|
| [#40](https://github.com/GarmanGroup/RADDOSE-3D/pull/40) | `jd_microed_fixes`, onto `master` | all of *MicroED* but its last two rows | merged as `e2df6e4` |
| [#42](https://github.com/GarmanGroup/RADDOSE-3D/pull/42) | `jd_mc_xfel_reproducible`, onto `master` | *Monte Carlo and XFEL*: reproducibility and running | merged as `82ac582` |
| [#43](https://github.com/GarmanGroup/RADDOSE-3D/pull/43) | `jd_mc_outer_shell`, onto `master` | *Monte Carlo and XFEL*: a photoelectron of negative energy | open |

#40 and #42 were squash-merged with the same source as `ca553bc` and `3c9afaf`, which the corpus
had been generated from. The corpus is now regenerated from `master` at `82ac582`, and no number
moved.

The reordering changed no code: the top of #38 had the same source as `eec659b`, which the corpus
had been generated from. The corpus is now regenerated from `master` at `167fb7a`, so its
`reference_commit` exists upstream. As expected, no number moved.

The table below records the local commits each port commit followed, which the PRs supersede.
The escape branch gained `63fa851`, which undid a slowdown that item 3 introduced: its direction
table was built of boxed integers, and photoelectron escape ran six times slower. The branches
above it were then rebased, and port commits made before that name the old hashes, given here in
brackets.

| items | RADDOSE-3D branch | port | state |
|---|---|---|---|
| cases that show the defects first: 109, 110, 111 | `master` 9ad4e9d | `7edf4ce` | done |
| 7, 8, 9, 10, 11, 13; `voxelfluences`; help text | `jd_upstream_input_fixes` | `de87c07` | done; decisions A and B taken as recommended |
| 1: energy accounting | `jd_upstream_escape_fixes` 4deb672 | `687e7e2` | done; physics change awaiting sign-off |
| 5: NaN L-shell fluorescence | `jd_upstream_escape_fixes` ce1c0b6 | `8d5544d` | done |
| 2: fluorescence direction | `jd_upstream_escape_fixes` 31fdb9b | `81afd79` | done; physics change awaiting sign-off |
| 3: separate direction tables | `jd_upstream_escape_fixes` 8ee300b | `07bfc7b` | done |
| 4: path in front of the surrounding | `jd_upstream_escape_fixes` 7322e36 | `8ed9b2c` | done |
| item 3's slowdown undone | `jd_upstream_escape_fixes` 63fa851 | `cdf03b6` | done; no result changes |
| 6: pink beams | `jd_upstream_pink_fixes` 8f78795 (d1e5ad5) | `59949f3`, `5348d95` | done |
| 12: collimation on measured beams | `jd_upstream_pgm_collimation` 32ed072 (cb8e2db) | `a931c11` | done; decisions C and D taken as recommended |
| a case that shows the reversed-wedge defect first: 067 | `jd_upstream_pgm_collimation` 32ed072 (cb8e2db) | `747dd6e` | done |
| the wedge image count, and a reversed wedge's step | `jd_upstream_wedge_count` eec659b | `cdf03b6` | done |

Every item above is done. Two older items remain open (see *Older upstream items*), and one
defect found while fixing these is recorded but not fixed (see *Found while fixing*).

## Found while fixing

- **An unreadable sequence file was ignored.** `parseSequenceFile` printed "Could not read sequence
  file." and carried on with no residues, giving a dose for a crystal of solvent alone. Now an
  error, in `jd_upstream_input_fixes`.
- **A reversed wedge also lost its requested step.** `Wedge` tested `AngularResolution` against the
  signed span, so a wedge running backwards always fell back to 2 degrees. The truncation was
  worse than the plan said too: 0 to 90 degrees at 3 had 30 images, not 31, and 30 to 0 at 2 had
  13, not 16. Both fixed in `jd_upstream_wedge_count`.
- **The container cases depend on NIST's website, and Java 8 cannot reach it.** RADDOSE-3D fetches
  NIST's attenuation tables on every run with a container. NIST refuses Java 8's requests (HTTP
  403), and the run then fails with a `NullPointerException`. JDK 25 is served. **Not fixed.** The
  fix is to vendor the tables in the Java as the port already does (`nist_elements.tsv`,
  `nist_mixtures.tsv`), which also makes runs reproducible offline. The corpus generator now
  refuses to run on a JVM other than the one the manifest records, which is how this was found.
- **The escape tolerances were not re-measured.** Items 2, 3 and 4 give the escape cases new random
  realisations. Every case still agrees with the Java within the existing tolerances, at worst
  2e-4 on a quantised quantity, but the self-sensitivity behind those tolerances was measured on
  the old realisations.
- **An NMR or EM model gives a dose for a cell of water.** Found reading 211 random PDB entries with
  Gemmi. `CoefCalcFromPDB` counts symmetry operators from `REMARK 290`, which NMR and EM models do
  not have. With no operators there is no macromolecule, and the placeholder 1 A cell is filled
  with solvent, so RADDOSE-3D reports a dose. 42 of the 211 entries were such models. **Not
  fixed** in the Java: it should refuse a file with no crystal cell. The port, which now takes
  the operators from the space group, refuses them.
- **With escape modelled, voxel doses depend on the JVM.** CI's Java 21 job failed #38's golden
  escape run: `Dose Threshold` was 1.05e-5 from the golden value, against a 1e-5 tolerance. x86-64
  Temurin 21 on macOS gives the same value, so it is the JVM. On the golden escape input, it and JDK
  25 on arm64 disagree on about half of the 120 voxels, by up to 1.5e-4, on `master` as much as on
  the fixes. Averaged fields agree to 5e-6. `Dose Threshold` is one voxel's dose and does not
  average. The escape fixes redrew the realisation, so it now lands on a voxel the two JVMs
  disagree on. #38 gives the fields that follow one voxel (`Max Dose`, `Dose Threshold`, and the
  three computed from them) a tolerance of 1e-3. **For the port:** the corpus was generated on JDK
  25 on arm64, and the port matches it on macOS. A corpus regenerated on another JVM could move
  escape cases' per-voxel values by up to 1.5e-4. Whether the port's tolerances absorb that has
  not been checked. The port also takes `cos` and `sin` from the platform's C library
  (`escape.py`), not from fdlibm as `StrictMath` does. It agrees with the Java on macOS but is
  untested on Linux.

---

## Why the two must change together

The port reproduces the Java on purpose, quirks included, and marks each quirk in its source. The
conformance corpus (`tests/conformance`) compares the two on 56 cases, value by value. That includes
escape modelling: the port draws from `java.util.Random` in the Java's order, so the same seed gives
the same dose.

So neither side can be fixed alone. A Java fix makes the regenerated corpus disagree with the port,
and a port fix makes the port disagree with the committed corpus. A fix lands as a Java PR plus one
pyRADDOSE-3D commit that carries both the port change and the corpus regenerated from the fixed
Java. RADDOSE-3D PRs #31–#35 were fixed in both this way.

### The procedure for each fix

1. **Show the defect first.** If no corpus case exercises it, add one and generate it from the
   *unfixed* Java. It passes, because the port reproduces the defect. Several defects below have
   no such case yet. Each section says so.
2. **Fix the Java,** with a JUnit test that fails before the fix. Run `ant test`.
3. **Regenerate the corpus** from a build of the fix branch (`tests/conformance/generate.py`).
   Review the diff: only the cases listed under the item should change. Any other change is a
   regression in the fix.
4. **Fix the port** to match. Delete its quirk marker and the item's bullet under *Upstream* in
   `ROADMAP.md`. Commit the port change with the regenerated corpus, naming the Java PR.
5. **Re-measure the escape tolerances** if the fix changes the random draws. Some fixes change how
   many draws are taken, or their bounds. Every draw after that point is then different, so the
   escape cases get a new random realisation. They still agree value by value, but their
   tolerances were measured on the old one, so re-measure them by the corpus's usual rule.
6. **Record the RADDOSE-3D commit.** `MANIFEST.yaml` records only the version string, `5.0.1083`,
   so add the commit hash, since several fixes will share that version.

---

## Summary

| # | defect | how much it matters | random draws change? | corpus cases that move |
|---|---|---|---|---|
| 1 | With PE escape alone, the atom's relaxation energy is deleted | **3.2% of the dose for insulin; 15.5% for a SeMet crystal above the Se K edge** | no | 100, 102–108 |
| 2 | FL goes only along +z | measured small: under 0.1% of the dose for SeMet above the edge, though FL escape removes 16% of it | only if the direction set changes | 101, 102, 103, 105 |
| 3 | The crystal's PEs use the surrounding's direction weights | probably small; not measured | yes | 103, 104, 107, 108 |
| 4 | The surrounding's attenuation is clamped to the wrong box | at most part of the surrounding's 9% share; not measured | no | 103, 104, 107, 108 |
| 5 | L-shell FL of every element from Mg to Co is NaN | energy small; prints `NaN` | no | 101, 102, 103, 105, slightly |
| 6 | Pink beams: surrounding ×200, wrong volume, fluences and coefficients | **surrounding dose ×200: whole-crystal dose 18× too high**, checked against the Java | no | 028, 029, 105, plus a new case |
| 7 | PE escape with `AbsCoefCalc Average` crashes | crash | — | none |
| 8 | `SAXSSEQ` ignores `NumCarb` | silent wrong answer | — | new case |
| 9 | A sequence without a `>` header fails misleadingly; `ISDNA`/`ISRNA` undocumented | confusing error; DNA silently read as protein | — | new case |
| 10 | `Type Experimental` can never work | crash with an unhelpful message | — | none |
| 11 | `PixelSize` means one thing in the user guide and another in the code | non-square pixels swapped | — | new case |
| 12 | Collimation on a measured beam ignores its sizes | silent wrong answer | — | 081 |
| 13 | `BeamExperimentalTest` tests nothing | no coverage | — | none |

Items 1, 2 and 6 change doses materially in normal use. Items 1 and 2 are physics changes, so
they need the group's sign-off: results published with RADDOSE-3D will move. Item 6 is plainly a
bug. The rest are bugs with only one reasonable fix.

---

## Escape modelling

### 1. With photoelectron escape alone, the atom's relaxation energy is deleted

**The physics.** An absorbed photon of energy *E* ejects a photoelectron with *E − B*, where *B* is
the binding energy of the shell it came from. The atom then relaxes and releases *B*, either as a
fluorescence photon, which can travel, or as Auger electrons, which deposit locally. Bookkeeping
that conserves energy therefore needs three parts:

- the photoelectron carries *E − B*;
- the fluorescence *F* is tracked if fluorescence escape is on, and deposited locally if not;
- the remainder, *B − F*, is deposited locally.

**What the Java computes.** These quantities are averaged over elements and shells, per absorbed
photon:

- the binding energy, *B* = Σ *p*ₛ*B*ₛ over K, L1–L3 and M1–M5 (`calculatePEEnergySubtraction`,
  `Crystal.java:684–711`);
- the fluorescence, *F* = Σ *p*ₛ*ω*ₛ*B*ₛ over K and L1–L3 (`calcFluorescence`, `Crystal.java:656–680`).
  This takes the edge energy as the photon's energy;
- the Auger energy, *A* = *p*ₖ(1 − *ω*ₖ)*B*ₖ, **K shell only** (`getAugerEnergy`,
  `Crystal.java:599–616`). The L terms are commented out.

The five branches of `exposeAngle` then use them as follows:

| branch | the PE carries | deposited locally | tracked as FL | energy unaccounted for |
|---|---|---|---|---|
| crystal, PE only (`:1186–1220`) | *E − B* | *A* | — | ***B − A* is lost**: the K fluorescence and all L and M relaxation |
| crystal, PE + FL (`:1141–1184`) | *E − F − A* | *A* | *F* | none, but the PE carries *B − A − F*, which belongs locally |
| crystal, FL only (`:1222–1236`) | not tracked | *E − F* | *F* | none |
| surrounding, PE only (`:1354–1356`) | *E − A* | — | — | the PE is over-credited by *B − A* |
| surrounding, PE + FL (`:1357–1366`) | *E − B* | — | not modelled | none that reaches the crystal |

**How much.** These are the energies as fractions of the absorbed energy, computed with the
Java's own arithmetic:

| crystal | lost with PE escape alone | over-credited to the PE with PE + FL |
|---|---|---|
| insulin, 12.1 keV | 3.2% | 0.13% |
| SeMet, 1 Se per 51-residue monomer, 12.7 keV, above the Se K edge | **15.5%** | 0.5% |
| the same, 12.5 keV, below the edge | 1.0% | 0.7% |
| a heavily substituted crystal, 8 Se per monomer, 12.7 keV | **40.8%** | 1.2% |

On a 100 µm crystal, where real escape is negligible, the PE-only whole-crystal dose falls by
exactly these amounts against no escape: −3.2%, −15.5% and −40.7%. Anyone who modelled
photoelectron escape alone for a SeMet crystal above the Se edge got a dose about 15% too low.
The cause is bookkeeping, not escape.

**Java fix,** in `Crystal.exposeAngle`, one rule for all five branches:

- the PE gets *D*(1 − *B*/*E*), where *D* is the voxel's photoelectric dose;
- fluorescence *F* is tracked when FL escape is on;
- *D*·(*B* − *F*ₜ)/*E* is deposited locally, where *F*ₜ is *F* if FL escape is on and 0 if not.
  This replaces `totAugerDose` in both crystal branches.
- In the surrounding, the PE gets *E*(1 − *B*/*E*) in both branches.
- For a pink beam, *E* should be the sample's energy. The code divides by
  `beam.getPhotonEnergy()`, the mean.

*F* still takes the edge energy as the photon's energy, though the emitted line is lower (Kα ≈
*B*ₖ − *B*ₗ). Under the rule above, energy is still conserved: the difference travels with the
fluorescence instead of staying local. All that remains wrong is *where* that energy lands.
Changing it is a physics decision; leave it for later.

**Port fix:** `backend/escape.py`, `Escape._crystal_step` (`:1081–1091`) and `_surrounding_step`
(`:1230–1236`).

**Corpus:** the draws are unchanged and only amounts move, in cases 100 and 102–108. Before fixing,
add a case that shows the size of the problem: a SeMet crystal above the Se K edge with PE escape
alone.

### 2. Fluorescence goes only along +z

`addDoseAfterFL` (`CrystalPolyhedron.java:2123–2177`) draws a random direction at `:2127` and never
uses it. The code that would have used it is commented out at `:2154–2158`. The loop over
directions runs `FL_ANGLE_RESOLUTION² = 1` time (`:162`), so it always uses track 0. That track is
θ = 0, which `findVoxelsReachedByFL` (`:2031–2080`) builds as (0, 0, *r*): straight along +z in the
crystal's voxel grid. All fluorescence from every voxel travels toward the crystal's +z face. How
much escapes then depends only on how far each voxel is from that face. A plate that is thin along
z overstates escape, and a needle along z understates it. For a SeMet crystal above the edge,
fluorescence is 15% of the absorbed energy, so this geometry error looked large.

**Measured after fixing, it is not.** SeMet at 12.7 keV, in the Java on `master` and on the fix,
with `--seed 20260917`: fluorescence escape removes about 16% of the dose, but sending it the
right way moves the whole-crystal dose by +0.01% for a 20 × 16 × 12 µm crystal and +0.08% for a
200 × 160 × 120 µm one, and the maximum dose by at most 0.18%. Se Kα mostly leaves crystals of
this size whichever way it goes. The effect should be larger where fluorescence is absorbed
within the crystal, at lower emission energies or in larger crystals; that is not measured.

**Java fix:**

- Use the drawn direction: `flRelativeVoxXYZ[n][l][m][randomTrack]`. Drop the `q` loop, or keep it
  at one iteration.
- Make the direction set isotropic. Fluorescence is emitted uniformly, so it needs no polarisation
  weighting and no rotation with the goniometer, but the current grid is uniform in θ and φ, which
  crowds directions toward the poles. Its south-pole de-duplication also tests `theta == π` after
  repeated floating-point additions, which may never hold exactly. Use an equal-area set, such as a
  Fibonacci sphere, or weight the grid by sin θ through a bias table, as the PE code weights its
  tracks.
- The PE direction grid (`findVoxelsReachedByPE`, `:1853`) has the same θ–φ layout. Check whether
  its polarisation weights already account for it.

**Port fix:** `Escape._fluorescence` (`escape.py:1138–1166`) uses the draw it already takes
(`draws[:, 0]`). `fluorescence_escape` builds the directions; today it only counts them.

**Corpus:** cases 101, 102, 103 and 105. The draws are unchanged if the number of directions stays
the same. A new direction set changes the bound of the draw, and so the realisation.

### 3. The crystal's photoelectrons use the surrounding's direction weights

`trackNumberBias` (`CrystalPolyhedron.java:177`) is one field. It holds the table that weights
photoelectron directions by the beam's polarisation. `findVoxelsReachedByPE` writes it at `:1974`.
`exposeAngle` calls that method for the crystal and then for the surrounding (`Crystal.java:1022–1027`),
so the surrounding's table wins. Both `addDoseAfterPE` (`:2093`) and `addDoseAfterPECryo` (`:2186`)
then draw from it. The two tables cover the same directions and differ only in their weights,
through the mix of shells each material ionises, so the effect is probably small. It has not been
measured.

**Java fix:**

- Keep two fields, `trackNumberBias` and `cryoTrackNumberBias`, each written by its own call.
- Build both once per energy rather than at every angle. The weights do not depend on the angle;
  only the rotated offsets do.
- Build the table with a growable list. That removes the "messy fix" that sizes it ×1.5 (`:1876`)
  and the scan for trailing zeros (`:1956–1972`). If the array is ever exactly full, that scan
  leaves the table empty, and `nextInt(0)` throws.

**Port fix:** `Escape._bias` (`escape.py:1010–1021`) returns the crystal's table for crystal voxels
and the surrounding's for surrounding voxels.

**Corpus:** the crystal's draws now have a different bound whenever the surrounding is on, giving
a new realisation in 103, 104, 107 and 108.

### 4. The surrounding's attenuation is clamped to the wrong box

`findDepth` returns zero for any point outside the crystal. So to estimate how much crystal the
beam crossed before reaching a surrounding voxel, `exposeAngle` clamps that voxel's position onto
the crystal's bounding box first (`Crystal.java:1318–1330`). That box, `minimumDimensions` and
`maximumDimensions`, is taken at `CrystalPolyhedron.java:571–574`. That is *before* the AngleP and
AngleL rotations, and it never turns with the goniometer. The point being clamped, though, has
already been rotated to this angle and translated. At any angle but zero the clamp therefore
compares a rotated point with an unrotated box.

**Java fix.** The proper fix is to compute the crystal path *upstream of the surrounding voxel
itself*:

- add a depth routine that sums the segments inside the crystal between sorted ray–surface
  intersections, which is valid for points outside the crystal too;
- drop the clamp.

Voxels upstream of the crystal then see no crystal attenuation, and those downstream see the full
chord. Keep `findDepth` unchanged for crystal voxels, because every deterministic corpus case
depends on its exact arithmetic.

A minimal alternative is to clamp against the box of the crystal as rotated at this angle. That is
cheaper, but still an approximation.

**Port fix:** add the new routine beside `geometry.find_depth`. Remove the clamp in
`escape.py:1211–1212` and the box computed for it in `solve.py:951`.

**Corpus:** draws unchanged; cases 103, 104, 107 and 108. The surrounding adds 8.9% to the
whole-crystal dose in case 103, shortened to a 10° wedge. The error can only be part of that
share. It has not been measured, but should be largest for rotated or thin crystals.

### 5. The L-shell fluorescence of every element from Mg to Co is NaN

`getFluorescentEscapeFactors` evaluates the crystal's absorption at each shell's mean emission
energy (`CoefCalcCompute.java:840`, `:857`, `:874`). For Z = 12–27, Mg to Co, the element table gives
an L fluorescence yield but no L emission energy: it is 0 keV. The absorption at 0 keV is NaN, so
these shells' escape distances are NaN (`CrystalPolyhedron.java:1236–1280`). `StrictMath.round(NaN)`
is 0, so their fluorescence is aimed at voxel (0, 0, 0). In every case checked, that voxel is
outside the crystal, so the energy is counted as escaped and lost.

The run then prints `Total energy that may escape by Fluorescent Escape: NaN J.`
(`Crystal.java:987–993`; confirmed by running case 101). That label is wrong as well: the total
is a sum of voxel doses in MGy, not joules.

The roadmap said "phosphorus and sulphur". It is every element from Mg to Co, which includes Cl,
Ca, Mn and Fe, all common in MX crystals. The energy involved is small, because L fluorescence
yields below Z = 28 are under 0.007, but the printed total is NaN for any crystal that has one of
these elements and fluorescence escape on.

**Java fix:**

- Best: tabulate the L emission energies for Z = 12–27.
- Otherwise, treat a shell with no tabulated emission energy as relaxing locally: zero its yield
  for escape purposes.
- Either way, assert that no escape distance is NaN, and correct the printed units.

**Port fix:** apply the same rule in `escape_factors`. Separately, and now, see *Port-only issues*
below: the port currently aims these NaNs at a different voxel from the Java.

**Corpus:** cases 101, 102, 103 and 105 move by about 10⁻⁶, probably within tolerance. Add a case
with Fe or Ca and fluorescence escape so the fix shows in the diff.

---

## Pink beams

### 6. Four defects in how pink beams are summed

A pink beam is sampled at 200 energies. For each angle, `expose` loops over the samples and calls
`exposeAngle` once per sample (`Crystal.java:925–960`), giving each sample 1/200 of the angle's
exposure. Four things do not account for that:

| | defect | where | effect |
|---|---|---|---|
| a | The surrounding's exposure is not divided over the samples | `Crystal.java:1283–1284`. The crystal's equivalent at `:1065–1066` divides by `energiesPerAngle`. | **The surrounding contributes 200 times its dose.** Case 103 on a 10° wedge, run in the Java: 17.4 MGy whole-crystal average with a monochromatic beam (16.0 without the surrounding), and 306.8 MGy with `EnergyFWHM 0.3`. That is 18× too high. |
| b | Each voxel is observed once per sample, and `ExposureSummary` counts every observation | `ExposureSummary.java:256`, `:286` | the per-image exposed volume in `DWDs.csv` is 200× too large |
| c | `imageFluences` is overwritten, not added to | `ExposureSummary.java:235` | `VoxelFluences.csv` holds the last sample's fluence, 1/200 of the image's |
| d | the printed coefficients come from whatever the CoefCalc last held | `Crystal.java:931` | `Summary.txt` reports the coefficients at the *highest* energy sampled, mean + 1.1 × FWHM (12.65 keV for 12.1 ± 0.5) |

Three more things use the mean energy for every sample. Two are deliberate, as the comments at
`:935–940` say: the escape factors and the container attenuation. Leave those. The third is the
binding fraction in item 1, which should use the sample's energy.

**Java fix:**

- Divide the surrounding's factor by `energiesPerAngle`.
- In `ExposureSummary`, accumulate each voxel's fluence over an image, and count a voxel as exposed
  once per image, when its fluence first becomes positive. Keep a separate per-observation count
  for the averages that divide by it (`averageRDE`).
- Report the coefficients at the mean energy and say so.

**A test for both codebases.** Without escape modelling, a pink beam with a negligible spread
(`EnergyFWHM 1e-6`) must reproduce the monochromatic result in every output file, to rounding.
Today it fails on (b) and (c). Item (a) needs escape on, and then a pink beam takes 200 times as
many random draws, so it cannot match exactly. Check instead that the surrounding adds the same
share of the dose as it does for the monochromatic beam.

**Port fix:**

- `escape.py:1213`: the attenuation factor;
- `backend/exposure.py`, `_Recorder` (`:550` on);
- `outputs.image_series`, the volume column (`outputs.py:227`);
- `solve.py:1040`: the reported coefficients.

**Corpus:** draws unchanged; cases 028, 029 and 105. No case combines a pink beam with a
surrounding. The port reproduced (a) from reading the code, and the figures above were then checked
against the Java. Add such a case, with a short wedge, before fixing. It is slow: 200 surrounding
passes per angle, about 4 minutes for 10° in the port.

---

## Input handling

### 7. Photoelectron escape with `AbsCoefCalc Average` crashes

`CoefCalcAverage.getPresentElements` returns null (`CoefCalcAverage.java:149–151`). The
photoelectron set-up iterates over it (`CrystalPolyhedron.java:1997`), so the run fails with a
`NullPointerException`.

Fluorescence escape alone does not crash. It prints a warning and models nothing
(`CoefCalcAverage.java:105–115`).

**Java fix:** refuse photoelectron escape with `Average` when the input is read, saying why. The
port already does (`solve.py:873`). For fluorescence alone, it is a choice between keeping the
warning and refusing. RADDOSE-3D PR #34 already refuses input it would otherwise ignore, such as
`SolventHeavyConc` with `SmallMole`, so refuse this too.

**Port fix:** match, if fluorescence alone becomes an error (`solve.py:878`).

### 8. `AbsCoefCalc SAXSSEQ` ignores `NumCarb`

`CoefCalcFromSequenceSAXS` calls `addCarbs(numCarb)` (`:154`), but `addCarbs` adds
`getNumCarb()` carbohydrate, and nothing on this path ever calls `setNumCarb`. So the count stays
zero, and `NumCarb` is read and then has no effect. `CoefCalcFromSequence` calls it at `:175`.

**Java fix:**

- Call `setNumCarb(numCarb)` before the solvent-fraction estimate.
- Count the carbohydrate's mass in the monomer weight that sets the monomer count. The constructor
  computes that count (`calculateNumMonomers`) *before* `calculateHeavyAtomOccurrences`, so the
  mass has to be added earlier.
- Decide which mass to use. `CoefCalcSAXS` weighs a carbohydrate residue at 180 Da (glucose); the
  solvent estimate uses 162 Da (a glycosyl residue). Use 180 here so the two SAXS routes agree, and
  treat the 162/180 split as a separate question.

**Port fix:**

- add `num_carb` to `SaxsSequenceComposition` (`models/composition.py:160`);
- pass it to `saxs_sequence_monomer_count` and `sequence_composition`;
- drop the reader's warning (`_raddose_txt.py:839`).

**Corpus:** add a case with `NumCarb`.

### 9. A sequence without a header fails misleadingly, and the type markers are undocumented

`CoefCalcFromSequence.residueType` is an `int` field that is never initialised, so it starts at 0
(`:25`). Only a `>` header line sets it. Residues before any header are looked up as type 0, which
`ResidueDatabase.idLetterConversion` does not know (`:809`). The run fails, calling the first residue
"not a recognised protein residue code", though it may be a perfectly good one.

The user guide says only that sequence files are FASTA (`doc/user-guide.tex:405`). Nothing tells a
user that a DNA or RNA chain needs `ISDNA` or `ISRNA` in its header. An ordinary header like
`>chain B, DNA` is read as protein, and since A, C, G and T are valid amino-acid codes, nothing
fails.

**Java fix:**

- Initialise `residueType` to `TYPE_PROTEIN` (decision A below), or refuse with a message that
  names the real cause.
- Document `ISDNA` and `ISRNA` in the `SEQFILE` section of the user guide.
- Warn when a chain read as protein consists only of A, C, G, T and U, since it is probably a
  nucleic acid whose marker is missing.

**Port fix:** `parse_sequence` (`backend/composition.py:571–626`) follows whichever is chosen.

### 10. `Type Experimental` can never work

`BeamFactory` maps `Experimental` to `BeamExperimental` (`BeamFactory.java:37–38`). `ClassFactory`
needs a constructor that takes the property map, which that class lacks (`ClassFactory.java:84–90`).
The run fails with "does not have a property constructor". Only `ExperimentalPGM` is documented
(`user-guide.tex:666`).

**Java fix:** make `BeamFactory` refuse `Experimental` with a message naming `ExperimentalPGM`.
Making it an alias would also work, but PGM is the only format there is.

**Port fix:** none needed; the reader already refuses it with that reason.

### 11. `PixelSize` means different things in the user guide and the code

The user guide says `PixelSize X Y` gives "the horizontal and vertical size of the pixels … in the
.pgm file respectively" (`user-guide.tex:711`). In the code, X goes to the beam's x axis
(`BeamExperimentalpgm.java:40`). The image is stored transposed (`:109–115`), so beam x runs down
the image, *vertically*.

The worked example `examples/Experimental_beam_part1.md:7` disagrees with both: it calls the first
number the y size. For square pixels none of this matters. For non-square pixels, any input
written from the user guide has them swapped.

**Fix:** keep the code. Its convention matches `FWHM` and `Collimation`, where X is RADDOSE-3D's
x, vertical for a horizontal goniometer. Correct the user guide and the example instead. Also
document the image's orientation: its horizontal axis lies along the goniometer, y, and its
vertical axis along x, both reversed. The port's model docstring and its notebook already describe
it that way.

**Corpus:** add a case with non-square pixels, so that a change to either the code or the port
shows up.

### 12. Collimation on a measured beam ignores its sizes

`BeamExperimental` never reads the collimation sizes. `Collimation Circular` only sets a flag
(`BeamExperimentalpgm.java:48`), which clips the profile to the ellipse inscribed in the *image*
(`BeamExperimental.java:111–125`), whatever sizes were given. `Rectangular`, `Horizontal` and
`Vertical` have no effect at all.

The user guide says collimation "delimits where the beam has non-zero intensity"
(`user-guide.tex:718`), with no exception for measured beams. It also says the flux to give is the
flux "after collimation" (`:672`).

**Java fix:** apply the aperture with its given sizes, rectangular or elliptical, as `BeamGaussian`
does. Normalise the flux over the open part of the profile, as the guide's definition of flux
requires. This changes results for existing inputs that combine `ExperimentalPGM` with a
`Collimation` line, so note it in the release.

**Port fix:** decisions C and D below.

---

## Tests

### 13. `BeamExperimentalTest` tests nothing

It never calls `applyContainerAttenuation`, so `attenuatedFlux` stays 0 and every intensity is 0
(`BeamExperimentalTest.java:49–60` and the tests after it). Its assertions are one-sided,
`intensity - expected < tolerance`, and 0 satisfies all of them. Its expected values also leave
out `KEVTOJOULES × energy`.

**Java fix:**

- Apply a `ContainerTransparent` before `generateBeamArray`.
- Include `KEVTOJOULES × energy` in the expected values.
- Use two-sided `assertEquals(expected, actual, tolerance)`.

**Port:** nothing to change. `tests/backend/test_beam.py` checks the same points exactly.

---

## The port's four interim decisions

Each of these was the port's answer to one of the Java defects above until its Java fix landed.
**All four are now settled as recommended**: A and B with the input fixes, C and D with item 12.
The text below records the reasoning.

### A. A sequence file whose residues come before any header is refused

**Now:** `parse_sequence` raises, naming the line. Before Milestone 4, the port silently read
everything as protein. The Java fails with the misleading message in item 9.

**Options:**

1. Default to protein in both codebases. A plain sequence with no header is almost always a
   protein, and this is what most users mean.
2. Refuse in both, with the port's message. This is strictly correct for FASTA, which the guide
   names.

**Recommendation:** option 1, plus the nucleic-acid warning from item 9. It accepts what users
actually write, and the warning catches the one case where it would be wrong.

### B. `NumCarb` with `SAXSSEQ` is accepted with a warning and ignored

**Now:** the reader warns that RADDOSE-3D ignores it, and the YAML model has no `num_carb` field,
so the result matches the Java's.

**Options:** keep warning until item 8 is fixed, or refuse it now.

**Recommendation:** keep the warning. It gives the Java's answer and says it is wrong. When item 8
lands, add `num_carb` and drop the warning.

### C. Collimation other than circular on a measured beam is dropped with a warning

**Now:**

- The reader drops `Rectangular`, `Horizontal` and `Vertical` on an `ExperimentalPGM` beam, with a
  warning.
- The YAML model refuses any collimation and offers `circular: true` for the inscribed ellipse.

This matches what the Java computes.

**Options:**

1. Follow item 12's fix, so the aperture applies with its sizes. The model gets an ordinary
   `collimation` field like the other beams, and both the warning and the `circular` flag go.
2. If the Java keeps today's behaviour instead, refuse non-circular collimation in both codebases,
   since it does nothing.

**Recommendation:** option 1. It is what the user guide promises.

### D. The `.txt` writer emits `Collimation Circular 0 0` for a circular measured beam

**Now:** the grammar requires two sizes after `Circular`, and the Java ignores them, so the writer
puts `0 0` and a comment.

**Under C option 1** this goes away: the model holds real sizes and the writer emits them. Case 081
then needs real sizes, because `0 0` would clip everything.

**Under C option 2** keep it. Teaching the grammar to accept `Circular` with no sizes would mean
regenerating the ANTLR 3 parser, which is not worth it for this.

---

## Port-only issues found while writing this

Both are resolved: item 5 removed the NaN, and item 2 replaced the code that aimed it; the
misdescriptions were corrected when this document was written.

- **NaN fluorescence goes to a different voxel from the Java's.** The Java computes the offsets
  for fluorescence track 0 as (*r* sin 0, *r* sin 0, *r*). With *r* = NaN, that is NaN in all three
  axes, so the dose goes to voxel (0, 0, 0). The port writes the x and y offsets as `+ 0.0`
  (`escape.py:1159`), so it sends the dose to (*i*, *j*, 0) instead. In every corpus case both
  voxels are outside the crystal, so nothing differs, but that is not guaranteed. Compute the
  offsets as *r* times each direction component, so NaN propagates as it does in Java. Item 5's
  fix removes the NaN either way.
- **The roadmap and `escape.py`'s docstring misdescribe three things:**
  - They say the PE + FL branch "skips the binding-energy subtraction". In fact it subtracts
    *F + A*, and the *PE-only* branch is the one that loses energy (item 1).
  - They say "phosphorus and sulphur" where it is every element from Mg to Co (item 5).
  - They say "the summary's volume" where it is `DWDs.csv`'s per-image volume (item 6b).

  The roadmap also omits the surrounding's own version of item 1. Corrected alongside this
  document.

---

## Older upstream items

These were found before Milestone 4. The first three are now fixed: the first two in
`jd_upstream_input_fixes`, the third in `jd_upstream_wedge_count`. The last two remain open.

| item | where | fix |
|---|---|---|
| `-o voxelfluences:…` never matches: `"voxeldose"` is tested twice | `OutputFactory.java:59–62` | **fixed** |
| the help text advertises `Summary`, `SummaryCSV:…` and `DoseStateCSV`, which are not modules | `RD3D.java:399–401` | **fixed** |
| a 0–30° wedge at 2° per image takes 15 images, not 16: the count is computed in radians and truncated | `Crystal.java:891–893` | **fixed**, with the reversed wedge's count and step; only case 066 moved, plus new case 067 |
| characters the lexer does not know are skipped with a message, not refused | the generated parser | refuse, as #34 did for parser errors. The port warns today. |
| `MicroED.getSolventFraction()`, and the `getBeamArea()`/`getExposure()` stubs | MicroED, `BeamGaussian.java:298–306` | MicroED now refuses any beam but TopHat, so the stubs are never read: see *MicroED* below. `getSolventFraction` is read only by MicroED's disabled charge model. |

---

## MicroED: found porting it

Found porting `Subprogram EMED` (`backend/microed.py`), against RADDOSE-3D `master` at
`167fb7a`. "Run" means seen in a Java run; "read" means found in the source. All but the last
two are fixed in RADDOSE-3D by [#40](https://github.com/GarmanGroup/RADDOSE-3D/pull/40), merged as `e2df6e4`,
and the corpus follows that commit.

| defect | where | effect | fix, in both |
|---|---|---|---|
| Every subprogram crashes in the summary writer once its own files are written | `Crystal.expose` returns without exposing the wedge; `OutputSummaryCSV.publishWedge` reads a null elastic yield | the run exits with an error, after a half-written `Summary.csv` (run) | `Experiment` publishes no wedge for a subprogram; the port writes MicroED's two files and nothing else |
| The PDB route never multiplies its electron-scattering atoms by the molecules in the cell, and gives them no ligands | `CoefCalcFromPDB:381–394`; `multiplyAtomsEM` is never called | the elastic mean free path is far too long -- 45% for 3I40 -- and the counts and grid searches with it (run) | the electron physics reads the one composition; cases 128 and 129 |
| The sequence, SAXS-sequence and CIF routes give the electron physics no macromolecule at all | no `...OccurrenceEM` calls in those classes | as above, worse (run) | as above; cases 130 and 131 |
| The distant-collision integral mixes two shell models | `integrateDist`, `getpDisW` read `getShellBinding(i)`, whose index counts shells (K, L1, M1), so the L2 integral started at the M1 edge | the inelastic mean free path is 7.5% shorter than the subshell model gives: 234.20 nm for insulin at 300 keV, where it is 253.15 (run) | both read `getShellBindingSubshell`. A physics change, merged in #40. |
| Eight elements have a zero binding energy for a subshell the GOS model reads, and aluminium a blank one | `MuCalcConstants.txt`; `getShellBindingSubshell` | Na, Ca, Sc, Ti, V, Cr, Ga, Ge: a NaN inelastic mean free path, so NaN counts and grid searches that end on 1990 kV and 1995 nm (run, Na). Al: `NullPointerException` (read) | the table has them now: Na L2 and L3, Ca–V M2 and M3, Cr M2 and M3, and Ga and Ge M2–M5 from the X-ray Data Booklet (Table 1-1), as the table's neighbouring M1 binding energies are; Al 3s and Cr 3d, which it does not list, from Carlson (1975) as Geant4 tabulates them. Should a table still lack one, a warning, NaN for everything the inelastic model feeds, and no search. Cases 127 and 132. A physics change, merged in #40. |
| A Gaussian beam gives MicroED no exposure | `BeamGaussian.getExposure` is a stub returning 0 | a dose of zero, NaN, 1990 kV and 1995 nm (run) | refused, as is any beam but TopHat |
| An experimental beam has no size | `BeamExperimental.getBeamX` returns null | a crash (read) | refused |
| A TopHat beam without `Exposure` | `BeamTophat.getExposure` unboxes null | `NullPointerException` (read) | refused |
| A crystal thinner than 5 nm has no slices | `EMStoppingPowerWay` | a dose of 0/0, NaN (read) | one slice; case 133 |
| The slice file's depth column assumes 5 nm slices | `MicroED.CalculateEM` | a thickness not a multiple of 5 nm is cut into wider slices than the depths say (read) | the depths step by the real width; cases 122 and 124 |
| MicroED's class cannot load without Selenium, which only its commented-out NIST lookup uses | `MicroED.java:14–25` | Selenium and its dependencies, 9 MB of the 53 MB jar, are needed to run MicroED at all (run) | the lookup and the library are gone |
| Elements from Z = 35 have no inner subshells in the GOS model | `getNumValenceElectronsSubshells` | every electron counted as valence (read) | open; the port reproduces it |
| A third `Dimensions` value makes a cylinder a cuboid | `Crystal:310–328` | MicroED takes 0.001 nm off a cylinder's thickness only when it has two dimensions (run) | open; the port always takes it off |

An earlier version of this table said the CIF route counts its molecular weight twice. It does
not: `CoefCalcFromCIF` calls `calculateDensity` once.

With the shell models made one, the port's inelastic mean free path agrees with the Java's to
2 ulp. The 3e-9 it differed by before came from the mixing: whether an integral took its last
point turned on the last bit of an energy computed from the other shell model.

## Monte Carlo and XFEL: found surveying them

From reading `MC.java` (`Subprogram MONTECARLO` and `GOS`) and `XFEL.java` on `master` at
`e2df6e4`, ahead of porting them (`ROADMAP.md`, Milestone 7). XFEL is a near copy of MC: 76 of
their methods are identical. "MC:n" and "XFEL:n" are line numbers there. The port reproduces the
physics defects until they are fixed in RADDOSE-3D, and is compared with it statistically.

### Reproducibility and running: fixed first, in [#42](https://github.com/GarmanGroup/RADDOSE-3D/pull/42)

Merged as `82ac582`. Every fix but the last two changes no number,
only which realisation a seed gives. Forty runs a side on `master` and the branch agreed for each
mode.

| defect | where | fix |
|---|---|---|
| The element a random number picks depends on the JVM. `Element` and `ElementEM` have no `hashCode`, and elements are chosen by walking `HashMap`s built from `HashSet`s, in identity-hash order. The same seed gave 50.8 and 53.6 MGy under two JVM hash modes (run) | `CoefCalcCompute` 647–683, 2125, 2805, 5269 | elements in atomic-number order; `hashCode` = Z |
| A circular Gaussian beam with unequal FWHM samples from a clock-seeded generator, so `--seed` cannot repeat it | MC:2850, XFEL:2890 | two seeded normals, the same distribution |
| `SimPhotons 1e6` and `1000000.0` are refused with an uncaught `NumberFormatException`, and so is any `Runs` not written as an integer | `Inputfile.g` 630, 640 | whole numbers in any notation |
| The overvoltage tables are re-read from disk for every electron, and the Sternheimer factor searched for again after every inelastic event | MC:1558, XFEL:1617; `getSturnheimera` | read once; cached, since it depends on the composition alone |
| XFEL allocates three 4-D arrays of 15,300 time bins before it knows the pulse: 719 MB for a 20 um crystal, and an `ArrayIndexOutOfBoundsException` for a 1 fs pulse through more than about 46 um | XFEL:188–192 | sized from the pulse |
| An experimental beam, a beam above 20 keV with an element beyond Z = 52, an XFEL beam without `PulseEnergy`, and the average composition each crash partway through | MC:865, 3338; XFEL:268 | refused before the run |
| The CSV files use the default locale, so a German one writes `57,305000` | `WriterFile` | `Locale.ROOT` |
| The stopping power sums each element's charge into a `long`, truncating as it goes, so a count one ulp below a whole number loses an electron or not depending on element order. Making the order fixed moved MicroED case 129 by 6e-7, against the port (run) | `calcStoppingPower` | a `double`. MicroED's doses move by up to 8e-5 (case 123, where electrons stop) |
| XFEL keeps the inside/outside answer for the first point tested in each voxel. Electrons start inside, so voxels the surface cuts through lean inside: on a 5 um cylinder the dose came out about 6% high, t = 1.9 over 20 runs each; no effect on a cuboid (run) | XFEL:3828–3875 | tested at each point, as MC does, at no measurable cost |

### A photoelectron of negative energy: fixed in [#43](https://github.com/GarmanGroup/RADDOSE-3D/pull/43)

Found porting the simple Monte Carlo. Branch `jd_mc_outer_shell` at `cc2afb6`.

| defect | where | fix |
|---|---|---|
| A photoelectric draw past every shell's running chance took the K shell, even below the K edge, where it freed a photoelectron of negative energy. The running chances fall short of 1 only where the model leaves shells out: sodium's L shells (8.4% of its absorption), and below their K edges iodine, xenon, samarium, gadolinium, and tantalum to uranium where the tables give M shells. Iodine at 12.1 keV gave 22% of its photoelectrons -21 keV; platinum's would be -66 keV. The K binding energy was then deposited where the atom relaxed, so energy balanced, but none of it travelled | MC:1156, XFEL:1190 | an outer shell: no binding energy, a photoelectron with the photon's whole energy, no relaxation. A 5 um crystal soaked in 1 M iodide loses 9% of its Monte Carlo dose (run); every element of the corpus's cases has shells for every draw, and their runs are unchanged to the last digit |

### Found porting the simple Monte Carlo

Reproduced by the port, to fix with the rest of the physics below:

- **Photons crossing a surrounding are measured from its front face** with the crystal's
  coefficients once past the surrounding's own, and one that lands in the surrounding is taken
  to have crossed the crystal and is drawn again from the crystal's back face (MC:569–575,
  631–657).
- **An electron coming back into the crystal is moved to the nearest face plane ahead**, not the
  face it crosses, and loses its energy entirely if it falls below 0.05 keV on the way
  (MC:1665–1679). One leaving into the surrounding with under 0.05 keV loses it too
  (MC:2258–2263).
- **The surrounding is the crystal's mesh with every vertex pushed out along each axis**, so a
  cylinder's or sphere's is not a uniform shell (MC:257–271). Its thickness is capped photon by
  photon, as a running minimum, so under a pink beam it depends on the order photons come in
  (MC:443–447).
- **`outputIonsPerAtom.CSV` lists elements in hash-bucket order** (sulfur first), and elements
  sharing a bucket in the order they were first ionised.
- **In simple mode, `outputXFEL.CSV`'s doses hold only the binding energies of atoms that do not
  relax** and the energy of electrons freed with under 0.05 keV (MC:1359–1394, 1631–1640), so
  its diffraction efficiency is the elastic count over that.

### Physics: to fix in both, with the group's sign-off

What some of these do to the simple Monte Carlo's dose, measured by correcting
each in the port alone (2.4 million photons, 10 runs a side), on crystals of a
few um that electrons escape from; in a 20 um crystal every effect is smaller:

| quirk | case | dose with it corrected |
|---|---|---|
| escape distances through the unrotated planes | 145, a 4 x 6 x 8 um crystal rotated 90 degrees | -11.4% +- 0.8% |
| elastic angles without their sin theta weight | 141, a 5 um cube | +3.4% +- 0.8% |
| "isotropic" photoelectron directions | 141 | -0.9% +- 0.8% |
| `getNewDirectionVector` | 141 | -0.4% +- 1.0% |
| the elastic photon chance | 141 | +0.4%, from the coefficients |
| circular exposed areas 4x too large | 144, a 5 um circular top-hat | exposed-region doses +27% (21% low as they are) |

Angles:
- **"Isotropic" directions are not**: the polar angle is uniform on [0, 2π) (MC:1242, 1306).
- **Compton scattering angles are uniform** (MC:1445), not Klein–Nishina.
- **The elastic angle is sampled without its sin θ weight**; above 20 keV the CDF ignores the
  table's uneven bins and maps index 146 onwards one bin low (`returnDeflectionAngle`, MC:3336).
- **`getNewDirectionVector` deflects by the wrong angle**: `AN = -(x/y)` and `V1 = AN·sin`,
  where Joy's formula has `-(cx/cz)` and `AM·sin`. Renormalising hides it: a requested 30°
  comes out between 6° and 35° (MC:2550).
- **The polarised emission direction is rotated by 2π − angle** though tracking is in the lab
  frame (MC:1209).

Energy accounting:
- **Auger and fluorescence cascades double-count hole energy**: the Auger branch deposits
  Eb − E_A and then recurses on both holes, which deposit their own binding energies (MC:1292,
  1323; XFEL:1370).
- **Phosphorus never relaxes**: Z = 15 is missing from the eligibility list (MC:1275).
- **MC looks fluorescence up in the Auger table**: `getTransitionIndex` ignores its flag
  (MC:1395). XFEL does not.
- **The elastic photon probability divides by photoelectric plus Compton**, not the total
  (MC:411).
- **Auger electrons below 50 eV deposit nothing** (MC:1629).
- **GOS mean free paths are inconsistent**: `1/gosOuterLambda` against `gosOuterLambda`, and an
  inner-shell λ never updated from its starting value (MC:1576, 2160, 2187).

Geometry and beams:
- **Escape distances use unrotated face normals** (MC:4131), and XFEL's occupancy test does too.
- **A circular top-hat bounds y by |x|** (MC:2771); a circular Gaussian draws its radius from a
  one-dimensional normal, truncated at the x aperture alone, so an ellipse narrower in y is
  overrun (MC:2853).
- **A wedge ending at 0° is a still**, whatever its start (MC:382).
- **A crystal face at exactly z = 0 is taken for a miss**: 0.0 is `getStartingZ`'s sentinel
  (MC:530, 4529).
- **Circular exposed areas are 4× too large**: π·x·y for diameters x and y (MC:2671).
- **Rotation runs the wrong way when end < start**, and XFEL truncates the angle to whole
  radians (XFEL:368).
- **The inverse transform is applied in the wrong order** when there are both offsets and
  rotation (MC:3914): a point is tested against the crystal translated then rotated, but its
  voxel is found by translating back before rotating back. The two differ by R(-a)o - o, so a
  point near a face lands outside the voxel grid, and the run fails with an
  `ArrayIndexOutOfBoundsException` (run: a 4 x 6 x 8 um crystal offset 0.5 um and rotated
  through 90 degrees). The port refuses the combination.

XFEL:
- The pulse length is rounded to whole femtoseconds, so a pulse under 0.5 fs divides by zero.
- Coefficients are left at the last photon's energy for the diffraction efficiency, which matters
  for a pink beam.
- Lifetimes are fixed delays, not sampled.
- Two random numbers are drawn and not used in fluorescence.

Refuted: **the stopping power does not turn negative above 50 eV**. The Joy–Luo correction is
applied twice, which keeps it positive.

## Order of work

Done as below, in this order.

1. **Add the missing cases first:**
   - a pink beam with a surrounding (item 6a);
   - a SeMet crystal above the Se K edge with PE escape alone (item 1);
   - Fe or Ca with fluorescence escape (item 5);
   - a measured beam with non-square pixels (item 11).

   Generate them from today's Java. The port passes them, and the corpus diff only adds cases.
2. **Java PR: input, messages and tests.**
   - Items 7, 8, 9, 10 and 13.
   - The `OutputFactory` and help-text items.
   - The user-guide corrections from items 9 and 11.

   These touch no existing corpus expectations. Decisions A and B are settled here.
3. **Java PR: escape.**
   - Item 1 first, then items 5, 2, 3 and 4, one commit each.
   - Regenerate the corpus after each, so every change in dose can be reviewed on its own.
   - Item 1 changes published-scale doses for heavy-atom crystals, so it needs sign-off from the
     group before merging.
4. **Java PR: pink beams.** Item 6, with the negligible-spread test.
5. **Decide item 12.** Then fix it with decisions C and D.
6. **The wedge image count,** on its own, because it moves the most cases.

The port follows each PR in one commit that carries the regenerated corpus. During development,
build the fix branch and regenerate the corpus against it locally. The port's escape changes can
then be checked value by value before the Java PR merges.
