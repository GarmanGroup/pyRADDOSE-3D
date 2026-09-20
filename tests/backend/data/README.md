# Element cross-section reference

`element_cross_sections.tsv.gz` is the output of RADDOSE-3D's `Element.java`
for every tabulated element at 20 energies spanning the MX range, printed at
`%.17g` so the comparison is limited by the calculation and not by formatting.

It is the reference for `tests/backend/test_elements.py`, which asserts
agreement to 1e-9 relative. The measured worst case is 3.7e-15 -- the port is
faithful to machine precision, so the assertion has six orders of magnitude of
headroom and a real divergence will stand out immediately rather than sitting
just inside a loose tolerance.

## Regenerating

`DumpXS.java` is the generator, kept here so the file is reproducible rather
than merely trusted. From a RADDOSE-3D checkout:

```
export JAVA_HOME=/path/to/jdk
javac -d /tmp/xs src/se/raddo/raddose3D/Element.java \
                 src/se/raddo/raddose3D/ElementDatabase.java DumpXS.java
java -cp /tmp/xs:. se.raddo.raddose3D.DumpXS | gzip -9 > element_cross_sections.tsv.gz
```

Run from the RADDOSE-3D root, so that `constants/MuCalcConstants.txt` resolves.

| | |
|---|---|
| Generated from | RADDOSE-3D `7a42ab2` |
| JVM | OpenJDK 25 (Homebrew), macOS arm64 |
| Rows | 1720 (86 elements x 20 energies) |

Regenerating must be a reviewable diff. A change here is a claim that the
Java's behaviour changed on purpose.

All three CoefCalc fixtures were regenerated when RADDOSE-3D adopted the
Fischer et al. (2004) protein density in place of a flat 1.35 g/cm^3. Every
case with a computed solvent fraction moved -- for the baseline, 0.70889 to
0.73058 -- while cases stating `SolventFraction` explicitly, and the
small-molecule cases which have no solvent model, are unchanged.


---

# CoefCalc reference

`coefficients.tsv` is the output of `DumpCoef.java`: six compositions built
through `CoefCalcFromParams`, each dumped as

| tag | contents |
|---|---|
| `CASE` | cell edges and angles |
| `CELLVOL` | unit cell volume, cubic Angstroms |
| `DENSITY` | g/cm^3 |
| `SOLFRAC` | solvent fraction |
| `ATOMS` | one row per element: Z, symbol, atoms per unit cell |
| `COEF` | energy, then absorption / attenuation / elastic / inelastic, 1/um |

Committed uncompressed: it is small, and a reference whose diffs are meant to
be reviewed should be readable in one.

The `ATOMS` rows are an input to the Python test, not an expectation --
building a composition from residue counts is a later module. Given the same
atoms, the test checks that the same coefficients come out. `SOLFRAC` is
recorded for when that module lands and is not yet asserted.

Regenerate as for the element sweep, substituting `DumpCoef`.


---

# Beam reference

`beams.tsv` is the output of `DumpBeam.java`: four Gaussian beams and two
tophats, probed at points chosen to sit either side of every edge in the
model -- just inside and just outside a rectangular aperture, on and off the
axis of an elliptical one, and well outside both.

| tag | contents |
|---|---|
| `PARAMS` | the flux and energy every beam was built with |
| `I` | beam id, x, y, off-axis displacement, intensity (J/s/um^2) |
| `MINDIM` | smallest length scale in the beam, um |

90 intensity points. 64 come out bit-identical in Python; the rest differ by
at most 1.3e-15, all in the elliptical aperture, where a 1000-step midpoint
quadrature is summed in a different order.

Regenerating needs commons-math3 on the classpath, since `BeamGaussian` uses
its `Gaussian` and `NormalDistribution`:

```
javac -d /tmp/beam -sourcepath src \
      -cp lib/commons-math3-3.6.1/commons-math3-3.6.1.jar DumpBeam.java
java -cp /tmp/beam:lib/commons-math3-3.6.1/commons-math3-3.6.1.jar:. \
     se.raddo.raddose3D.DumpBeam > beams.tsv
```


---

# CoefCalc front-end reference

`coefficients_frontends.tsv` is the output of `DumpCoef2.java`, covering the
`AbsCoefCalc` routes that `coefficients.tsv` does not: `SmallMole`, `SAXS`
and `Sequence`. Same tags as that file, plus:

| tag | contents |
|---|---|
| `MONOMERS` | monomer count SAXS derived from the concentration |
| `RESIDUES` | amino acids, RNA and DNA counted from the sequence file |

`sequences/` holds the FASTA files the sequence cases were built from. They
are inputs to both the Java generator and the Python test.

One case exists to pin a defect rather than to check arithmetic:

A `smallmole-solv` case used to sit here too, built with 500 mM sodium and
containing none, recording that `SolventHeavyConc` was accepted under
`AbsCoefCalc SmallMole` and silently discarded. RADDOSE-3D now refuses it, so
there is nothing left to record and the case is gone.

- `saxs-rna` and `saxs-dna` use equal counts of each nucleic acid, and the
  heavier residue (RNA) must give the smaller monomer count. It gave the
  larger until `calculateNumMonomers` was fixed: it was declared
  `(residues, DNA, RNA)` and called `(residues, RNA, DNA)`. This fixture was
  regenerated from the fixed Java, so the two counts are now 296 RNA and 307
  DNA rather than the other way round.

## Residue table

`residues.tsv` (vendored into the package, not here) is `DumpResidues.java`
run against `ResidueDatabase.createResidueMap()`, plus the single-letter to
three-character key conversion for each sequence type. Dumped rather than
transcribed: the Java source is 815 lines of hand-written constants.

Regenerate as for the other dumpers. `DumpCoef2` takes the two FASTA paths as
arguments:

```
java -cp /tmp/coef2:. se.raddo.raddose3D.DumpCoef2 \
     seq_test.fasta seq_dna.fasta > coefficients_frontends.tsv
```


---

# PDB and CIF reference

`coefficients_files.tsv` is the output of `DumpCoef3.java` against the three
structure files in `tests/data/`: insulin 3I40, and two Crystallography Open
Database small-molecule entries. Same tags as the other CoefCalc fixtures.

Two cases exist to pin defects:

- Both CIF cases state a crystal density, which RADDOSE-3D does not read, so
  they check the calculation against something external rather than against
  itself. `CoefCalcFromCIF` used to ignore `_cell_formula_units_Z` and fill
  the cell with a single formula unit, making both low by exactly Z: urea
  0.683 against 1.367 with Z=2, COD 1008123 0.364 against 2.9 with Z=8. This
  fixture was regenerated from the fixed Java, so it now records 1.367 and
  2.913.
- The PDB case has 24 crystallographic symmetry operators and no
  non-crystallographic ones, so the monomer count comes out at 24. The Z on
  the CRYST1 line is ignored deliberately -- the Java's comment says it is
  "often mis-calculated in the PDB".

`DumpCoef3` takes the PDB path first, then any number of CIF paths:

```
java -cp /tmp/coef3:. se.raddo.raddose3D.DumpCoef3 \
     3I40.pdb COD-1008123.cif COD-1008775-urea.cif > coefficients_files.tsv
```


---

# findDepth reference

`find_depth.tsv.gz` is the output of `DumpDepth.java`: four crystals, each at
four rotations, probed on a 7x7x7 grid that reaches outside the crystal in
every direction so that misses, single crossings and grazing hits are all
covered. 5488 depths in total.

| tag | contents |
|---|---|
| `VERT` | vertex index, x, y, z -- the mesh before any rotation |
| `TRI` | triangle index and its three **one-based** vertex indices |
| `WEDGE` | the wedge start position, applied before the rotation |
| `DEPTH` | rotation angle (rad), voxel x, y, z, and the depth in um |

98% come out bit-identical in Python; the worst deviation is 2.7e-15 um.

Two of the four crystals exist for specific reasons. `cuboid-rotated` has
non-zero `AngleP` and `AngleL`, so no face is axis-aligned. `cylinder` is a
64-sided prism, which is the only case here with enough triangles for the
shared-edge de-duplication to matter -- removing that de-duplication fails
this case by a full 30 um and leaves the cuboids untouched.

The 90 degree rotation is the awkward one: `cos` is 6e-17 rather than 0, so
faces end up very nearly parallel to the beam and the ray-plane quotient
divides a cancelling numerator by a near-zero denominator. That is what
exposed the batch-dependence bug described in `test_geometry.py`.

Regenerate as for the other dumpers; `DumpDepth` takes no arguments.


---

# Exposure reference

`exposure.tsv` is the output of `DumpExposure.java`: five complete exposures
of a 40 x 30 x 20 um cuboid, each run end to end through `Crystal.expose`.

| tag | contents |
|---|---|
| `SETUP` | the four coefficients, density and voxels per micron |
| `WEDGE` | start, end, angular resolution, seconds, start position |
| `BEAM` | beam type and collimation |
| `GRID` | voxel counts along x, y, z |
| `SUMMARY` | occupied voxels, average dose, max dose |
| `VOX` | i, j, k, occupancy, dose, fluence, elastic yield |

| case | what it exercises |
|---|---|
| `still-tophat` | a still exposure, uniform beam |
| `rotation-gauss` | 46 angles, uncollimated Gaussian |
| `collimated` | a 50 um aperture, so the flux normalisation matters |
| `rotated-crystal` | AngleP 45, AngleL 20 -- no axis-aligned faces, and a different grid |
| `offset` | a non-zero wedge start position |

Grid shapes and occupancy counts match exactly. Doses match to ~1e-15 in the
interior. Voxels lying **exactly on a crystal face** do not and cannot: they
sit on the knife edge of the inside/outside parity test, and Java's
`Math.cos` differs from NumPy's in the last bit. Only `StrictMath` is
required to be reproducible, which is why RADDOSE-3D already varies against
itself across JVMs.

`test_on_face_voxels_are_intrinsically_unstable` measures this rather than
asserting it: nudging the crystal by a single ulp moves 606 voxels -- all of
them on a face, none in the interior -- and shifts the average dose by 3e-4,
which is *larger* than the disagreement with Java (1.6e-5). The port tracks
RADDOSE-3D more closely than the calculation tracks itself.

`AngleP` and `AngleL` are given in **degrees** in the input file; the
constructor converts. Passing radians to the generator looks right and
silently gives a 57-fold smaller rotation, which is what happened the first
time this fixture was built.

`DumpExposure` takes no arguments. Setting `FULL=1` dumps every voxel rather
than a strided sample, which is useful when locating a disagreement.
