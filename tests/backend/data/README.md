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

Two cases exist to pin defects rather than to check arithmetic:

- `smallmole-solv` is `smallmole` plus 500 mM sodium, and contains no sodium.
  `CoefCalcSmallMolecules` guards `calculateSolventWater` behind a hardcoded
  `boolean fillRestWithWater = false`, so `SolventHeavyConc` is accepted and
  discarded.
- `saxs-rna` and `saxs-dna` use equal counts of each nucleic acid and give
  *different* monomer counts, because `calculateNumMonomers` is declared
  `(residues, DNA, RNA)` and called `(residues, RNA, DNA)`.

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
