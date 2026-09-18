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
