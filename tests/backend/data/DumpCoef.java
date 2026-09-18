package se.raddo.raddose3D;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/** Dumps CoefCalc compositions and the coefficients computed from them. */
public final class DumpCoef {
  private DumpCoef() { }

  private static final List<String> NO_S = new ArrayList<String>();
  private static final List<Double> NO_D = new ArrayList<Double>();

  private static void dump(final String id, final double a, final double b,
      final double c, final double al, final double be, final double ga,
      final int monomers, final int residues, final int rna, final int dna,
      final List<String> haNames, final List<Double> haNums,
      final List<String> solNames, final List<Double> solNums,
      final Double solventFraction) {

    CoefCalcFromParams cc = new CoefCalcFromParams(a, b, c, al, be, ga,
        monomers, residues, rna, dna, haNames, haNums, solNames, solNums,
        NO_S, NO_D, solventFraction, null, null, 0, NO_S, NO_D, 0.0, 0L);

    StringBuilder sb = new StringBuilder();
    sb.append("CASE\t").append(id)
      .append("\t").append(a).append("\t").append(b).append("\t").append(c)
      .append("\t").append(al).append("\t").append(be).append("\t").append(ga);
    System.out.println(sb);
    // Inputs, so the fixture describes the calculation it came from.
    System.out.printf("INPUT\t%s\t%d\t%d\t%d\t%d\t%s%n", id, monomers,
        residues, rna, dna,
        solventFraction == null ? "auto" : String.format("%.17g", solventFraction));
    for (int i = 0; i < haNames.size(); i++) {
      System.out.printf("HAPROT\t%s\t%s\t%.17g%n", id, haNames.get(i), haNums.get(i));
    }
    for (int i = 0; i < solNames.size(); i++) {
      System.out.printf("HASOLV\t%s\t%s\t%.17g%n", id, solNames.get(i), solNums.get(i));
    }
    System.out.printf("CELLVOL\t%s\t%.17g%n", id, cc.cellVolume(a, b, c, al, be, ga));
    System.out.printf("DENSITY\t%s\t%.17g%n", id, cc.getDensity());
    System.out.printf("SOLFRAC\t%s\t%.17g%n", id, cc.getSolventFraction());

    Set<Element> present = cc.getPresentElements(false);
    for (Element e : present) {
      System.out.printf("ATOMS\t%s\t%d\t%s\t%.17g%n", id, e.getAtomicNumber(),
          e.getElementName(), cc.totalAtoms(e));
    }

    double[] energies = {6.0, 8.0, 12.1, 12.398, 16.0, 20.0};
    for (double energy : energies) {
      cc.updateCoefficients(energy);
      System.out.printf("COEF\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g%n",
          id, energy,
          cc.getAbsorptionCoefficient(), cc.getAttenuationCoefficient(),
          cc.getElasticCoefficient(), cc.getInelasticCoefficient());
    }
  }

  public static void main(final String[] args) {
    // Baseline: the lysozyme-like cell from the RADDOSE-3D examples.
    dump("baseline", 78.27, 78.27, 78.27, 90.0, 90.0, 90.0,
        8, 129, 0, 0, NO_S, NO_D, NO_S, NO_D, null);

    // Explicit solvent fraction rather than a computed one.
    dump("solvent-explicit", 78.27, 78.27, 78.27, 90.0, 90.0, 90.0,
        8, 129, 0, 0, NO_S, NO_D, NO_S, NO_D, 0.5);

    // Heavy atoms in the protein: 2 S and 1 Fe per monomer.
    dump("heavy-protein", 78.27, 78.27, 78.27, 90.0, 90.0, 90.0,
        8, 129, 0, 0,
        Arrays.asList("S", "Fe"), Arrays.asList(2.0, 1.0), NO_S, NO_D, null);

    // Heavy atoms in the solvent, in mmol/l.
    dump("heavy-solvent", 78.27, 78.27, 78.27, 90.0, 90.0, 90.0,
        8, 129, 0, 0, NO_S, NO_D,
        Arrays.asList("Na", "Cl"), Arrays.asList(150.0, 150.0), null);

    // Triclinic: every angle different, so the volume formula is exercised.
    dump("triclinic", 40.0, 55.5, 71.2, 71.0, 83.5, 99.2,
        4, 250, 0, 0, NO_S, NO_D, NO_S, NO_D, null);

    // Nucleic acid.
    dump("rna-dna", 60.0, 60.0, 120.0, 90.0, 90.0, 120.0,
        2, 0, 40, 20, NO_S, NO_D, NO_S, NO_D, null);
  }
}
