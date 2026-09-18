package se.raddo.raddose3D;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** Dumps the file-reading CoefCalc front-ends: PDB and CIF. */
public final class DumpCoef3 {
  private DumpCoef3() { }

  private static final List<String> NO_S = new ArrayList<String>();
  private static final List<Double> NO_D = new ArrayList<Double>();

  private static void report(final String id, final CoefCalcCompute cc) {
    System.out.printf("DENSITY\t%s\t%.17g%n", id, cc.getDensity());
    System.out.printf("SOLFRAC\t%s\t%.17g%n", id, cc.getSolventFraction());
    System.out.printf("MONOMERS\t%s\t%d%n", id, cc.getNumMonomers());
    System.out.printf("RESIDUES\t%s\t%.0f\t%.0f\t%.0f%n", id,
        cc.getNumAminoAcids(), cc.getNumRNA(), cc.getNumDNA());
    Set<Element> present = cc.getPresentElements(false);
    for (Element e : present) {
      System.out.printf("ATOMS\t%s\t%d\t%s\t%.17g%n", id, e.getAtomicNumber(),
          e.getElementName(), cc.totalAtoms(e));
    }
    for (double energy : new double[] {6.0, 12.1, 20.0}) {
      cc.updateCoefficients(energy);
      System.out.printf("COEF\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g%n",
          id, energy,
          cc.getAbsorptionCoefficient(), cc.getAttenuationCoefficient(),
          cc.getElasticCoefficient(), cc.getInelasticCoefficient());
    }
  }

  public static void main(final String[] args) {
    System.out.println("CASE\tpdb\t" + args[0]);
    report("pdb", new CoefCalcFromPDB(args[0], NO_S, NO_D, null, null,
        NO_S, NO_D, 0.0, 0L));

    for (int i = 1; i < args.length; i++) {
      String id = "cif" + i;
      System.out.println("CASE\t" + id + "\t" + args[i]);
      report(id, new CoefCalcFromCIF(args[i]));
    }
  }
}
