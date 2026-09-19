package se.raddo.raddose3D;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/** Dumps the remaining CoefCalc front-ends: SmallMole, SAXS and Sequence. */
public final class DumpCoef2 {
  private DumpCoef2() { }

  private static final List<String> NO_S = new ArrayList<String>();
  private static final List<Double> NO_D = new ArrayList<Double>();

  private static void report(final String id, final CoefCalcCompute cc) {
    System.out.printf("DENSITY\t%s\t%.17g%n", id, cc.getDensity());
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
    // --- SmallMole: explicit atoms, no residues, no solvent ---
    System.out.println("CASE\tsmallmole\t10.0\t12.0\t14.0\t90.0\t90.0\t90.0\t4");
    report("smallmole", new CoefCalcSmallMolecules(10.0, 12.0, 14.0,
        90.0, 90.0, 90.0, 4,
        Arrays.asList("C", "H", "N", "O", "S"),
        Arrays.asList(6.0, 12.0, 2.0, 3.0, 1.0),
        NO_S, NO_D, null, NO_S, NO_D, null, null, NO_S, NO_D, 0.0, 0L));

    // A "smallmole-solv" case used to live here, with 500 mM Na, recording
    // that RADDOSE-3D accepted SolventHeavyConc under SmallMole and silently
    // discarded it. It is now refused, so there is nothing left to record --
    // the refusal is covered by SilentNoOpTest on the Java side and by
    // test_frontends.py on the Python side.

    // --- SAXS: monomer count derived from concentration ---
    System.out.println("CASE\tsaxs\tdefault-cell");
    CoefCalcSAXS saxs = new CoefCalcSAXS(null, null, null, null, null, null,
        129, 0, 0, NO_S, NO_D, NO_S, NO_D, null, 10.0, NO_S, NO_D,
        null, null, 0, NO_S, NO_D, 0.0);
    System.out.printf("MONOMERS\tsaxs\t%d%n", saxs.getNumMonomers());
    report("saxs", saxs);

    // SAXS with RNA, which is where the swapped-argument defect shows.
    System.out.println("CASE\tsaxs-rna\tdefault-cell");
    CoefCalcSAXS saxsRna = new CoefCalcSAXS(null, null, null, null, null, null,
        0, 60, 0, NO_S, NO_D, NO_S, NO_D, null, 10.0, NO_S, NO_D,
        null, null, 0, NO_S, NO_D, 0.0);
    System.out.printf("MONOMERS\tsaxs-rna\t%d%n", saxsRna.getNumMonomers());
    report("saxs-rna", saxsRna);

    // SAXS with DNA, the mirror case.
    System.out.println("CASE\tsaxs-dna\tdefault-cell");
    CoefCalcSAXS saxsDna = new CoefCalcSAXS(null, null, null, null, null, null,
        0, 0, 60, NO_S, NO_D, NO_S, NO_D, null, 10.0, NO_S, NO_D,
        null, null, 0, NO_S, NO_D, 0.0);
    System.out.printf("MONOMERS\tsaxs-dna\t%d%n", saxsDna.getNumMonomers());
    report("saxs-dna", saxsDna);

    // --- Sequence: real per-residue compositions ---
    System.out.println("CASE\tsequence\t78.27\t78.27\t78.27\t90.0\t90.0\t90.0\t8");
    CoefCalcFromSequence seq = new CoefCalcFromSequence(78.27, 78.27, 78.27,
        90.0, 90.0, 90.0, 8, NO_S, NO_D, NO_S, NO_D, null,
        args[0], NO_S, NO_D, null, null, 0, NO_S, NO_D, 0.0);
    System.out.printf("RESIDUES\tsequence\t%.0f\t%.0f\t%.0f%n",
        seq.getNumAminoAcids(), seq.getNumRNA(), seq.getNumDNA());
    report("sequence", seq);

    System.out.println("CASE\tsequence-dna\t60.0\t60.0\t120.0\t90.0\t90.0\t120.0\t2");
    CoefCalcFromSequence seqDna = new CoefCalcFromSequence(60.0, 60.0, 120.0,
        90.0, 90.0, 120.0, 2, NO_S, NO_D, NO_S, NO_D, null,
        args[1], NO_S, NO_D, null, null, 0, NO_S, NO_D, 0.0);
    System.out.printf("RESIDUES\tsequence-dna\t%.0f\t%.0f\t%.0f%n",
        seqDna.getNumAminoAcids(), seqDna.getNumRNA(), seqDna.getNumDNA());
    report("sequence-dna", seqDna);
  }
}
