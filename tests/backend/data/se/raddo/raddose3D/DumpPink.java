package se.raddo.raddose3D;

/** Prints SampleNormalEnergyDistribution's energies, for test_beam_pink.py. */
public class DumpPink {
  public static void main(String[] args) {
    double[] energies = new SampleNormalEnergyDistribution(12.1, 0.5, 200).getSampledEnergies();
    StringBuilder line = new StringBuilder();
    for (double e : energies) {
      line.append(e).append(' ');
    }
    System.out.println(line.toString().trim());
  }
}
