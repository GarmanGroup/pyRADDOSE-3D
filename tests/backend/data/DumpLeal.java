import se.raddo.raddose3D.DDMLeal;

/** Prints DDMLeal.calcDecay at a few doses, for test_decay.py. */
public class DumpLeal {
  public static void main(String[] args) {
    double[][] parameters = {{0.3, 10.0, 0.03}, {0.1, 5.0, 0.5}, {0.0, 0.0, 0.0}};
    double[] doses = {0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0};
    for (double[] p : parameters) {
      DDMLeal ddm = new DDMLeal(p[0], p[1], p[2]);
      StringBuilder line = new StringBuilder(p[0] + " " + p[1] + " " + p[2] + ":");
      for (double d : doses) {
        line.append(" ").append(ddm.calcDecay(d));
      }
      System.out.println(line);
    }
  }
}
