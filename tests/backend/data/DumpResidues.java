package se.raddo.raddose3D;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/** Dumps the residue composition table so it can be vendored as data. */
public final class DumpResidues {
  private DumpResidues() { }

  public static void main(final String[] args) {
    HashMap<String, Residue> map = ResidueDatabase.createResidueMap();
    System.out.println(
        "# RES key identifier type C H O N S Se P molecular_weight");
    for (Map.Entry<String, Residue> e : new TreeMap<String, Residue>(map).entrySet()) {
      Residue r = e.getValue();
      System.out.printf("RES\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.17g%n",
          e.getKey(), r.getIdentifier(), r.getResidueType(),
          r.getCarbons(), r.getHydrogens(), r.getOxygens(), r.getNitrogens(),
          r.getSulphurs(), r.getSeleniums(), r.getPhosphoruses(),
          r.getMolecularWeight());
    }
    // The single-letter to three-character conversion, which depends on the
    // sequence type. Dumped rather than transcribed.
    ResidueDatabase db = ResidueDatabase.getInstance();
    for (int type = 1; type <= 3; type++) {
      for (char c = 'A'; c <= 'Z'; c++) {
        String key = db.idLetterConversion(String.valueOf(c), type);
        Residue r = db.getResidue(String.valueOf(c), type);
        System.out.printf("LETTER\t%d\t%c\t%s\t%s%n", type, c,
            key == null ? "" : key, r == null ? "UNKNOWN" : "OK");
      }
    }
  }
}
