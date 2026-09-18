package se.raddo.raddose3D;

import java.util.Map;
import se.raddo.raddose3D.Element.CrossSection;

/** Dumps Element cross-sections across all elements and a range of energies. */
public final class DumpXS {
  private DumpXS() { }

  public static void main(final String[] args) {
    ElementDatabase db = ElementDatabase.getInstance();
    // Energies spanning the MX range plus points that land inside L and M
    // shells, on the E == 1 special case, and just either side of edges.
    double[] energies = {
      0.05, 0.1, 0.3, 0.5, 0.999, 1.0, 1.001, 1.5, 2.0, 3.0, 4.0,
      6.0, 8.0, 10.0, 12.1, 12.398, 15.0, 20.0, 25.0, 30.0,
    };
    System.out.println("z\tname\tenergy\tphotoelectric\tcoherent\tcompton\ttotal");
    for (int z = 1; z <= 92; z++) {
      Element e = db.getElement(z);
      if (e == null) {
        continue;
      }
      for (double energy : energies) {
        Map<CrossSection, Double> xs = e.getAbsCoefficients(energy);
        System.out.printf("%d\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g%n",
            z, e.getElementName(), energy,
            xs.get(CrossSection.PHOTOELECTRIC),
            xs.get(CrossSection.COHERENT),
            xs.get(CrossSection.COMPTON),
            xs.get(CrossSection.TOTAL));
      }
    }
  }
}
