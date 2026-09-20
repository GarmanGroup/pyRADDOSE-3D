package se.raddo.raddose3D;

import java.util.HashMap;
import java.util.Map;

/** Dumps the fixed vertex and index tables of the built-in crystal shapes. */
public final class DumpMeshes {
  private DumpMeshes() { }

  private static Map<Object, Object> props(final double d, final double h) {
    Map<Object, Object> m = new HashMap<Object, Object>();
    m.put(Crystal.CRYSTAL_DIM_X, d);
    m.put(Crystal.CRYSTAL_DIM_Y, h);
    m.put(Crystal.CRYSTAL_DIM_Z, d);
    m.put(Crystal.CRYSTAL_RESOLUTION, 0.5);
    m.put(Crystal.CRYSTAL_ANGLE_P, 0.0);
    m.put(Crystal.CRYSTAL_ANGLE_L, 0.0);
    m.put(Crystal.CRYSTAL_COEFCALC, new CoefCalcAverage());
    return m;
  }

  private static void dump(final String id, final CrystalPolyhedron c,
      final double scale) {
    for (int v = 0; v < c.vertices.length; v++) {
      System.out.printf("VERT\t%s\t%d\t%.17g\t%.17g\t%.17g%n", id, v,
          c.vertices[v][0] / scale, c.vertices[v][1] / scale,
          c.vertices[v][2] / scale);
    }
    for (int t = 0; t < c.indices.length; t++) {
      System.out.printf("TRI\t%s\t%d\t%d\t%d\t%d%n", id, t,
          c.indices[t][0], c.indices[t][1], c.indices[t][2]);
    }
  }

  public static void main(final String[] a) {
    // Unit sphere: the table is scaled by the diameter, so dividing it out
    // again recovers the shape itself.
    dump("sphere", new CrystalSphericalNew(props(1.0, 1.0)), 1.0);

    // The cylinder is procedural, so it is dumped at a known size and the
    // construction is ported rather than the table.
    dump("cylinder-40x30", new CrystalCylinder(props(40.0, 30.0)), 1.0);
  }
}
