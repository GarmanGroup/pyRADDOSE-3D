package se.raddo.raddose3D;

import java.util.HashMap;
import java.util.Map;

/** Dumps crystal meshes and findDepth results for the Python port. */
public final class DumpDepth {
  private DumpDepth() { }

  private static Map<Object, Object> crystalProperties(final double x,
      final double y, final double z, final double p, final double l) {
    Map<Object, Object> m = new HashMap<Object, Object>();
    m.put(Crystal.CRYSTAL_DIM_X, x);
    m.put(Crystal.CRYSTAL_DIM_Y, y);
    m.put(Crystal.CRYSTAL_DIM_Z, z);
    m.put(Crystal.CRYSTAL_RESOLUTION, 0.5);
    m.put(Crystal.CRYSTAL_ANGLE_P, p);
    m.put(Crystal.CRYSTAL_ANGLE_L, l);
    m.put(Crystal.CRYSTAL_COEFCALC, new CoefCalcAverage());
    return m;
  }

  private static Wedge wedge(final double startX, final double startY,
      final double startZ) {
    Map<Wedge.WedgeProperties, Double> w =
        new HashMap<Wedge.WedgeProperties, Double>();
    w.put(Wedge.WedgeProperties.ANGULAR_RESOLUTION, Math.toRadians(2));
    w.put(Wedge.WedgeProperties.START_POSITION_X, startX);
    w.put(Wedge.WedgeProperties.START_POSITION_Y, startY);
    w.put(Wedge.WedgeProperties.START_POSITION_Z, startZ);
    w.put(Wedge.WedgeProperties.ANGLE_START, 0d);
    w.put(Wedge.WedgeProperties.ANGLE_END, Math.toRadians(90));
    w.put(Wedge.WedgeProperties.EXPOSURE_TIME, 1d);
    return new Wedge(w);
  }

  private static void dump(final String id, final CrystalPolyhedron c,
      final Wedge w, final double[] angles, final double extent) {
    // The mesh, before any rotation.
    for (int v = 0; v < c.vertices.length; v++) {
      System.out.printf("VERT\t%s\t%d\t%.17g\t%.17g\t%.17g%n", id, v,
          c.vertices[v][0], c.vertices[v][1], c.vertices[v][2]);
    }
    for (int t = 0; t < c.indices.length; t++) {
      System.out.printf("TRI\t%s\t%d\t%d\t%d\t%d%n", id, t,
          c.indices[t][0], c.indices[t][1], c.indices[t][2]);
    }
    System.out.printf("WEDGE\t%s\t%.17g\t%.17g\t%.17g%n", id,
        w.getStartX(), w.getStartY(), w.getStartZ());

    for (double angle : angles) {
      c.setupDepthFinding(angle, w);
      // Probe on a grid that reaches outside the crystal in every direction,
      // so misses, single crossings and grazing hits are all covered.
      int n = 7;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            double[] p = {
                -extent + 2 * extent * i / (n - 1.0),
                -extent + 2 * extent * j / (n - 1.0),
                -extent + 2 * extent * k / (n - 1.0)};
            System.out.printf("DEPTH\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g%n",
                id, angle, p[0], p[1], p[2], c.findDepth(p, angle, w));
          }
        }
      }
    }
  }

  public static void main(final String[] args) {
    double[] angles = {0.0, Math.toRadians(30), Math.toRadians(90),
        Math.toRadians(137)};

    dump("cuboid", new CrystalCuboid(crystalProperties(40, 30, 20, 0, 0)),
        wedge(0, 0, 0), angles, 30.0);

    dump("cuboid-rotated",
        new CrystalCuboid(crystalProperties(40, 30, 20, Math.toRadians(45),
            Math.toRadians(20))), wedge(0, 0, 0), angles, 35.0);

    dump("cuboid-offset", new CrystalCuboid(crystalProperties(40, 30, 20, 0, 0)),
        wedge(5, -3, 2), angles, 35.0);

    dump("cylinder", new CrystalCylinder(crystalProperties(40, 40, 40, 0, 0)),
        wedge(0, 0, 0), angles, 30.0);
  }
}
