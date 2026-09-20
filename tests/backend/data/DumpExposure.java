package se.raddo.raddose3D;

import java.util.HashMap;
import java.util.Map;

/** Dumps a whole exposure: grid, occupancy and per-voxel dose. */
public final class DumpExposure {
  private DumpExposure() { }

  private static void run(final String id, final double xdim, final double ydim,
      final double zdim, final double ppum, final double angleP,
      final double angleL, final boolean tophat, final Double collimation,
      final double startAngle, final double endAngle, final double angRes,
      final double startX, final double startY, final double startZ) {

    Map<Object, Object> cp = new HashMap<Object, Object>();
    cp.put(Crystal.CRYSTAL_DIM_X, xdim);
    cp.put(Crystal.CRYSTAL_DIM_Y, ydim);
    cp.put(Crystal.CRYSTAL_DIM_Z, zdim);
    cp.put(Crystal.CRYSTAL_RESOLUTION, ppum);
    cp.put(Crystal.CRYSTAL_ANGLE_P, angleP);
    cp.put(Crystal.CRYSTAL_ANGLE_L, angleL);
    cp.put(Crystal.CRYSTAL_DDM, new DDMSimple());
    CoefCalcFromParams cc = new CoefCalcFromParams(78.27, 78.27, 78.27,
        90.0, 90.0, 90.0, 8, 129, 0, 0,
        new java.util.ArrayList<String>(), new java.util.ArrayList<Double>(),
        new java.util.ArrayList<String>(), new java.util.ArrayList<Double>(),
        new java.util.ArrayList<String>(), new java.util.ArrayList<Double>(),
        null, null, null, 0,
        new java.util.ArrayList<String>(), new java.util.ArrayList<Double>(),
        0.0, 0L);
    cp.put(Crystal.CRYSTAL_COEFCALC, cc);
    CrystalPolyhedron c = new CrystalCuboid(cp);

    Map<Object, Object> bp = new HashMap<Object, Object>();
    bp.put(Beam.BEAM_ENERGY, 12.1);
    bp.put(Beam.BEAM_FLUX, 2.0e12);
    Beam b;
    if (tophat) {
      bp.put(Beam.BEAM_COLL_H, 100.0);
      bp.put(Beam.BEAM_COLL_V, 100.0);
      b = new BeamTophat(bp);
    } else {
      bp.put(Beam.BEAM_FWHM_X, 80.0);
      bp.put(Beam.BEAM_FWHM_Y, 60.0);
      if (collimation != null) {
        bp.put(Beam.BEAM_COLL_H, collimation);
        bp.put(Beam.BEAM_COLL_V, collimation);
      }
      b = new BeamGaussian(bp);
    }

    Map<Wedge.WedgeProperties, Double> wp =
        new HashMap<Wedge.WedgeProperties, Double>();
    wp.put(Wedge.WedgeProperties.ANGULAR_RESOLUTION, angRes);
    wp.put(Wedge.WedgeProperties.ANGLE_START, startAngle);
    wp.put(Wedge.WedgeProperties.ANGLE_END, endAngle);
    wp.put(Wedge.WedgeProperties.EXPOSURE_TIME, 50.0);
    wp.put(Wedge.WedgeProperties.START_POSITION_X, startX);
    wp.put(Wedge.WedgeProperties.START_POSITION_Y, startY);
    wp.put(Wedge.WedgeProperties.START_POSITION_Z, startZ);
    Wedge w = new Wedge(wp);

    // Coefficients and density, so the Python does not have to rebuild the
    // composition to compare the exposure.
    cc.updateCoefficients(b);
    System.out.printf("SETUP\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g%n", id,
        cc.getAbsorptionCoefficient(), cc.getAttenuationCoefficient(),
        cc.getElasticCoefficient(), cc.getInelasticCoefficient(),
        cc.getDensity(), ppum);
    System.out.printf("WEDGE\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g%n",
        id, startAngle, endAngle, angRes, 50.0, startX, startY, startZ);
    System.out.printf("BEAM\t%s\t%s\t%s%n", id, tophat ? "tophat" : "gaussian",
        collimation == null ? "none" : String.valueOf(collimation));

    int[] size = c.getCrystSizeVoxels();
    System.out.printf("GRID\t%s\t%d\t%d\t%d%n", id, size[0], size[1], size[2]);
    double[] v0 = c.getCrystCoord(0, 0, 0);
    System.out.printf("ORIGIN\t%s\t%.17g\t%.17g\t%.17g%n", id, v0[0], v0[1], v0[2]);

    c.expose(b, w);

    int occupied = 0;
    double total = 0, peak = 0;
    for (int i = 0; i < size[0]; i++) {
      for (int j = 0; j < size[1]; j++) {
        for (int k = 0; k < size[2]; k++) {
          if (c.isCrystalAt(i, j, k)) {
            occupied++;
            double d = c.getDose(i, j, k);
            total += d;
            peak = Math.max(peak, d);
          }
        }
      }
    }
    System.out.printf("SUMMARY\t%s\t%d\t%.17g\t%.17g%n", id, occupied,
        total / occupied, peak);

    // A sparse sample of individual voxels, so a disagreement can be located.
    int stride = System.getenv("FULL") != null ? 1 : Math.max(1, size[0] / 4);
    for (int i = 0; i < size[0]; i += stride) {
      for (int j = 0; j < size[1]; j += (System.getenv("FULL") != null ? 1 : Math.max(1, size[1] / 4))) {
        for (int k = 0; k < size[2]; k += (System.getenv("FULL") != null ? 1 : Math.max(1, size[2] / 4))) {
          System.out.printf("VOX\t%s\t%d\t%d\t%d\t%b\t%.17g\t%.17g\t%.17g%n",
              id, i, j, k, c.isCrystalAt(i, j, k), c.getDose(i, j, k),
              c.getFluence(i, j, k), c.getElastic(i, j, k));
        }
      }
    }
  }

  public static void main(final String[] args) {
    double deg = Math.PI / 180;
    run("still-tophat", 40, 30, 20, 0.5, 0, 0, true, null, 0, 0, 2 * deg, 0, 0, 0);
    run("rotation-gauss", 40, 30, 20, 0.5, 0, 0, false, null,
        0, 90 * deg, 2 * deg, 0, 0, 0);
    run("collimated", 40, 30, 20, 0.5, 0, 0, false, 50.0,
        0, 90 * deg, 5 * deg, 0, 0, 0);
    run("rotated-crystal", 40, 30, 20, 0.5, 45.0, 20.0, false, null,
        0, 90 * deg, 5 * deg, 0, 0, 0);
    run("offset", 40, 30, 20, 0.5, 0, 0, false, null,
        0, 90 * deg, 5 * deg, 5, -3, 2);
  }
}
