package se.raddo.raddose3D;

import java.util.HashMap;
import java.util.Map;

/** Dumps beam intensity profiles and normalisation factors. */
public final class DumpBeam {
  private DumpBeam() { }

  private static final double FLUX = 2.0e12;
  private static final double ENERGY = 12.1;

  private static void probe(final String id, final Beam b) {
    // normFactor is private; recover it from the ratio of the on-axis
    // intensity to the same beam uncollimated, exactly as the Java tests do.
    double[][] points = {
      {0, 0}, {10, 0}, {0, 10}, {25, 25}, {-30, 15}, {49, 0}, {51, 0},
      {0, 49}, {0, 51}, {40, 40}, {-45, -20}, {100, 100},
    };
    for (double[] pt : points) {
      System.out.printf("I\t%s\t%.17g\t%.17g\t%.17g\t%.17g%n",
          id, pt[0], pt[1], 0.0, b.beamIntensity(pt[0], pt[1], 0.0));
    }
    // With an off-axis displacement.
    for (double[] pt : new double[][] {{0, 0}, {20, 10}, {-20, -10}}) {
      System.out.printf("I\t%s\t%.17g\t%.17g\t%.17g\t%.17g%n",
          id, pt[0], pt[1], 15.0, b.beamIntensity(pt[0], pt[1], 15.0));
    }
    System.out.printf("MINDIM\t%s\t%.17g%n", id, b.beamMinumumDimension());
  }

  private static Beam gaussian(final double fwhmX, final double fwhmY,
      final Double collX, final Double collY, final boolean circular) {
    Map<Object, Object> p = new HashMap<Object, Object>();
    p.put(Beam.BEAM_FWHM_X, fwhmX);
    p.put(Beam.BEAM_FWHM_Y, fwhmY);
    p.put(Beam.BEAM_ENERGY, ENERGY);
    p.put(Beam.BEAM_FLUX, FLUX);
    if (collX != null) {
      p.put(Beam.BEAM_COLL_H, collX);
      p.put(Beam.BEAM_COLL_V, collY);
    }
    if (circular) {
      p.put(Beam.BEAM_CIRCULAR, "TRUE");
    }
    BeamGaussian b = new BeamGaussian(p);
    b.applyContainerAttenuation(new ContainerTransparent());
    return b;
  }

  private static Beam tophat(final double x, final double y,
      final boolean circular) {
    Map<Object, Object> p = new HashMap<Object, Object>();
    p.put(Beam.BEAM_COLL_H, x);
    p.put(Beam.BEAM_COLL_V, y);
    p.put(Beam.BEAM_ENERGY, ENERGY);
    p.put(Beam.BEAM_FLUX, FLUX);
    if (circular) {
      p.put(Beam.BEAM_CIRCULAR, "TRUE");
    }
    BeamTophat b = new BeamTophat(p);
    b.applyContainerAttenuation(new ContainerTransparent());
    return b;
  }

  public static void main(final String[] args) {
    System.out.printf("PARAMS\tflux\t%.17g\tenergy\t%.17g%n", FLUX, ENERGY);
    probe("gauss-rect", gaussian(80.0, 60.0, 100.0, 100.0, false));
    probe("gauss-uncollimated", gaussian(80.0, 60.0, null, null, false));
    probe("gauss-ellipse", gaussian(80.0, 60.0, 100.0, 100.0, true));
    probe("gauss-narrow-coll", gaussian(80.0, 60.0, 40.0, 30.0, false));
    probe("tophat-rect", tophat(100.0, 100.0, false));
    probe("tophat-ellipse", tophat(100.0, 80.0, true));
  }
}
