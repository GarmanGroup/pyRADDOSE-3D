import java.util.Random;

/** Prints java.util.Random draws, for test_java_random.py. */
public class DumpRandom {
  public static void main(String[] args) {
    Random r = new Random(20260917L);
    int[] bounds = {1, 2, 7, 100, 128, 1000, 1500, 2147483647, 1073741825};
    StringBuilder ints = new StringBuilder();
    for (int i = 0; i < 40; i++) {
      ints.append(r.nextInt(bounds[i % bounds.length])).append(' ');
    }
    System.out.println(ints.toString().trim());
    StringBuilder doubles = new StringBuilder();
    for (int i = 0; i < 5; i++) {
      doubles.append(r.nextDouble()).append(' ');
    }
    System.out.println(doubles.toString().trim());
    Random negative = new Random(-123456789L);
    System.out.println(negative.nextInt(1000) + " " + negative.nextInt(3));
  }
}
