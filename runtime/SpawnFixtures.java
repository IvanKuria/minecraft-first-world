import java.util.Random;
/** Independent Java positive controls. Ground-truth seed is evaluation data only. */
public class SpawnFixtures {
    public static void main(String[] args) {
        Random fixtures = new Random(20260926L);
        for (int sample = 0; sample < 64; sample++) {
            long seed = fixtures.nextLong();
            Random rng = new Random(seed);
            System.out.print("{\"seed\":" + seed + ",\"coordinates\":[");
            for (int i = 0; i < 30; i++) {
                if (i > 0) System.out.print(",");
                System.out.print(Double.toString((double)((float)rng.nextDouble() * 256)));
            }
            System.out.println("],\"held_out_next_double\":" + rng.nextDouble() + "}");
        }
    }
}
