import java.lang.reflect.Field;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
/** Java ground truth for lcg.py. Prints one JSON object per line. */
public class RandomFixtures {
    static long state(Random r) throws Exception {
        Field f = Random.class.getDeclaredField("seed"); f.setAccessible(true);
        return ((AtomicLong) f.get(r)).get();
    }
    public static void main(String[] a) throws Exception {
        Random meta = new Random(15102026L);
        int[] bounds = {1, 2, 3, 7, 16, 64, 100, 255, 256, 1000, 65536, 4194304, 1 << 30, 2147483647, 1431655765};
        for (int s = 0; s < 200; s++) {
            long seed = meta.nextLong();
            Random r = new Random(seed);
            StringBuilder sb = new StringBuilder();
            sb.append("{\"seed\":").append(seed).append(",\"state0\":").append(state(r)).append(",\"ops\":[");
            int n = 60;
            for (int i = 0; i < n; i++) {
                int op = meta.nextInt(7);
                if (i > 0) sb.append(',');
                long before = state(r);
                String v;
                switch (op) {
                    case 0: v = Double.toString(r.nextDouble()); break;
                    case 1: v = Float.toString(r.nextFloat()); break;
                    case 2: v = Integer.toString(r.nextInt()); break;
                    case 3: { int b = bounds[meta.nextInt(bounds.length)]; v = b + ":" + r.nextInt(b); break; }
                    case 4: v = Long.toString(r.nextLong()); break;
                    case 5: v = Boolean.toString(r.nextBoolean()); break;
                    default: v = Double.toString((double)((float)r.nextDouble() * (float)256)); break;
                }
                sb.append("[").append(op).append(",\"").append(v).append("\",").append(before).append(',').append(state(r)).append(']');
            }
            sb.append("],\"state_end\":").append(state(r)).append('}');
            System.out.println(sb);
        }
        // jump check: count raw steps for a long run of Math.random-style draws
        Random r = new Random(123456789L);
        long s0 = state(r);
        for (int i = 0; i < 1000000; i++) r.nextDouble();
        System.out.println("{\"jump_seed\":123456789,\"s0\":" + s0 + ",\"draws\":1000000,\"s_end\":" + state(r) + "}");
    }
}
