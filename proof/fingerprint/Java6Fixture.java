import java.lang.reflect.Field;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Synthetic fixtures of the hypothesis "Math.random() stream seeded by JDK6
 * new Random() on Windows HotSpot 6".
 *
 * JDK8's Random() default constructor differs from JDK6, so we reproduce the
 * JDK6 constructor explicitly:  new Random(++seedUniquifier + System.nanoTime())
 * with seedUniquifier starting at 8682522807148012L (JDK6 Random.java line 78),
 * and nanoTime emulated with the HotSpot 6 os_windows.cpp formula
 *   (jlong)(((double)qpc / (double)freq) * NANOSECS_PER_SEC).
 * Random(long) itself (scramble with 0x5DEECE66D) and nextDouble() are
 * unchanged between JDK6 and JDK8 and are executed by the real JDK here.
 *
 * Args: count rngSeed nMin nMax
 * Output (one JSON object per line): k, f, qpc, nanos, N, state_open (state
 * before the opening x draw, read by reflection), opening x/z exactly as
 * rd-132211 Player.resetPos computes them.
 */
public class Java6Fixture {
    static final long U = 8682522807148012L;
    static final long[] FREQS = {3579545L, 14318180L, 1193182L, 2394000000L};

    static long hsNanos(long q, long f) {
        double current = (double) q;
        double freq = (double) f;
        return (long) ((current / freq) * 1000000000.0);
    }

    public static void main(String[] a) throws Exception {
        int count = Integer.parseInt(a[0]);
        Random meta = new Random(Long.parseLong(a[1]));
        int nMin = Integer.parseInt(a[2]), nMax = Integer.parseInt(a[3]);
        Field fseed = Random.class.getDeclaredField("seed");
        fseed.setAccessible(true);
        for (int i = 0; i < count; i++) {
            int k = 1 + meta.nextInt(8);
            long f = FREQS[meta.nextInt(FREQS.length)];
            // uptime between 60 s and 20 h
            double up = 60 + meta.nextDouble() * (20 * 3600 - 60);
            long q = (long) (up * f);
            long nanos = hsNanos(q, f);
            Random r = new Random(U + k + nanos);   // == JDK6 new Random() with ++seedUniquifier == U+k
            int N = nMin + meta.nextInt(nMax - nMin + 1);
            for (int c = 0; c < N; c++) r.nextDouble();   // generator's Math.random() calls
            long state = ((AtomicLong) fseed.get(r)).get();
            float x = (float) r.nextDouble() * (float) 256;  // Player.resetPos
            float y = 64 + 10;
            float z = (float) r.nextDouble() * (float) 256;
            System.out.println("{\"k\":" + k + ",\"f\":" + f + ",\"qpc\":" + q + ",\"nanos\":" + nanos
                + ",\"N\":" + N + ",\"state_open\":" + state + ",\"x\":" + x + ",\"z\":" + z + "}");
        }
    }
}
