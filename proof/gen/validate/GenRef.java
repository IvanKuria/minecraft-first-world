import java.util.Random;

/* Independent Java reference for the GEN family, used only to validate bit-exactness of the C engine.
 * Uses the real java.util.Random (subclass only counts next() calls; algorithm untouched).
 * "math" variants call r.nextDouble(), which is exactly what Math.random() does on its global Random.
 * Usage: java GenRef rng <state48> [count]          -> fixture lines
 *        java GenRef gen <state48> W H D init seed rng pick nb order rule K M nbdraw
 */
public class GenRef {
    static class CRandom extends Random {
        long calls = 0;
        CRandom(long internal) { super(internal ^ 0x5DEECE66DL); }
        protected int next(int bits) { calls++; return super.next(bits); }
    }
    static CRandom R; static boolean MATH;
    static int draw(int n) { return MATH ? (int) (R.nextDouble() * n) : R.nextInt(n); }

    public static void main(String[] a) throws Exception {
        if (a[0].equals("rng")) {
            long s = Long.decode(a[1]); int cnt = a.length > 2 ? Integer.parseInt(a[2]) : 20;
            CRandom r = new CRandom(s);
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < cnt; i++) {
                sb.append(r.nextInt(256)).append(' ').append(r.nextInt(64)).append(' ').append(r.nextInt(6)).append(' ')
                  .append(r.nextInt(2)).append(' ').append(r.nextInt(4194304)).append(' ').append(r.nextInt(1000)).append(' ')
                  .append(Double.doubleToRawLongBits(r.nextDouble())).append(' ').append(Float.floatToRawIntBits(r.nextFloat())).append('\n');
            }
            sb.append("calls ").append(r.calls).append('\n');
            System.out.print(sb);
            return;
        }
        long s = Long.decode(a[1]);
        int W = Integer.parseInt(a[2]), H = Integer.parseInt(a[3]), D = Integer.parseInt(a[4]);
        String init = a[5], seed = a[6]; MATH = a[7].equals("math"); String pick = a[8], nb = a[9];
        int order = Integer.parseInt(a[10]); String rule = a[11]; long K = Long.parseLong(a[12]); long M = Long.parseLong(a[13]);
        boolean lazy = a.length > 14 && a[14].equals("iffree");
        int mg = a.length > 15 ? Integer.parseInt(a[15]) : 0; boolean until = a.length > 16 && a[16].equals("1");
        R = new CRandom(s);
        int[][][] ORD = {{{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}}, {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}}};
        int[][] O = ORD[order];
        byte[] b = new byte[W * H * D];
        final byte UNSET = (byte) 255;
        for (int y = 0; y < D; y++) for (int z = 0; z < H; z++) for (int x = 0; x < W; x++) {
            int i = (y * H + z) * W + x;
            b[i] = init.equals("unset") ? UNSET : init.equals("slab") ? (byte) (y <= D * 2 / 3 ? 1 : 0) : 0;
        }
        for (long k = 0; k < K; k++) {
            int x, y, z;
            if (pick.equals("idx")) { int i = draw(W * H * D); x = i % W; z = (i / W) % H; y = i / (W * H); }
            else { x = draw(W - 2*mg) + mg; y = draw(D - 2*mg) + mg; z = draw(H - 2*mg) + mg; }
            byte t = 1;
            if (seed.equals("air")) t = 0; else if (seed.equals("rand2")) t = (byte) draw(2);
            b[(y * H + z) * W + x] = t;
        }
        byte FREE = init.equals("unset") ? UNSET : 0;
        long unset = 0;
        if (until) for (int y = mg; y < D - mg; y++) for (int z = mg; z < H - mg; z++) for (int x = mg; x < W - mg; x++) if (b[(y * H + z) * W + x] == UNSET) unset++;
        for (long it = 0; until ? unset > 0 : it < M; it++) {
            int x, y, z;
            if (pick.equals("idx")) { int i = draw(W * H * D); x = i % W; z = (i / W) % H; y = i / (W * H); }
            else { x = draw(W - 2*mg) + mg; y = draw(D - 2*mg) + mg; z = draw(H - 2*mg) + mg; }
            int i = (y * H + z) * W + x;
            int d = -1;
            if (nb.equals("rand") && !(lazy && rule.equals("mono"))) d = draw(6);
            if (rule.equals("mono")) {
                if (b[i] != FREE) continue;
                if (nb.equals("rand") && lazy) d = draw(6);
                if (d >= 0) {
                    int xx = x + O[d][0], yy = y + O[d][1], zz = z + O[d][2];
                    if (xx < 0 || yy < 0 || zz < 0 || xx >= W || yy >= D || zz >= H) continue;
                    byte src = b[(yy * H + zz) * W + xx];
                    if (src != FREE) { b[i] = src; if (FREE == UNSET) unset--; }
                } else {
                    for (int q = 0; q < 6; q++) {
                        int xx = x + O[q][0], yy = y + O[q][1], zz = z + O[q][2];
                        if (xx < 0 || yy < 0 || zz < 0 || xx >= W || yy >= D || zz >= H) continue;
                        byte src = b[(yy * H + zz) * W + xx];
                        if (src != FREE) { b[i] = src; if (FREE == UNSET) unset--; break; }
                    }
                }
            } else {
                if (d < 0) d = 0;
                int xx = x + O[d][0], yy = y + O[d][1], zz = z + O[d][2];
                if (xx < 0 || yy < 0 || zz < 0 || xx >= W || yy >= D || zz >= H) continue;
                b[i] = b[(yy * H + zz) * W + xx];
            }
        }
        long h = 1469598103934665603L;
        for (int i = 0; i < b.length; i++) { if (b[i] == UNSET) b[i] = 0; h ^= (b[i] & 0xff); h *= 1099511628211L; }
        long state = 0;
        java.lang.reflect.Field f = Random.class.getDeclaredField("seed"); f.setAccessible(true);
        state = ((java.util.concurrent.atomic.AtomicLong) f.get(R)).get();
        System.out.printf("fnv64 %016x steps %d state_out %d%n", h, R.calls, state);
    }
}
