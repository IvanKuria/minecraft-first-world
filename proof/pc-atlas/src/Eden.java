/**
 * IDEA-MAP C4: multi-type growth ("Eden" / Voronoi competition). A hypothesis, not recovered source.
 *
 * Array starts UNSET. S seed attempts each draw x, y (in band B), z, then a type
 * floor(U*types) with 0 = air, 1 = rock(solid), 2 = dirt(solid, only when types == 3).
 * Then growth attempts draw x, y, z; only an UNSET target can change:
 *   rule FIRST : copy the first SET 6-neighbour in the order -x,+x,-y,+y,-z,+z (same order as variant 1);
 *   rule RANDOM: draw face floor(U*6) (only for an UNSET target), copy that neighbour if it is SET.
 * Runs until no cell is UNSET; attempts are counted. Grass (render-only) = solid with air directly above.
 * U is a draw from a per-instance java.util.Random.nextDouble(), identical to Math.random() on a Random.
 * Exact call count: N = 4S + 3T (FIRST) or 4S + 3T + E (RANDOM, E = attempts whose target was UNSET).
 */
public final class Eden {
    public static final byte UNSET = -1;
    public final int width, height, depth;
    public final byte[] blocks;
    public java.util.Random rng = new java.util.Random(0L);
    public long calls;
    public long unset;
    private double rnd() { calls++; return rng.nextDouble(); }

    public Eden(int w, int h, int d) {
        width = w; height = h; depth = d;
        blocks = new byte[w * h * d];
        java.util.Arrays.fill(blocks, UNSET);
        unset = blocks.length;
    }
    private int index(int x, int y, int z) { return (y * height + z) * width + x; }
    private boolean inside(int x, int y, int z) {
        return x >= 0 && y >= 0 && z >= 0 && x < width && y < depth && z < height;
    }
    private boolean isSet(int x, int y, int z) { return inside(x, y, z) && blocks[index(x, y, z)] != UNSET; }

    public void seed(int count, int seedHeight, int types) {
        for (int i = 0; i < count; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * seedHeight);
            int z = (int)(rnd() * height);
            byte t = (byte)(int)(rnd() * types);
            int a = index(x, y, z);
            if (blocks[a] == UNSET) unset--;
            blocks[a] = t; // collisions overwrite, as in the variant-1 seed()
        }
    }

    /** Grow until full or until maxAttempts; returns attempts used. */
    public long grow(boolean randomNeighbour, long maxAttempts) {
        long i = 0;
        for (; i < maxAttempts && unset > 0; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            int a = index(x, y, z);
            if (blocks[a] != UNSET) continue;
            if (!randomNeighbour) {
                byte t;
                if (isSet(x - 1, y, z)) t = blocks[index(x - 1, y, z)];
                else if (isSet(x + 1, y, z)) t = blocks[index(x + 1, y, z)];
                else if (isSet(x, y - 1, z)) t = blocks[index(x, y - 1, z)];
                else if (isSet(x, y + 1, z)) t = blocks[index(x, y + 1, z)];
                else if (isSet(x, y, z - 1)) t = blocks[index(x, y, z - 1)];
                else if (isSet(x, y, z + 1)) t = blocks[index(x, y, z + 1)];
                else continue;
                blocks[a] = t; unset--;
            } else {
                int face = (int)(rnd() * 6);
                int xx = x, yy = y, zz = z;
                if (face == 0) yy--;
                if (face == 1) yy++;
                if (face == 2) zz--;
                if (face == 3) zz++;
                if (face == 4) xx--;
                if (face == 5) xx++;
                if (!isSet(xx, yy, zz)) continue;
                blocks[a] = blocks[index(xx, yy, zz)]; unset--;
            }
        }
        return i;
    }
}
