/** Independent hypotheses, not recovered Notch source. Java 6 source/API subset. */
public final class VariantsR {
    public final int width, height, depth;
    public final byte[] blocks;
    // PC-ATLAS EDIT: per-instance java.util.Random (same LCG/nextDouble as Math.random) + call counter, so 28 threads do not contend on the global Random.
    public java.util.Random rng = new java.util.Random(0L);
    public long calls;
    private double rnd() { calls++; return rng.nextDouble(); }

    public VariantsR(int w, int h, int d) {
        if (w < 3 || h < 3 || d < 3 || (long)w * h * d > Integer.MAX_VALUE)
            throw new IllegalArgumentException("dimensions");
        width = w; height = h; depth = d;
        blocks = new byte[w * h * d];
    }
    private int index(int x, int y, int z) {
        return (y * this.height + z) * this.width + x;
    }
    private boolean inside(int x, int y, int z) {
        return x >= 0 && y >= 0 && z >= 0 && x < width && y < depth && z < height;
    }
    public boolean isTile(int x, int y, int z) {
        return inside(x, y, z) && blocks[index(x, y, z)] == 1;
    }
    // Shared initialization: S attempts, with replacement. No uniqueness retries.
    public void seed(int count, int seedHeight) {
        if (count < 0 || seedHeight < 1 || seedHeight > depth)
            throw new IllegalArgumentException("seeds");
        for (int i = 0; i < count; i++) {
            int x = (int)(rnd() * this.width);
            int y = (int)(rnd() * seedHeight);
            int z = (int)(rnd() * this.height);
            this.blocks[index(x, y, z)] = 1;
        }
    }
    // 1: first occupied face wins; air never replaces solid.
    public void firstFace(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            if (isTile(x - 1, y, z)) blocks[index(x, y, z)] = blocks[index(x - 1, y, z)];
            else if (isTile(x + 1, y, z)) blocks[index(x, y, z)] = blocks[index(x + 1, y, z)];
            else if (isTile(x, y - 1, z)) blocks[index(x, y, z)] = blocks[index(x, y - 1, z)];
            else if (isTile(x, y + 1, z)) blocks[index(x, y, z)] = blocks[index(x, y + 1, z)];
            else if (isTile(x, y, z - 1)) blocks[index(x, y, z)] = blocks[index(x, y, z - 1)];
            else if (isTile(x, y, z + 1)) blocks[index(x, y, z)] = blocks[index(x, y, z + 1)];
        }
    }
    // Direction numbering follows Tile.renderFace: -y,+y,-z,+z,-x,+x.
    private void copyFace(int x, int y, int z, int face, boolean copyAir, boolean push) {
        int xx = x, yy = y, zz = z;
        if (face == 0) yy--;
        if (face == 1) yy++;
        if (face == 2) zz--;
        if (face == 3) zz++;
        if (face == 4) xx--;
        if (face == 5) xx++;
        if (!inside(xx, yy, zz)) return;
        int a = index(x, y, z), b = index(xx, yy, zz);
        if (push) { if (blocks[a] == 1) blocks[b] = blocks[a]; }
        else if (copyAir || blocks[b] == 1) blocks[a] = blocks[b];
    }
    // 2: independent ifs, so last occupied face wins (binary-equivalent to 1).
    public void lastFace(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            for (int face = 0; face < 6; face++) copyFace(x, y, z, face, false, false);
        }
    }
    // 3: one sampled face; invalid neighbor wastes the attempt.
    public void randomFace(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            int face = (int)(rnd() * 6);
            copyFace(x, y, z, face, false, false);
        }
    }
    // 4: draw direction only for an empty target.
    public void emptyTarget(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            if (!isTile(x, y, z)) {
                int face = (int)(rnd() * 6);
                copyFace(x, y, z, face, false, false);
            }
        }
    }
    // 5: occupied source pushes into a sampled neighbor; conditional direction.
    public void push(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            if (isTile(x, y, z)) {
                int face = (int)(rnd() * 6);
                copyFace(x, y, z, face, false, true);
            }
        }
    }
    // 6: simple three-offset neighborhood, including diagonals and self.
    public void nearby(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            int xx = x + (int)(rnd() * 3) - 1;
            int yy = y + (int)(rnd() * 3) - 1;
            int zz = z + (int)(rnd() * 3) - 1;
            if (isTile(xx, yy, zz)) blocks[index(x, y, z)] = blocks[index(xx, yy, zz)];
        }
    }
    // 7: literal neighbor-state copying, including air (voter dynamics).
    public void copyAir(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            int face = (int)(rnd() * 6);
            copyFace(x, y, z, face, true, false);
        }
    }
    // 8: random array slot instead of three random coordinates.
    public void flatIndex(long steps) {
        for (long i = 0; i < steps; i++) {
            int p = (int)(rnd() * blocks.length);
            int x = p % width;
            int z = p / width % height;
            int y = p / (width * height);
            int face = (int)(rnd() * 6);
            copyFace(x, y, z, face, false, false);
        }
    }
    // 9: avoid boundary tests on the chosen neighbor by excluding target edges.
    public void interior(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * (width - 2)) + 1;
            int y = (int)(rnd() * (depth - 2)) + 1;
            int z = (int)(rnd() * (height - 2)) + 1;
            int face = (int)(rnd() * 6);
            copyFace(x, y, z, face, false, false);
        }
    }
    // 10: axis and sign chosen independently, rather than one six-way draw.
    public void axisAndSign(long steps) {
        for (long i = 0; i < steps; i++) {
            int x = (int)(rnd() * width);
            int y = (int)(rnd() * depth);
            int z = (int)(rnd() * height);
            int axis = (int)(rnd() * 3);
            int sign = (int)(rnd() * 2);
            copyFace(x, y, z, axis * 2 + sign, false, false);
        }
    }
    public void grow(int variant, long steps) {
        if (steps < 0) throw new IllegalArgumentException("steps");
        switch (variant) {
        case 1: firstFace(steps); break;
        case 2: lastFace(steps); break;
        case 3: randomFace(steps); break;
        case 4: emptyTarget(steps); break;
        case 5: push(steps); break;
        case 6: nearby(steps); break;
        case 7: copyAir(steps); break;
        case 8: flatIndex(steps); break;
        case 9: interior(steps); break;
        case 10: axisAndSign(steps); break;
        default: throw new IllegalArgumentException("variant");
        }
    }
    /** Render-only mask: grass for highest occupied tile per column. No RNG. */
    public boolean[] grassSky() {
        boolean[] grass = new boolean[blocks.length];
        for (int x = 0; x < width; x++) {
            for (int z = 0; z < height; z++) {
                int y = depth - 1;
                while (y >= 0 && !isTile(x, y, z)) y--;
                if (y >= 0) grass[index(x, y, z)] = true;
            }
        }
        return grass;
    }
    /** Alternative: grass wherever a solid has air immediately above. No RNG. */
    public boolean[] grassExposed() {
        boolean[] grass = new boolean[blocks.length];
        for (int x = 0; x < width; x++)
            for (int y = 0; y < depth; y++)
                for (int z = 0; z < height; z++)
                    grass[index(x, y, z)] = isTile(x, y, z) && !isTile(x, y + 1, z);
        return grass;
    }
    /** Tiny smoke test only; deliberately no full-size generation entry point. */
    public static void main(String[] args) {
        for (int v = 1; v <= 10; v++) {
            VariantsR level = new VariantsR(32, 32, 16);
            level.seed(64, 8);
            level.grow(v, 10000L);
            int solid = 0, sky = 0, exposed = 0;
            boolean[] a = level.grassSky(), b = level.grassExposed();
            for (int i = 0; i < level.blocks.length; i++) {
                if (level.blocks[i] != 0 && level.blocks[i] != 1) throw new AssertionError("id");
                if (level.blocks[i] == 1) solid++;
                if (a[i]) sky++;
                if (b[i]) exposed++;
                if (a[i] && !b[i]) throw new AssertionError("grass");
            }
            if (v != 7 && solid < 1) throw new AssertionError("growth");
            System.out.println(v + ": solid=" + solid + " sky=" + sky + " exposed=" + exposed);
        }
    }
}
