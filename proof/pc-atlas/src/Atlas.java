import java.awt.Graphics2D;
import java.awt.image.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.zip.GZIPOutputStream;
import javax.imageio.ImageIO;

/**
 * Generator atlas sweep (PC lane). Full size 256x256x64: x width, y vertical depth, z height,
 * address (y*256+z)*256+x as in the decompiled Level. Compact outputs only.
 * Usage: java Atlas <outDir> <threads> [smoke]
 */
public final class Atlas {
    static final int W = 256, H = 256, D = 64, V = W * H * D;
    static final String[] NAMES = {"", "firstFace", "lastFace", "randomFace", "emptyTarget", "push",
        "nearby", "copyAir", "flatIndex", "interior", "axisAndSign"};
    static final int[] S_GRID = {256, 1024, 4096, 16384};
    static final int[] B_GRID = {64, 42, 32};
    static final int[] T_MULT = {5, 10, 25, 50};
    static final int REPS = 3;
    static final long EDEN_CAP_MULT = 400; // safety cap; hitting it is recorded
    // representative raw world per variant
    static final int REP_S = 1024, REP_B = 42, REP_T = 25;

    static File out;
    static boolean smoke;
    static final AtomicInteger doneJobs = new AtomicInteger();
    static int totalJobs;
    static PrintWriter csv;

    static long mix(String key) {
        long z = 0x9E3779B97F4A7C15L;
        for (char c : key.toCharArray()) z = (z ^ c) * 0xBF58476D1CE4E5B9L;
        z ^= z >>> 31; z *= 0x94D049BB133111EBL; z ^= z >>> 29;
        return z & ((1L << 48) - 1);
    }

    // ---------------------------------------------------------------- scratch per thread
    static final class Scratch {
        final int[] label = new int[V];
        final int[] queue = new int[V];
        int[] compSize = new int[1 << 16];
        byte[] compFlag = new byte[1 << 16]; // bit0 air, bit1 touches top, bit2 touches side/bottom
    }
    static final ThreadLocal<Scratch> SCR = ThreadLocal.withInitial(Scratch::new);

    // ---------------------------------------------------------------- analysis
    static final class Stats {
        double solidFraction;
        double[] solidByY = new double[D];
        int[] heightHist = new int[D + 1]; // index h+1, h=-1 empty column
        int[] heights = new int[W * H];   // z*W+x
        double heightMean, heightStd, roughness;
        int emptyColumns;
        int airComponents, solidComponents, largestSolid;
        int enclosedCount, enclosedInterior, largestEnclosed;
        long enclosedVoxels, airVoxels, airTopVoxels;
        int[] enclosedLog2 = new int[24];
        long grassSky, grassExposed;
    }

    static boolean solid(byte b) { return b > 0; }

    static Stats analyze(byte[] m, Scratch s) {
        Stats st = new Stats();
        long totalSolid = 0;
        for (int y = 0; y < D; y++) {
            int c = 0, base = y << 16;
            for (int i = 0; i < W * H; i++) if (m[base + i] > 0) c++;
            st.solidByY[y] = c / (double)(W * H);
            totalSolid += c;
        }
        st.solidFraction = totalSolid / (double)V;
        double sum = 0, sum2 = 0; int n = 0;
        for (int z = 0; z < H; z++) for (int x = 0; x < W; x++) {
            int y = D - 1;
            while (y >= 0 && m[(y << 16) | (z << 8) | x] <= 0) y--;
            st.heights[z * W + x] = y;
            st.heightHist[y + 1]++;
            if (y < 0) st.emptyColumns++; else { sum += y; sum2 += (double)y * y; n++; }
        }
        if (n > 0) { st.heightMean = sum / n; st.heightStd = Math.sqrt(Math.max(0, sum2 / n - st.heightMean * st.heightMean)); }
        double rs = 0; int rn = 0;
        for (int z = 0; z < H; z++) for (int x = 0; x < W; x++) {
            int h = st.heights[z * W + x]; if (h < 0) continue;
            if (x + 1 < W && st.heights[z * W + x + 1] >= 0) { rs += Math.abs(h - st.heights[z * W + x + 1]); rn++; }
            if (z + 1 < H && st.heights[(z + 1) * W + x] >= 0) { rs += Math.abs(h - st.heights[(z + 1) * W + x]); rn++; }
        }
        st.roughness = rn > 0 ? rs / rn : 0;
        // grass masks
        for (int i = 0; i < V; i++) if (m[i] > 0) {
            int y = i >>> 16;
            if (y == D - 1 || m[i + (1 << 16)] <= 0) st.grassExposed++;
        }
        for (int c = 0; c < W * H; c++) if (st.heights[c] >= 0) st.grassSky++;
        // components (6-connected), air and solid separately
        int[] label = s.label, q = s.queue;
        Arrays.fill(label, 0);
        int comps = 0;
        for (int i0 = 0; i0 < V; i0++) {
            if (label[i0] != 0) continue;
            boolean air = m[i0] <= 0;
            comps++;
            if (comps >= s.compSize.length) {
                s.compSize = Arrays.copyOf(s.compSize, s.compSize.length * 2);
                s.compFlag = Arrays.copyOf(s.compFlag, s.compFlag.length * 2);
            }
            int head = 0, tail = 0, size = 0; byte flag = (byte)(air ? 1 : 0);
            label[i0] = comps; q[tail++] = i0;
            while (head < tail) {
                int i = q[head++]; size++;
                int x = i & 255, z = (i >>> 8) & 255, y = i >>> 16;
                if (y == D - 1) flag |= 2;
                if (x == 0 || x == W - 1 || z == 0 || z == H - 1 || y == 0) flag |= 4;
                if (x > 0) { int j = i - 1; if (label[j] == 0 && (m[j] <= 0) == air) { label[j] = comps; q[tail++] = j; } }
                if (x < W - 1) { int j = i + 1; if (label[j] == 0 && (m[j] <= 0) == air) { label[j] = comps; q[tail++] = j; } }
                if (z > 0) { int j = i - 256; if (label[j] == 0 && (m[j] <= 0) == air) { label[j] = comps; q[tail++] = j; } }
                if (z < H - 1) { int j = i + 256; if (label[j] == 0 && (m[j] <= 0) == air) { label[j] = comps; q[tail++] = j; } }
                if (y > 0) { int j = i - 65536; if (label[j] == 0 && (m[j] <= 0) == air) { label[j] = comps; q[tail++] = j; } }
                if (y < D - 1) { int j = i + 65536; if (label[j] == 0 && (m[j] <= 0) == air) { label[j] = comps; q[tail++] = j; } }
            }
            s.compSize[comps] = size; s.compFlag[comps] = flag;
            if (air) {
                st.airComponents++; st.airVoxels += size;
                if ((flag & 2) != 0) st.airTopVoxels += size;
                else {
                    st.enclosedCount++; st.enclosedVoxels += size;
                    if ((flag & 4) == 0) st.enclosedInterior++;
                    if (size > st.largestEnclosed) st.largestEnclosed = size;
                    st.enclosedLog2[31 - Integer.numberOfLeadingZeros(size)]++;
                }
            } else {
                st.solidComponents++;
                if (size > st.largestSolid) st.largestSolid = size;
            }
        }
        return st;
    }

    // ---------------------------------------------------------------- images
    static final int SKY = rgb(0.5, 0.8, 1.0);
    static int rgb(double r, double g, double b) {
        return ((int)Math.round(Math.min(1, r) * 255) << 16) | ((int)Math.round(Math.min(1, g) * 255) << 8) | (int)Math.round(Math.min(1, b) * 255);
    }

    static BufferedImage heightImage(Stats st) {
        BufferedImage im = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster r = im.getRaster();
        for (int z = 0; z < H; z++) for (int x = 0; x < W; x++)
            r.setSample(x, z, 0, (st.heights[z * W + x] + 1) * 255 / D);
        return im;
    }

    static final IndexColorModel SLICE_CM;
    static {
        byte[] R = {(byte)160, (byte)134, (byte)128, (byte)20, (byte)90}, G = {(byte)160, (byte)96, (byte)204, (byte)24, (byte)40}, B = {(byte)160, (byte)67, (byte)255, (byte)90, (byte)140};
        SLICE_CM = new IndexColorModel(8, 5, R, G, B);
    }
    /** 0 rock grey, 1 dirt brown, 2 air open to top (sky blue), 3 enclosed air touching side/bottom (navy), 4 interior enclosed air (purple). */
    static BufferedImage sliceImage(byte[] m, Scratch s, int y) {
        BufferedImage im = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_INDEXED, SLICE_CM);
        WritableRaster r = im.getRaster();
        for (int z = 0; z < H; z++) for (int x = 0; x < W; x++) {
            int i = (y << 16) | (z << 8) | x;
            int c;
            if (m[i] == 1) c = 0; else if (m[i] == 2) c = 1;
            else { byte f = s.compFlag[s.label[i]]; c = (f & 2) != 0 ? 2 : ((f & 4) != 0 ? 3 : 4); }
            r.setSample(x, z, 0, c);
        }
        return im;
    }

    /** First-person ray cast. Eye at (128.5, surface+1+1.62, 128.5) looking +x (yRot 90, right = +z), 70 deg vertical FOV, pitch 0. */
    static BufferedImage view(byte[] m, Stats st, int w, int h, double[] eyeOut) {
        int surf = st.heights[128 * W + 128];
        double ex = 128.5, ey = surf + 1 + 1.62, ez = 128.5;
        eyeOut[0] = surf; eyeOut[1] = ey;
        int[] lightDepth = new int[W * H];
        for (int c = 0; c < W * H; c++) lightDepth[c] = st.heights[c] + 1;
        BufferedImage im = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        double tv = Math.tan(Math.toRadians(35)), th = tv * w / (double)h;
        for (int j = 0; j < h; j++) for (int i = 0; i < w; i++) {
            double dx = 1, dz = ((i + 0.5) / w * 2 - 1) * th, dy = (1 - (j + 0.5) / h * 2) * tv;
            double len = Math.sqrt(dx * dx + dy * dy + dz * dz); dx /= len; dy /= len; dz /= len;
            im.setRGB(i, j, trace(m, lightDepth, ex, ey, ez, dx, dy, dz));
        }
        return im;
    }

    static int trace(byte[] m, int[] lightDepth, double ex, double ey, double ez, double dx, double dy, double dz) {
        int cx = (int)Math.floor(ex), cy = (int)Math.floor(ey), cz = (int)Math.floor(ez);
        int sx = dx > 0 ? 1 : (dx < 0 ? -1 : 0), sy = dy > 0 ? 1 : (dy < 0 ? -1 : 0), sz = dz > 0 ? 1 : (dz < 0 ? -1 : 0);
        double tdx = sx != 0 ? Math.abs(1 / dx) : Double.POSITIVE_INFINITY;
        double tdy = sy != 0 ? Math.abs(1 / dy) : Double.POSITIVE_INFINITY;
        double tdz = sz != 0 ? Math.abs(1 / dz) : Double.POSITIVE_INFINITY;
        double tmx = sx > 0 ? (cx + 1 - ex) * tdx : (sx < 0 ? (ex - cx) * tdx : Double.POSITIVE_INFINITY);
        double tmy = sy > 0 ? (cy + 1 - ey) * tdy : (sy < 0 ? (ey - cy) * tdy : Double.POSITIVE_INFINITY);
        double tmz = sz > 0 ? (cz + 1 - ez) * tdz : (sz < 0 ? (ez - cz) * tdz : Double.POSITIVE_INFINITY);
        int axis = -1; double t = 0;
        for (int step = 0; step < 4000; step++) {
            if (tmx < tmy && tmx < tmz) { cx += sx; t = tmx; tmx += tdx; axis = 0; }
            else if (tmy < tmz) { cy += sy; t = tmy; tmy += tdy; axis = 1; }
            else { cz += sz; t = tmz; tmz += tdz; axis = 2; }
            if ((cx < 0 && sx <= 0) || (cx >= W && sx >= 0) || (cz < 0 && sz <= 0) || (cz >= H && sz >= 0)
                || (cy < 0 && sy <= 0) || (cy >= D && sy >= 0)) return SKY;
            if (cx < 0 || cx >= W || cy < 0 || cy >= D || cz < 0 || cz >= H) continue;
            byte b = m[(cy << 16) | (cz << 8) | cx];
            if (b <= 0) continue;
            // face hit: normal points back toward the ray
            int ax = cx, ay = cy, az = cz;
            double shade; boolean top = false;
            if (axis == 0) { ax -= sx; shade = 0.6; }
            else if (axis == 1) { ay -= sy; if (sy < 0) { shade = 1.0; top = true; } else shade = 0.5; }
            else { az -= sz; shade = 0.8; }
            boolean lit = ay >= D || ax < 0 || ax >= W || az < 0 || az >= H || ay >= lightDepth[az * W + ax];
            double r, g, bb;
            boolean grassTop = top && (cy == D - 1 || m[((cy + 1) << 16) | (cz << 8) | cx] <= 0);
            if (grassTop) { r = 106 / 255.0; g = 170 / 255.0; bb = 64 / 255.0; }
            else if (b == 2) { r = 134 / 255.0; g = 96 / 255.0; bb = 67 / 255.0; }
            else { r = g = bb = 130 / 255.0; }
            double k = shade * (lit ? 1.0 : 0.55);
            // faint block edges for legibility
            double px = ex + dx * t, py = ey + dy * t, pz = ez + dz * t;
            double u = axis == 0 ? py : px, v = axis == 2 ? py : pz;
            u -= Math.floor(u); v -= Math.floor(v);
            double e = 0.035 + t * 0.0012;
            if (u < e || u > 1 - e || v < e || v > 1 - e) k *= 0.85;
            return rgb(r * k, g * k, bb * k);
        }
        return SKY;
    }

    static void png(BufferedImage im, File f) throws IOException { ImageIO.write(im, "png", f); }

    // ---------------------------------------------------------------- JSON
    static String arr(double[] a) { StringBuilder b = new StringBuilder("["); for (int i = 0; i < a.length; i++) { if (i > 0) b.append(','); b.append(String.format(Locale.ROOT, "%.5f", a[i])); } return b.append(']').toString(); }
    static String arr(int[] a) { StringBuilder b = new StringBuilder("["); for (int i = 0; i < a.length; i++) { if (i > 0) b.append(','); b.append(a[i]); } return b.append(']').toString(); }
    static String f(double d) { return String.format(Locale.ROOT, "%.5f", d); }

    static String json(Map<String, Object> head, Stats st, double[] eye) {
        StringBuilder b = new StringBuilder("{\n");
        for (Map.Entry<String, Object> e : head.entrySet()) {
            Object v = e.getValue();
            b.append("  \"").append(e.getKey()).append("\": ").append(v instanceof RawJson ? ((RawJson)v).s : v instanceof String ? "\"" + v + "\"" : String.valueOf(v)).append(",\n");
        }
        b.append("  \"solidFraction\": ").append(f(st.solidFraction)).append(",\n");
        b.append("  \"solidFractionByY\": ").append(arr(st.solidByY)).append(",\n");
        b.append("  \"surfaceHeightHist_hPlus1\": ").append(arr(st.heightHist)).append(",\n");
        b.append("  \"surfaceHeightMean\": ").append(f(st.heightMean)).append(",\n");
        b.append("  \"surfaceHeightStd\": ").append(f(st.heightStd)).append(",\n");
        b.append("  \"surfaceRoughnessMeanAbsNeighbourDiff\": ").append(f(st.roughness)).append(",\n");
        b.append("  \"emptyColumns\": ").append(st.emptyColumns).append(",\n");
        b.append("  \"airComponents\": ").append(st.airComponents).append(",\n");
        b.append("  \"enclosedAirComponents\": ").append(st.enclosedCount).append(",\n");
        b.append("  \"enclosedAirInteriorComponents\": ").append(st.enclosedInterior).append(",\n");
        b.append("  \"enclosedAirVoxels\": ").append(st.enclosedVoxels).append(",\n");
        b.append("  \"largestEnclosedAir\": ").append(st.largestEnclosed).append(",\n");
        b.append("  \"enclosedAirSizeHistLog2\": ").append(arr(st.enclosedLog2)).append(",\n");
        b.append("  \"airVoxels\": ").append(st.airVoxels).append(",\n");
        b.append("  \"airFractionReachingTop\": ").append(f(st.airVoxels > 0 ? st.airTopVoxels / (double)st.airVoxels : 0)).append(",\n");
        b.append("  \"solidComponents\": ").append(st.solidComponents).append(",\n");
        b.append("  \"largestSolidComponent\": ").append(st.largestSolid).append(",\n");
        b.append("  \"grassSkyCount\": ").append(st.grassSky).append(",\n");
        b.append("  \"grassExposedCount\": ").append(st.grassExposed).append(",\n");
        b.append("  \"cameraSurfaceY\": ").append((int)eye[0]).append(",\n");
        b.append("  \"cameraEyeY\": ").append(f(eye[1])).append("\n}\n");
        return b.toString();
    }

    // ---------------------------------------------------------------- outputs per world state
    static synchronized void csvRow(String row) { csv.println(row); csv.flush(); }

    static void emit(File dir, String stem, byte[] m, Map<String, Object> head, boolean raw) throws IOException {
        Scratch s = SCR.get();
        Stats st = analyze(m, s);
        double[] eye = new double[2];
        png(heightImage(st), new File(dir, stem + "_height.png"));
        for (int y : new int[]{20, 32, 44}) png(sliceImage(m, s, y), new File(dir, stem + "_y" + y + ".png"));
        png(view(m, st, 640, 360, eye), new File(dir, stem + "_view.png"));
        try (Writer w = new OutputStreamWriter(new FileOutputStream(new File(dir, stem + ".json")), "UTF-8")) { w.write(json(head, st, eye)); }
        if (raw) try (OutputStream o = new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(new File(dir, stem + "_blocks.u8.gz"))))) { o.write(m); }
        csvRow(String.format(Locale.ROOT, "%s,%s,%s,%s,%s,%s,%s,%s,%.5f,%.3f,%.3f,%.3f,%d,%d,%d,%.5f,%d",
            head.get("variant"), head.get("S"), head.get("B"), head.get("rep"), head.get("T_over_V"), head.get("attempts"),
            head.get("randomCalls"), head.get("seconds"), st.solidFraction, st.heightMean, st.heightStd, st.roughness,
            st.enclosedCount, st.largestEnclosed, st.emptyColumns,
            st.airVoxels > 0 ? st.airTopVoxels / (double)st.airVoxels : 0, st.solidComponents));
    }

    // ---------------------------------------------------------------- jobs
    static void monotoneJob(int v, int S, int B, int rep) throws IOException {
        String key = "v" + v + "|" + S + "|" + B + "|" + rep;
        long seed = mix(key);
        String vname = String.format("v%02d-%s", v, NAMES[v]);
        File dir = new File(out, "worlds/" + vname); dir.mkdirs();
        VariantsR lv = new VariantsR(W, H, D);
        lv.rng = new Random(seed);
        long t0 = System.nanoTime();
        lv.seed(S, B);
        long done = 0;
        int[] mults = smoke ? new int[]{5} : T_MULT;
        for (int k : mults) {
            long target = (long)k * V;
            lv.grow(v, target - done); done = target;
            double secs = (System.nanoTime() - t0) / 1e9;
            Map<String, Object> head = new LinkedHashMap<>();
            head.put("variant", vname); head.put("method", NAMES[v]); head.put("S", S); head.put("B", B); head.put("rep", rep);
            head.put("javaRandomSeed48", seed); head.put("T_over_V", k); head.put("attempts", done);
            head.put("randomCalls", lv.calls); head.put("seconds", f(secs));
            head.put("note", "T checkpoints share one stream: world at kV is the 5V..kV prefix of the same run");
            String stem = "S" + S + "_B" + B + "_r" + rep + "_T" + k + "V";
            emit(dir, stem, lv.blocks, head, S == REP_S && B == REP_B && rep == 0 && k == REP_T);
        }
        prog(vname + " " + key);
    }

    static final class RawJson { final String s; RawJson(String s) { this.s = s; } }
    static File GEN;
    /** Surviving GEN family via the validated gen CLI: base id with K and ntypes overrides (air share = 1/ntypes). */
    static void uJob(String base, int K, int ntypes, int rep) throws Exception {
        String vname = "u-" + base;
        String spec = base + ":K=" + K + ",ntypes=" + ntypes;
        String key = spec + "|" + rep;
        String stem = "K" + K + "_n" + ntypes + "_r" + rep;
        File dir = new File(out, "worlds/" + vname); dir.mkdirs();
        if (new File(dir, stem + ".json").exists()) { prog("skip " + key); return; }
        long state = mix(key);
        File tmp = File.createTempFile("ublk", ".blocks");
        long t0 = System.nanoTime();
        Process pr = new ProcessBuilder(GEN.getPath(), "run", spec, Long.toString(state), tmp.getPath()).redirectErrorStream(true).start();
        String outp = new String(pr.getInputStream().readAllBytes(), "UTF-8").trim();
        if (pr.waitFor() != 0) throw new IOException("gen failed: " + outp);
        String line = outp.substring(outp.lastIndexOf((char)10) + 1).trim();
        byte[] m = java.nio.file.Files.readAllBytes(tmp.toPath()); tmp.delete();
        if (m.length != V) throw new IOException("bad size " + m.length);
        Map<String, Object> head = new LinkedHashMap<>();
        head.put("variant", vname); head.put("method", "GEN " + spec); head.put("S", K); head.put("B", 42); head.put("rep", rep);
        head.put("ntypes", ntypes); head.put("airSeedShare", "1/" + ntypes);
        head.put("state48_in", state); head.put("T_over_V", "spec"); head.put("attempts", "spec"); head.put("randomCalls", "see gen");
        head.put("seconds", f((System.nanoTime() - t0) / 1e9)); head.put("gen", new RawJson(line));
        emit(dir, stem, m, head, K == 1024 && ntypes == 6 && rep == 0);
        prog(key);
    }

    static void edenJob(int types, boolean rnd, int S, int B, int rep) throws IOException {
        String vname = "c4-eden-" + (types == 2 ? "airsolid" : "airrockdirt") + "-" + (rnd ? "randomNeighbour" : "firstNeighbour");
        String key = vname + "|" + S + "|" + B + "|" + rep;
        long seed = mix(key);
        File dir = new File(out, "worlds/" + vname); dir.mkdirs();
        if (new File(dir, "S" + S + "_B" + B + "_r" + rep + ".json").exists()) { prog("skip " + key); return; }
        Eden e = new Eden(W, H, D);
        e.rng = new Random(seed);
        long t0 = System.nanoTime();
        e.seed(S, B, types);
        long seedCalls = e.calls;
        long cap = (smoke ? 60L : EDEN_CAP_MULT) * V;
        long att = e.grow(rnd, cap);
        double secs = (System.nanoTime() - t0) / 1e9;
        Map<String, Object> head = new LinkedHashMap<>();
        head.put("variant", vname); head.put("method", "C4 Eden"); head.put("types", types);
        head.put("neighbourRule", rnd ? "random" : "first(-x,+x,-y,+y,-z,+z)");
        head.put("S", S); head.put("B", B); head.put("rep", rep); head.put("javaRandomSeed48", seed);
        head.put("T_over_V", f(att / (double)V)); head.put("attempts", att); head.put("seedCalls", seedCalls);
        head.put("randomCalls", e.calls); head.put("unsetRemaining", e.unset); head.put("capHit", e.unset > 0);
        head.put("seconds", f(secs));
        // any remaining UNSET (cap hit) is treated as air for analysis
        if (e.unset > 0) for (int i = 0; i < V; i++) if (e.blocks[i] == Eden.UNSET) e.blocks[i] = 0;
        String stem = "S" + S + "_B" + B + "_r" + rep;
        emit(dir, stem, e.blocks, head, S == REP_S && B == REP_B && rep == 0);
        prog(key + " attempts/V=" + f(att / (double)V));
    }

    static final long T_START = System.nanoTime();
    static void prog(String what) {
        int d = doneJobs.incrementAndGet();
        System.out.printf(Locale.ROOT, "[%7.1fs] %d/%d %s%n", (System.nanoTime() - T_START) / 1e9, d, totalJobs, what);
    }

    public static void main(String[] args) throws Exception {
        out = new File(args[0]); out.mkdirs();
        int threads = Integer.parseInt(args[1]);
        smoke = args.length > 2 && args[2].equals("smoke");
        File csvf = new File(out, "summary.csv"); boolean fresh = !csvf.exists();
        csv = new PrintWriter(new OutputStreamWriter(new FileOutputStream(csvf, true), "UTF-8"));
        GEN = new File(args.length > 3 ? args[3] : "../gen/gen.exe");
        if (fresh) csv.println("variant,S,B,rep,T_over_V,attempts,randomCalls,seconds,solidFraction,surfaceMean,surfaceStd,roughness,enclosedAirComponents,largestEnclosedAir,emptyColumns,airFractionReachingTop,solidComponents");
        List<Callable<Void>> jobs = new ArrayList<>();
        int[] Sg = smoke ? new int[]{1024} : S_GRID, Bg = smoke ? new int[]{42} : B_GRID;
        int reps = smoke ? 1 : REPS;
        // Eden first (variable length), then the heavier monotone jobs, largest S last
        for (int S : Sg) for (int B : Bg) for (int r = 0; r < reps; r++)
            for (int types : new int[]{2, 3}) for (boolean rnd : new boolean[]{false, true}) {
                final int fs = S, fb = B, fr = r, ft = types; final boolean fn = rnd;
                jobs.add(() -> { edenJob(ft, fn, fs, fb, fr); return null; });
            }
        // Monotone variants dropped (excluded by GEN look-alike test); surviving U family instead
        for (String base : new String[]{"U6s", "U6sm", "U6r", "U6rm"})
            for (int K : smoke ? new int[]{1024} : new int[]{16, 64, 256, 1024, 4096})
                for (int nt : smoke ? new int[]{6} : new int[]{2, 4, 6, 8, 12})
                    for (int r = 0; r < (smoke ? 1 : 4); r++) {
                        final String fb2 = base; final int fk = K, fn2 = nt, fr = r;
                        jobs.add(() -> { uJob(fb2, fk, fn2, fr); return null; });
                    }
        totalJobs = jobs.size();
        System.out.println("jobs=" + totalJobs + " threads=" + threads);
        ExecutorService ex = Executors.newFixedThreadPool(threads);
        List<Future<Void>> fs = ex.invokeAll(jobs);
        ex.shutdown();
        int failed = 0;
        for (Future<Void> fu : fs) try { fu.get(); } catch (ExecutionException ee) { failed++; ee.getCause().printStackTrace(); }
        csv.close();
        System.out.printf(Locale.ROOT, "TOTAL %.1fs failed=%d%n", (System.nanoTime() - T_START) / 1e9, failed);
    }
}
