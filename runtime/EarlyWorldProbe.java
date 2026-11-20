import java.lang.reflect.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;

/** Executes the publisher's Level class without replacing its world logic. */
public final class EarlyWorldProbe {
    public static void main(String[] args) throws Exception {
        Class<?> type = Class.forName("com.mojang.rubydung.level.Level");
        Object level = type.getConstructor(int.class, int.class, int.class)
            .newInstance(256, 256, 64);
        Field field = type.getDeclaredField("blocks");
        field.setAccessible(true);
        byte[] blocks = (byte[]) field.get(level);
        int[] layerSolid = new int[64];
        long[] histogram = new long[256];
        for (int i = 0; i < blocks.length; i++) {
            int block = blocks[i] & 255;
            histogram[block]++;
            if (block != 0) layerSolid[i / 65536]++;
        }
        System.out.println("{\"loaded_level_class\":\"" + type.getName() + "\",");
        System.out.println("\"block_bytes\":" + blocks.length + ",");
        System.out.println("\"block_sha256\":\"" + HexFormat.of().formatHex(
            MessageDigest.getInstance("SHA-256").digest(blocks)) + "\",");
        System.out.println("\"non_air_by_y\":" + Arrays.toString(layerSolid) + ",");
        System.out.println("\"block_histogram\":" + Arrays.toString(histogram) + ",");
        type.getMethod("save").invoke(level);
        System.out.println("\"saved_bytes\":" + Files.size(Path.of("level.dat")) + "}");
    }
}
