import com.mojang.rubydung.RubyDung;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;
import org.lwjgl.BufferUtils;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.GL11;

/** Calls the preserved game's own init/render methods with a controlled camera. */
public final class GameRenderProbe {
    private static void number(Object object, String name, float value) throws Exception {
        Field f = object.getClass().getField(name);
        f.setFloat(object, value);
    }
    public static void main(String[] args) throws Exception {
        RubyDung game = new RubyDung();
        try {
            game.init();
            Mouse.setGrabbed(false);
            Display.setTitle("Minecraft research: preserved rd-132211 renderer");
            int requestedWidth = Integer.getInteger("probe.width", 1024);
            int requestedHeight = Integer.getInteger("probe.height", 768);
            if (requestedWidth != 1024 || requestedHeight != 768) {
                Display.setDisplayMode(new org.lwjgl.opengl.DisplayMode(requestedWidth, requestedHeight));
                Field fw = RubyDung.class.getDeclaredField("width");
                Field fh = RubyDung.class.getDeclaredField("height");
                fw.setAccessible(true); fh.setAccessible(true);
                fw.setInt(game, requestedWidth); fh.setInt(game, requestedHeight);
            }
            Field field = RubyDung.class.getDeclaredField("player");
            field.setAccessible(true);
            Object player = field.get(game);
            float x = args.length > 0 ? Float.parseFloat(args[0]) : 128;
            float y = args.length > 1 ? Float.parseFloat(args[1]) : 48;
            float z = args.length > 2 ? Float.parseFloat(args[2]) : 128;
            float pitch = args.length > 3 ? Float.parseFloat(args[3]) : 20;
            float yaw = args.length > 4 ? Float.parseFloat(args[4]) : 0;
            number(player, "x", x); number(player, "xo", x);
            number(player, "y", y); number(player, "yo", y);
            number(player, "z", z); number(player, "zo", z);
            number(player, "xRot", pitch); number(player, "yRot", yaw);
            for (int i = 0; i < 300; i++) game.render(1.0f);
            int width = Display.getWidth(), height = Display.getHeight();
            ByteBuffer pixels = BufferUtils.createByteBuffer(width * height * 4);
            GL11.glReadBuffer(GL11.GL_FRONT);
            GL11.glReadPixels(0, 0, width, height, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, pixels);
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            for (int py = 0; py < height; py++) for (int px = 0; px < width; px++) {
                int i = (py * width + px) * 4;
                int r = pixels.get(i) & 255, g = pixels.get(i+1) & 255, b = pixels.get(i+2) & 255;
                image.setRGB(px, height-1-py, 0xff000000 | (r << 16) | (g << 8) | b);
            }
            ImageIO.write(image, "png", new File("actual-game-render.png"));
            System.out.println("Captured original game renderer " + width + "x" + height);
        } finally {
            Mouse.setGrabbed(false);
            game.destroy();
        }
    }
}
