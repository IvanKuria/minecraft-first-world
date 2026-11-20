import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/** Contact sheet per variant. Rows = S x B. Columns = T checkpoints (rep 0) for monotone variants, reps for C4.
 *  Each cell: first-person view (192x108) | top-down heightmap (108) | slice y=32 (108). Usage: java Sheets <atlasDir> */
public final class Sheets {
    static final int[] S = {256, 1024, 4096, 16384}, B = {64, 42, 32}, T = {5, 10, 25, 50};
    static final int VW = 192, VH = 108, CW = VW + 2 * VH + 8, CH = VH + 16, LW = 110, TOP = 40;

    public static void main(String[] a) throws Exception {
        File root = new File(a[0]), worlds = new File(root, "worlds"), sheets = new File(root, "sheets");
        sheets.mkdirs();
        for (File vd : worlds.listFiles()) {
            if (!vd.isDirectory()) continue;
            boolean eden = vd.getName().startsWith("c4");
            if (vd.getName().startsWith("u-")) { uSheet(vd, sheets); continue; }
            int cols = eden ? 3 : 4;
            BufferedImage im = new BufferedImage(LW + cols * CW, TOP + S.length * B.length * CH, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = im.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            g.setColor(new Color(24, 24, 28)); g.fillRect(0, 0, im.getWidth(), im.getHeight());
            g.setColor(Color.WHITE); g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 15));
            g.drawString(vd.getName() + (eden ? "   columns: rep 0,1,2 (grown to full)" : "   columns: T = 5V, 10V, 25V, 50V (rep 0)")
                + "   cell: view +x from (128,128) | heightmap | slice y=32", 8, 18);
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
            for (int c = 0; c < cols; c++) g.drawString(eden ? "rep " + c : "T=" + T[c] + "V", LW + c * CW + 4, TOP - 6);
            int row = 0;
            for (int s : S) for (int b : B) {
                int y0 = TOP + row * CH;
                g.setColor(Color.WHITE);
                g.drawString("S=" + s, 6, y0 + 40); g.drawString("B=" + b, 6, y0 + 56);
                for (int c = 0; c < cols; c++) {
                    String stem = eden ? "S" + s + "_B" + b + "_r" + c : "S" + s + "_B" + b + "_r0_T" + T[c] + "V";
                    int x0 = LW + c * CW;
                    draw(g, new File(vd, stem + "_view.png"), x0, y0, VW, VH);
                    draw(g, new File(vd, stem + "_height.png"), x0 + VW + 4, y0, VH, VH);
                    draw(g, new File(vd, stem + "_y32.png"), x0 + VW + VH + 8, y0, VH, VH);
                }
                row++;
            }
            g.dispose();
            ImageIO.write(im, "png", new File(sheets, vd.getName() + ".png"));
            System.out.println("sheet " + vd.getName());
        }
    }
    static void uSheet(File vd, File sheets) throws Exception {
        int[] K = {16, 64, 256, 1024, 4096}, NT = {2, 4, 6, 8, 12};
        BufferedImage im = new BufferedImage(LW + NT.length * CW, TOP + K.length * CH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = im.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setColor(new Color(24, 24, 28)); g.fillRect(0, 0, im.getWidth(), im.getHeight());
        g.setColor(Color.WHITE); g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 15));
        g.drawString(vd.getName() + "   rows: K seeds   columns: air-seed share 1/ntypes (rep 0)   cell: view +x | heightmap | slice y=32", 8, 18);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
        for (int c = 0; c < NT.length; c++) g.drawString("air 1/" + NT[c], LW + c * CW + 4, TOP - 6);
        for (int r = 0; r < K.length; r++) {
            int y0 = TOP + r * CH;
            g.setColor(Color.WHITE); g.drawString("K=" + K[r], 6, y0 + 50);
            for (int c = 0; c < NT.length; c++) {
                String stem = "K" + K[r] + "_n" + NT[c] + "_r0";
                int x0 = LW + c * CW;
                draw(g, new File(vd, stem + "_view.png"), x0, y0, VW, VH);
                draw(g, new File(vd, stem + "_height.png"), x0 + VW + 4, y0, VH, VH);
                draw(g, new File(vd, stem + "_y32.png"), x0 + VW + VH + 8, y0, VH, VH);
            }
        }
        g.dispose();
        ImageIO.write(im, "png", new File(sheets, vd.getName() + ".png"));
        System.out.println("sheet " + vd.getName());
    }
    static void draw(Graphics2D g, File f, int x, int y, int w, int h) throws Exception {
        if (!f.exists()) { g.setColor(Color.RED); g.drawRect(x, y, w - 1, h - 1); return; }
        g.drawImage(ImageIO.read(f), x, y, w, h, null);
    }
}
