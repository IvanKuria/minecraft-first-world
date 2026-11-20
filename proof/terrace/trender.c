/* TERRACE renderer: rd-132211 look with the AUTHENTIC terrain.png (tile 0 = grass texture, tile 1 = rock texture),
 * GL_NEAREST texel lookup floor(frac*15.984), per-face UV orientation and shades copied from rd-132211 Tile.render
 * (top/bottom 1.0, z faces 0.8, x faces 0.6; unlit x0.8 and GL_EXP fog density 0.2 colour 0x0E0B0A on eye depth),
 * 2D light-depth map (Level.calcLightDepths), sky (0.5,0.8,1.0).  Camera = footage pose convention
 * (yaw, pitch, FOV 70 vertical, 0.3 setback): camera = player - 0.3*forward.
 * Output 640x360 (or W H), 2x2 supersampled, P6 ppm.  Also writes an optional label image (face class per pixel).
 * usage: trender blocks.bin W H D px py pz yaw pitch outW outH grassrule gy out.ppm [labels.pgm]
 *   blocks: uint8 W*H*D, index (y*H+z)*W+x, nonzero = solid.  px,py,pz = PLAYER position (eye/camera before setback).
 *   grassrule: eq (y==gy grass-textured) | ge (y>=gy) | top (column-top solid) | topge (column top and y>=gy) | none
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int W, H, D, GR, GY; static uint8_t *B; static int *LD;
static unsigned char TEX[256 * 256 * 3]; /* terrain.png RGB */
static inline int solid(int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H) return 0;
    return B[((long)y * H + z) * W + x] != 0;
}
static inline int lightd(int x, int z) { return LD[x + z * W]; }
static inline int litcell(int x, int y, int z) { /* Level.getBrightness == 1.0 ? */
    if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H) return 1;
    return !(y < LD[x + z * W]);
}
static inline int isGrass(int x, int y, int z) {
    switch (GR) {
    case 0: return y == GY;
    case 1: return y >= GY;
    case 2: return y == lightd(x, z);
    case 3: return y == lightd(x, z) && y >= GY;
    default: return 0;
    }
}
static int load_tex(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    char m[3]; int w, h, mx; if (fscanf(f, "%2s %d %d %d", m, &w, &h, &mx) != 4 || w != 256 || h != 256) { fclose(f); return -1; }
    fgetc(f); size_t n = fread(TEX, 1, sizeof TEX, f); fclose(f); return n == sizeof TEX ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc < 16) { fprintf(stderr, "usage: trender blocks W H D px py pz yaw pitch outW outH grassrule gy tex.ppm out.ppm [labels.pgm]\n"); return 2; }
    W = atoi(argv[2]); H = atoi(argv[3]); D = atoi(argv[4]);
    double px = atof(argv[5]), py = atof(argv[6]), pz = atof(argv[7]), yaw = atof(argv[8]), pitch = atof(argv[9]);
    int ow = atoi(argv[10]), oh = atoi(argv[11]);
    const char *g = argv[12]; GY = atoi(argv[13]);
    GR = !strcmp(g, "eq") ? 0 : !strcmp(g, "ge") ? 1 : !strcmp(g, "top") ? 2 : !strcmp(g, "topge") ? 3 : 4;
    if (load_tex(argv[14])) { fprintf(stderr, "tex fail\n"); return 1; }
    long N = (long)W * H * D; B = malloc(N);
    FILE *f = fopen(argv[1], "rb"); if (!f || fread(B, 1, N, f) != (size_t)N) { fprintf(stderr, "read fail\n"); return 1; } fclose(f);
    LD = malloc(sizeof(int) * W * H);
    for (int x = 0; x < W; x++) for (int z = 0; z < H; z++) { int y = D - 1; while (y > 0 && !solid(x, y, z)) y--; LD[x + z * W] = y; }
    double s = sin(yaw), c = cos(yaw), sp = sin(pitch), cp = cos(pitch);
    double fx = s * cp, fy = -sp, fz = -c * cp;              /* forward */
    double ex = px - .3 * fx, ey = py - .3 * fy, ez = pz - .3 * fz; /* camera */
    double fpx = (oh / 2.0) / tan(35.0 * M_PI / 180.0);
    unsigned char *img = malloc((size_t)ow * oh * 3); unsigned char *lab = malloc((size_t)ow * oh);
    const int SS = 2;
    for (int j = 0; j < oh; j++) for (int i = 0; i < ow; i++) {
        double acc[3] = {0, 0, 0}; int lcount[8] = {0};
        for (int sj = 0; sj < SS; sj++) for (int si = 0; si < SS; si++) {
            double rx = (i + (si + .5) / SS - ow / 2.0) / fpx, ry = (j + (sj + .5) / SS - oh / 2.0) / fpx;
            double dx = c * rx - s * sp * ry + s * cp, dy = -cp * ry - sp, dz = s * rx + c * sp * ry - c * cp;
            double len = sqrt(dx * dx + dy * dy + dz * dz); dx /= len; dy /= len; dz /= len;
            double r = .5, gg = .8, b = 1.0; int cls = 0; /* 0 sky */
            int X = (int)floor(ex), Y = (int)floor(ey), Z = (int)floor(ez);
            int sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1, sz = dz > 0 ? 1 : -1;
            double tdx = fabs(1 / dx), tdy = fabs(1 / dy), tdz = fabs(1 / dz);
            double tx = (dx > 0 ? (X + 1 - ex) : (ex - X)) * tdx, ty = (dy > 0 ? (Y + 1 - ey) : (ey - Y)) * tdy, tz = (dz > 0 ? (Z + 1 - ez) : (ez - Z)) * tdz;
            double dist = 0; int face = -1;
            for (int st = 0; st < 2000; st++) {
                if (tx < ty && tx < tz) { X += sx; dist = tx; tx += tdx; face = 0; }
                else if (ty < tz) { Y += sy; dist = ty; ty += tdy; face = 1; }
                else { Z += sz; dist = tz; tz += tdz; face = 2; }
                if ((X < -1 && sx < 0) || (X > W && sx > 0) || (Z < -1 && sz < 0) || (Z > H && sz > 0) || (Y < -1 && sy < 0) || (Y > D + 64 && sy > 0)) break;
                if (solid(X, Y, Z)) {
                    double hx = ex + dx * dist, hy = ey + dy * dist, hz = ez + dz * dist;
                    double shade, u, v; int lit;
                    if (face == 1) {
                        if (sy < 0) { /* top face (y+1 side) */ shade = 1.0; lit = litcell(X, Y, Z); u = hx - X; v = hz - Z; cls = 1; }
                        else { /* bottom face */ shade = 1.0; lit = litcell(X, Y - 1, Z); u = hx - X; v = hz - Z; cls = 6; }
                    } else if (face == 2) {
                        shade = 0.8;
                        if (sz > 0) { /* hit z- face (z0) */ lit = litcell(X, Y, Z - 1); u = 1 - (hx - X); v = 1 - (hy - Y); cls = 2; }
                        else { /* z+ face */ lit = litcell(X, Y, Z + 1); u = hx - X; v = 1 - (hy - Y); cls = 3; }
                    } else {
                        shade = 0.6;
                        if (sx > 0) { /* x- face */ lit = litcell(X - 1, Y, Z); u = hz - Z; v = 1 - (hy - Y); cls = 4; }
                        else { /* x+ face */ lit = litcell(X + 1, Y, Z); u = 1 - (hz - Z); v = 1 - (hy - Y); cls = 5; }
                    }
                    if (u < 0) u = 0; if (u > .999999) u = .999999; if (v < 0) v = 0; if (v > .999999) v = .999999;
                    int tile = isGrass(X, Y, Z) ? 0 : 1;
                    int tu = (int)floor(u * 15.984), tv = (int)floor(v * 15.984);
                    const unsigned char *t = TEX + ((size_t)tv * 256 + tile * 16 + tu) * 3;
                    double br = shade * (lit ? 1.0 : 0.8);
                    r = t[0] / 255.0 * br; gg = t[1] / 255.0 * br; b = t[2] / 255.0 * br;
                    if (!lit) { double ez_ = dist * (dx * fx + dy * fy + dz * fz); double fo = exp(-0.2 * ez_);
                        r = r * fo + (14 / 255.0) * (1 - fo); gg = gg * fo + (11 / 255.0) * (1 - fo); b = b * fo + (10 / 255.0) * (1 - fo); }
                    if (!tile) cls += 0; else cls += 8; /* rock classes offset (label image only) */
                    break;
                }
            }
            acc[0] += r; acc[1] += gg; acc[2] += b; lcount[cls & 7]++;
        }
        unsigned char *p = img + ((size_t)j * ow + i) * 3;
        for (int k = 0; k < 3; k++) { double v = acc[k] / (SS * SS); p[k] = (unsigned char)(fmin(1, fmax(0, v)) * 255 + .5); }
        int best = 0; for (int k = 1; k < 8; k++) if (lcount[k] > lcount[best]) best = k; lab[(size_t)j * ow + i] = (unsigned char)best;
    }
    FILE *o = fopen(argv[15], "wb"); fprintf(o, "P6\n%d %d\n255\n", ow, oh); fwrite(img, 1, (size_t)ow * oh * 3, o); fclose(o);
    if (argc > 16) { FILE *l = fopen(argv[16], "wb"); fprintf(l, "P5\n%d %d\n255\n", ow, oh); fwrite(lab, 1, (size_t)ow * oh, l); fclose(l); }
    return 0;
}
