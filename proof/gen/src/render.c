/* Minimal voxel ray-caster imitating rd-132211's look: unit cubes, face shades 1.0/0.8/0.6,
 * 2D light-depth map (lit faces unfogged, unlit faces x0.8 and GL_EXP fog density 0.2 to colour 0x0E0B0A),
 * sky clear colour (0.5,0.8,1.0), gluPerspective(70) vertical FOV, camera = T(0,0,-0.3) Rx(pitch) Ry(yaw) T(-player).
 * Textures are procedural stand-ins (not terrain.png).
 * usage: render blocks W H D px py pz yaw pitch outW outH grassrule out.ppm
 *   grassrule: y42 (solid at y==D*2/3 is grass, rd-132211 Chunk rule) | ge42 | top (top solid of column) | type (value 2) | none */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int W, H, D; static uint8_t *B; static int *LD; static int GR;
static inline int solid(int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H) return 0;
    return B[((long)y * H + z) * W + x] != 0;
}
static inline int lit(int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H) return 1;
    return y >= LD[x + z * W];
}
static inline int isGrass(int x, int y, int z) {
    switch (GR) {
    case 0: return y == D * 2 / 3;
    case 1: return y >= D * 2 / 3;
    case 2: return y == LD[x + z * W];
    case 3: return B[((long)y * H + z) * W + x] == 2;
    default: return 0;
    }
}
static unsigned hsh(unsigned a) { a ^= a >> 16; a *= 0x7feb352d; a ^= a >> 15; a *= 0x846ca68b; a ^= a >> 16; return a; }

int main(int argc, char **argv) {
    if (argc < 14) { fprintf(stderr, "usage\n"); return 2; }
    W = atoi(argv[2]); H = atoi(argv[3]); D = atoi(argv[4]);
    double px = atof(argv[5]), py = atof(argv[6]), pz = atof(argv[7]), yaw = atof(argv[8]) * M_PI / 180, pitch = atof(argv[9]) * M_PI / 180;
    int ow = atoi(argv[10]), oh = atoi(argv[11]);
    const char *g = argv[12];
    GR = !strcmp(g, "y42") ? 0 : !strcmp(g, "ge42") ? 1 : !strcmp(g, "top") ? 2 : !strcmp(g, "type") ? 3 : 4;
    long N = (long)W * H * D; B = malloc(N);
    FILE *f = fopen(argv[1], "rb"); if (!f || fread(B, 1, N, f) != (size_t)N) { fprintf(stderr, "read fail\n"); return 1; } fclose(f);
    LD = malloc(sizeof(int) * W * H);
    for (int x = 0; x < W; x++) for (int z = 0; z < H; z++) { int y = D - 1; while (y > 0 && !solid(x, y, z)) y--; LD[x + z * W] = y; }
    /* inverse view rotation: world = Ry(-yaw) Rx(-pitch) view */
    double cp = cos(-pitch), sp = sin(-pitch), cy = cos(-yaw), sy = sin(-yaw);
    #define ROT(vx, vy, vz, ox, oy, oz) { double ax = vx, ay = cp*vy - sp*vz, az = sp*vy + cp*vz; ox = cy*ax + sy*az; oy = ay; oz = -sy*ax + cy*az; }
    double ex, ey, ez; ROT(0, 0, 0.3, ex, ey, ez); ex += px; ey += py; ez += pz;
    double t = tan(70.0 * M_PI / 360.0), asp = (double)ow / oh;
    unsigned char *img = malloc((size_t)ow * oh * 3);
    for (int j = 0; j < oh; j++) for (int i = 0; i < ow; i++) {
        double u = ((i + 0.5) / ow * 2 - 1) * t * asp, v = (1 - (j + 0.5) / oh * 2) * t;
        double dx, dy, dz; ROT(u, v, -1, dx, dy, dz);
        double len = sqrt(dx*dx + dy*dy + dz*dz); dx /= len; dy /= len; dz /= len;
        double r = 0.5, gg = 0.8, b = 1.0;
        int X = (int)floor(ex), Y = (int)floor(ey), Z = (int)floor(ez);
        int sx = dx > 0 ? 1 : -1, syy = dy > 0 ? 1 : -1, sz = dz > 0 ? 1 : -1;
        double tdx = fabs(1 / dx), tdy = fabs(1 / dy), tdz = fabs(1 / dz);
        double tx = (dx > 0 ? (X + 1 - ex) : (ex - X)) * tdx, ty = (dy > 0 ? (Y + 1 - ey) : (ey - Y)) * tdy, tz = (dz > 0 ? (Z + 1 - ez) : (ez - Z)) * tdz;
        double dist = 0; int face = -1;
        for (int step = 0; step < 1200; step++) {
            if (tx < ty && tx < tz) { X += sx; dist = tx; tx += tdx; face = 0; }
            else if (ty < tz) { Y += syy; dist = ty; ty += tdy; face = 1; }
            else { Z += sz; dist = tz; tz += tdz; face = 2; }
            /* leave world box entirely? */
            if ((X < -1 && sx < 0) || (X > W && sx > 0) || (Z < -1 && sz < 0) || (Z > H && sz > 0) || (Y < -1 && syy < 0) || (Y > D + 40 && syy > 0)) break;
            if (solid(X, Y, Z)) {
                /* adjacent air cell is the one we came from */
                int ax = X, ay = Y, az = Z; double c;
                if (face == 0) { ax -= sx; c = 0.6; } else if (face == 1) { ay -= syy; c = 1.0; } else { az -= sz; c = 0.8; }
                int L = lit(ax, ay, az);
                double hx = ex + dx * dist, hy = ey + dy * dist, hz = ez + dz * dist;
                double fu = face == 0 ? hz : hx, fv = face == 1 ? hz : hy;
                int tu = (int)floor((fu - floor(fu)) * 16), tv = (int)floor((fv - floor(fv)) * 16);
                unsigned hh = hsh((unsigned)(X * 73856093) ^ (unsigned)(Y * 19349663) ^ (unsigned)(Z * 83492791) ^ (unsigned)(tu * 16 + tv) * 2654435761u ^ face);
                double n = (hh & 255) / 255.0;
                double cr, cg, cb;
                if (isGrass(X, Y, Z)) { cr = 0.30 + 0.12 * n; cg = 0.62 + 0.18 * n; cb = 0.18 + 0.08 * n; }
                else { double k = 0.30 + 0.45 * n * n; cr = cg = cb = k; }
                double br = c * (L ? 1.0 : 0.8);
                cr *= br; cg *= br; cb *= br;
                if (!L) { double fo = exp(-0.2 * dist); cr = cr * fo + (14 / 255.0) * (1 - fo); cg = cg * fo + (11 / 255.0) * (1 - fo); cb = cb * fo + (10 / 255.0) * (1 - fo); }
                r = cr; gg = cg; b = cb; break;
            }
        }
        unsigned char *p = img + ((size_t)j * ow + i) * 3;
        p[0] = (unsigned char)(fmin(1, r) * 255); p[1] = (unsigned char)(fmin(1, gg) * 255); p[2] = (unsigned char)(fmin(1, b) * 255);
    }
    FILE *o = fopen(argv[13], "wb"); fprintf(o, "P6\n%d %d\n255\n", ow, oh); fwrite(img, 1, (size_t)ow * oh * 3, o); fclose(o);
    return 0;
}
