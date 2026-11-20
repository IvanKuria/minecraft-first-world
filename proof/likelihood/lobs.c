/* LIKELIHOOD lane: synthetic footage-like observation of a full-height world from the footage poses (no rendering).
 * Per pixel (640x360, FOV 70 => f = 180/tan35, pixel centres, 0.3 setback, zero roll; obs_footage.project conventions)
 * a voxel DDA finds the first solid cell.  A hit on a column-top block is GREEN (grass on all faces, TERRACE H-grass),
 * any other hit is OPEN-EVIDENCE (stone / dark).
 *   MAP  (obs_footage.py / obs_r1.py rules): every ray crossing the plain's top-face plane y = ytop+1 within 32 blocks
 *        votes its class for the crossing column; label if >= 24 pooled votes and a 60% majority; open components
 *        (4-conn) >= 4 kept; closed columns 8-adjacent to any open column dropped.  Output in c5search .obs format.
 *   SWEEP proxy (TERRACE sweep.py): for green pixels and planes k = -8..+3 (y = ytop+1+k) the crossing column gets n+1,
 *        and h+1 if the ray really hits that column's top face at height k.  Per column, per frame entries with n >= 20
 *        are pooled; r_k = RHO * h/n stands in for the texture correlation; label = best k if r >= 0.35 and it beats the
 *        runner-up by 0.12 (TERRACE's frozen thresholds); only columns with d < 10 from the player are kept.
 *   TRUTH: true column depth D = ytop - top for every observed column.
 *   BLUR rule (default on): a pixel counts as a texture match only if its 4 neighbour pixels hit the same face.
 * usage: lobs <world.blocks> <ytop> <spawns.txt> <out.txt> [RHO=0.85] [BLUR=1]
 *   spawns.txt lines: <id> <r1|r2> <px> <pz>     (px,pz = player position incl. phase; registration = floor(px),floor(pz))
 *   out.txt: per spawn  "SPAWN id which px pz" then O/C lines (post-processed map, c5search format), M dx dz lab (raw map
 *            labels), S dx dz k r (sweep labels), U dx dz (sweep-scored but unlabelled), T dx dz D (truth).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#define W 256
#define H 256
#define D 64
typedef struct { double yaw, pitch, h; } Pose;
static const Pose R1P[4] = {{0.8723395869759836, 0.7476627651893996, 8.902646463378982}, {0.8197652187813413, 0.7967635967042872, 8.452703654933146},
                            {0.4609524263042897, 0.8749450911851742, 7.19692649388464}, {-0.02853240416465555, 0.9835284426226765, 6.279740322656448}};
static const Pose R2P[4] = {{0.9629822982451468, 0.6294091009185605, 9.547240479548899}, {1.4196994674044625, 0.8099710792327597, 8.64972879972648},
                            {2.2520413984319965, 0.7357062633000372, 7.151904068461688}, {2.2515223125511685, 0.7344982514935497, 5.894047108821224}};
static uint8_t *B; static int TOP[W * H];
static inline int solid(int x, int y, int z) { if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H) return 0; return B[((long)y * H + z) * W + x] != 0; }
/* DDA; returns 1 on hit with cell and entry face (0..5 = -y,+y,-z,+z,-x,+x) */
static int cast(double ox, double oy, double oz, double vx, double vy, double vz, double maxd, int *cx, int *cy, int *cz, int *face) {
    double len = sqrt(vx * vx + vy * vy + vz * vz); vx /= len; vy /= len; vz /= len;
    double t0 = 0, t1 = maxd, lo[3] = {0, 0, 0}, hi[3] = {W, D, H}, o[3] = {ox, oy, oz}, v[3] = {vx, vy, vz};
    for (int a = 0; a < 3; a++) {
        if (fabs(v[a]) < 1e-12) { if (o[a] < lo[a] || o[a] > hi[a]) return 0; continue; }
        double ta = (lo[a] - o[a]) / v[a], tb = (hi[a] - o[a]) / v[a]; if (ta > tb) { double s = ta; ta = tb; tb = s; }
        if (ta > t0) t0 = ta; if (tb < t1) t1 = tb; if (t0 > t1) return 0;
    }
    double px = ox + vx * (t0 + 1e-9), py = oy + vy * (t0 + 1e-9), pz = oz + vz * (t0 + 1e-9);
    int x = (int)floor(px), y = (int)floor(py), z = (int)floor(pz);
    if (x >= W) x = W - 1; if (y >= D) y = D - 1; if (z >= H) z = H - 1; if (x < 0) x = 0; if (y < 0) y = 0; if (z < 0) z = 0;
    int sx = vx > 0 ? 1 : -1, sy = vy > 0 ? 1 : -1, sz = vz > 0 ? 1 : -1;
    double tdx = fabs(vx) < 1e-12 ? 1e30 : fabs(1 / vx), tdy = fabs(vy) < 1e-12 ? 1e30 : fabs(1 / vy), tdz = fabs(vz) < 1e-12 ? 1e30 : fabs(1 / vz);
    double tmx = fabs(vx) < 1e-12 ? 1e30 : ((sx > 0 ? (x + 1 - ox) : (ox - x)) * tdx);
    double tmy = fabs(vy) < 1e-12 ? 1e30 : ((sy > 0 ? (y + 1 - oy) : (oy - y)) * tdy);
    double tmz = fabs(vz) < 1e-12 ? 1e30 : ((sz > 0 ? (z + 1 - oz) : (oz - z)) * tdz);
    int f = 255;
    if (t0 > 0) {
        double ex = fabs(vx) < 1e-12 ? -1 : ((sx > 0 ? 0 : W) - ox) / vx, ey = fabs(vy) < 1e-12 ? -1 : ((sy > 0 ? 0 : D) - oy) / vy, ez = fabs(vz) < 1e-12 ? -1 : ((sz > 0 ? 0 : H) - oz) / vz;
        if (fabs(ey - t0) < 1e-7) f = sy > 0 ? 0 : 1; else if (fabs(ez - t0) < 1e-7) f = sz > 0 ? 2 : 3; else if (fabs(ex - t0) < 1e-7) f = sx > 0 ? 4 : 5;
    }
    double tc = t0;
    for (int it = 0; it < 4096; it++) {
        if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H || tc > t1 + 1e-9) return 0;
        if (solid(x, y, z)) { if (f == 255) return 0; *cx = x; *cy = y; *cz = z; *face = f; return 1; }
        if (tmx < tmy && tmx < tmz) { tc = tmx; x += sx; tmx += tdx; f = sx > 0 ? 4 : 5; }
        else if (tmy < tmz) { tc = tmy; y += sy; tmy += tdy; f = sy > 0 ? 0 : 1; }
        else { tc = tmz; z += sz; tmz += tdz; f = sz > 0 ? 2 : 3; }
    }
    return 0;
}
#define G 160            /* local grid (columns) around the spawn, offsets -80..79 */
#define OFF 80
#define NK 12            /* k = -8..+3 */
static int vg[G * G], vo[G * G];
static int sn[4][NK][G * G], sh[4][NK][G * G];
static int HID[640 * 360]; static int BLUR = 1;
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "see header\n"); return 2; }
    B = malloc((size_t)W * H * D); FILE *f = fopen(argv[1], "rb"); if (!f || fread(B, 1, (size_t)W * H * D, f) != (size_t)W * H * D) { fprintf(stderr, "world\n"); return 2; } fclose(f);
    int ytop = atoi(argv[2]); double RHO = argc > 5 ? atof(argv[5]) : 0.85; if (argc > 6) BLUR = atoi(argv[6]);
    for (int z = 0; z < H; z++) for (int x = 0; x < W; x++) { int t = -1; for (int y = D - 1; y >= 0; y--) if (solid(x, y, z)) { t = y; break; } TOP[z * W + x] = t; }
    FILE *sp = fopen(argv[3], "r"); FILE *out = fopen(argv[4], "w");
    const double Fc = 180.0 / tan(35.0 * M_PI / 180.0);
    char id[64], which[8]; double px, pz;
    while (fscanf(sp, "%63s %7s %lf %lf", id, which, &px, &pz) == 4) {
        const Pose *P = !strcmp(which, "r1") ? R1P : R2P;
        int fx = (int)floor(px), fz = (int)floor(pz);
        memset(vg, 0, sizeof vg); memset(vo, 0, sizeof vo); memset(sn, 0, sizeof sn); memset(sh, 0, sizeof sh);
        for (int fr = 0; fr < 4; fr++) {
            double s = sin(P[fr].yaw), c = cos(P[fr].yaw), spp = sin(P[fr].pitch), cp = cos(P[fr].pitch);
            double cx = px - .3 * s * cp, cz = pz + .3 * c * cp, cy = ytop + 1 + P[fr].h;
            for (int Y = 0; Y < 360; Y++) for (int X = 0; X < 640; X++) {
                double rx = (X + .5 - 320) / Fc, ry = (Y + .5 - 180) / Fc;
                double dx = c * rx - s * spp * ry + s * cp, dy = -cp * ry - spp, dz = s * rx + c * spp * ry - c * cp;
                HID[Y * 640 + X] = -1;
                if (dy >= -1e-6) continue;
                int hx, hy, hz, hf; if (!cast(cx, cy, cz, dx, dy, dz, 200.0, &hx, &hy, &hz, &hf)) continue;
                HID[Y * 640 + X] = (int)((((long)hy * H + hz) * W + hx) * 6 + hf) & 0x7fffffff;
            }
            for (int Y = 0; Y < 360; Y++) for (int X = 0; X < 640; X++) {
                double rx = (X + .5 - 320) / Fc, ry = (Y + .5 - 180) / Fc;
                double dx = c * rx - s * spp * ry + s * cp, dy = -cp * ry - spp, dz = s * rx + c * spp * ry - c * cp;
                if (dy >= -1e-6) continue;   /* looks at or above the horizon: sky in the footage (not used by either map) */
                int id0 = HID[Y * 640 + X]; if (id0 < 0) continue;
                int hf = id0 % 6, cell = id0 / 6, hx = cell % W, hz = (cell / W) % H, hy = cell / (W * H);
                int interior = 1;   /* BLUR rule: a texture match needs the 4-neighbour pixels on the same face */
                if (BLUR) { if (X == 0 || Y == 0 || X == 639 || Y == 359) interior = 0;
                    else if (HID[Y * 640 + X - 1] != id0 || HID[Y * 640 + X + 1] != id0 || HID[(Y - 1) * 640 + X] != id0 || HID[(Y + 1) * 640 + X] != id0) interior = 0; }
                int green = hy == TOP[hz * W + hx];
                /* map vote at the plain top-face plane */
                double t = ((ytop + 1) - cy) / dy, X3 = cx + t * dx, Z3 = cz + t * dz;
                if (hypot(X3 - cx, Z3 - cz) <= 32.0) {
                    int ix = (int)floor(X3) - fx + OFF, iz = (int)floor(Z3) - fz + OFF;
                    if (ix >= 0 && iz >= 0 && ix < G && iz < G) { if (green) vg[iz * G + ix]++; else vo[iz * G + ix]++; }
                }
                if (!green) continue;
                for (int k = -8; k <= 3; k++) {
                    double hk = cy - (ytop + 1 + k); if (hk <= 0.05) continue;
                    double tt = hk / (-dy), Xp = cx + tt * dx, Zp = cz + tt * dz;
                    if (hypot(Xp - cx, Zp - cz) > 32.0) continue;
                    int qx = (int)floor(Xp), qz = (int)floor(Zp);
                    int ix = qx - fx + OFF, iz = qz - fz + OFF; if (ix < 0 || iz < 0 || ix >= G || iz >= G) continue;
                    sn[fr][k + 8][iz * G + ix]++;
                    if (interior && hx == qx && hz == qz && hy == ytop + k && hf == 1) sh[fr][k + 8][iz * G + ix]++;
                }
            }
        }
        fprintf(out, "SPAWN %s %s %.6f %.6f\n", id, which, px, pz);
        /* map labels */
        static int8_t lab[G * G]; static int comp[G * G], csz[G * G], st[G * G];
        for (int i = 0; i < G * G; i++) { int n = vg[i] + vo[i]; lab[i] = -1; if (n >= 24) { if (vo[i] >= 0.6 * n) lab[i] = 1; else if (vg[i] >= 0.6 * n) lab[i] = 0; } comp[i] = -1; }
        int nc = 0;
        for (int i = 0; i < G * G; i++) {
            if (lab[i] != 1 || comp[i] >= 0) continue;
            int spn = 0; st[spn++] = i; comp[i] = nc; csz[nc] = 0;
            while (spn) { int j = st[--spn]; csz[nc]++; int x = j % G, z = j / G; const int ddx[4] = {1, -1, 0, 0}, ddz[4] = {0, 0, 1, -1};
                for (int d = 0; d < 4; d++) { int xx = x + ddx[d], zz = z + ddz[d]; if (xx < 0 || zz < 0 || xx >= G || zz >= G) continue; int k = zz * G + xx; if (lab[k] == 1 && comp[k] < 0) { comp[k] = nc; st[spn++] = k; } } }
            nc++;
        }
        static int remap[G * G]; int kept = 0; static double ccx[G * G], ccz[G * G]; static int cn[G * G];
        for (int c = 0; c < nc; c++) { remap[c] = csz[c] >= 4 ? kept++ : -1; }
        memset(ccx, 0, sizeof(double) * (kept + 1)); memset(ccz, 0, sizeof(double) * (kept + 1)); memset(cn, 0, sizeof(int) * (kept + 1));
        for (int i = 0; i < G * G; i++) {
            if (lab[i] < 0) continue; int x = i % G - OFF, z = i / G - OFF;
            if (lab[i] == 1) { int r = remap[comp[i]]; if (r < 0) continue; fprintf(out, "O %d %d 1 %d\n", x, z, r); ccx[r] += x + .5; ccz[r] += z + .5; cn[r]++; }
            else { int rim = 0; for (int dz = -1; dz <= 1 && !rim; dz++) for (int dx = -1; dx <= 1; dx++) { int xx = i % G + dx, zz = i / G + dz; if (xx < 0 || zz < 0 || xx >= G || zz >= G) continue; if (lab[zz * G + xx] == 1) { rim = 1; break; } }
                   if (!rim) fprintf(out, "O %d %d 0 -1\n", x, z); }
        }
        for (int r = 0; r < kept; r++) fprintf(out, "C %.3f %.3f %d\n", ccx[r] / cn[r], ccz[r] / cn[r], cn[r]);
        for (int i = 0; i < G * G; i++) if (lab[i] >= 0) fprintf(out, "M %d %d %d\n", i % G - OFF, i / G - OFF, lab[i]);   /* raw labels (all components, rims kept) */
        /* sweep labels (near field d < 10) */
        double phx = px - fx, phz = pz - fz;
        for (int i = 0; i < G * G; i++) {
            int x = i % G - OFF, z = i / G - OFF; if (hypot(x + .5 - phx, z + .5 - phz) >= 10.0) continue;
            double best = -1, second = -1; int bk = 0, any = 0;
            for (int k = 0; k < NK; k++) {
                long n = 0, hh = 0; for (int fr = 0; fr < 4; fr++) if (sn[fr][k][i] >= 20) { n += sn[fr][k][i]; hh += sh[fr][k][i]; }
                if (!n) continue; any = 1; double r = RHO * (double)hh / n;
                if (r > best) { second = best; best = r; bk = k; } else if (r > second) second = r;
            }
            if (!any) continue;
            if (best >= 0.35 && best - second >= 0.12) fprintf(out, "S %d %d %d %.3f\n", x, z, bk - 8, best);
            else fprintf(out, "U %d %d\n", x, z);   /* scored but unlabelled */
        }
        /* truth for every column of the local grid that has any map votes or sweep counts */
        for (int i = 0; i < G * G; i++) {
            int x = i % G - OFF, z = i / G - OFF, X = fx + x, Z = fz + z; if (X < 0 || Z < 0 || X >= W || Z >= H) continue;
            if (vg[i] + vo[i] == 0) continue;
            fprintf(out, "T %d %d %d\n", x, z, ytop - TOP[Z * W + X]);
        }
        fflush(out);
    }
    fclose(out); return 0;
}
