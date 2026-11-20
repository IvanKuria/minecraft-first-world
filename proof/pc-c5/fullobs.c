/* Footage-like synthetic observation of a FULL-HEIGHT world (.blocks file), for the planted-seed check of C5 under the
 * full-height shell model. Same ray/plane construction, pose lists, votes, minhole and rim rules as c5sim.c/r1sim.c
 * (observe_spawn/build_obs), with the grass-top plane at PLANE (64 = top face of y=63) instead of 43, and a label rule:
 *   lab=air   : column open iff cell (x, PLANE-1, z) is air          (C5's own surface-layer semantics)
 *   lab=depthD: column open iff its top solid is <= PLANE-1-D        (pit at least D deep; shallow grass pits closed)
 * usage: fullobs <world.blocks> <plane> <labrule> <x2> <z2> <x1> <z1> <out2.obs> <out1.obs> [minhole]
 * Output files in c5search format (R 2 / R 1), registered to floor(x2,z2) / floor(x1,z1). */
#define C5_NO_MAIN
#include "../c5/c5sim.c"

static const Pose PR1[4] = {{0.8723395869759836, 0.7476627651893996, 8.902646463378982}, {0.8197652187813413, 0.7967635967042872, 8.452703654933146},
                            {0.4609524263042897, 0.8749450911851742, 7.19692649388464}, {-0.02853240416465555, 0.9835284426226765, 6.279740322656448}};
static int PLANE = 64, DEPTHD = -1;

static int top_at(const uint8_t *a, int x, int z) { for (int y = DD - 1; y >= 0; y--) if (solid_at(a, x, y, z)) return y; return -1; }

static void observe_full(const uint8_t *a, double px, double pz, const Pose *P, int np, double dmax, int *votes, int8_t *lab) {
    const double F = 180.0 / tan(35.0 * M_PI / 180.0);
    for (int f = 0; f < np; f++) {
        double s = sin(P[f].yaw), c = cos(P[f].yaw), sp = sin(P[f].pitch), cp = cos(P[f].pitch);
        double cx = px - .3 * s * cp, cz = pz + .3 * c * cp, cy = PLANE + P[f].h;
        for (int Y = 0; Y < 360; Y += 2) for (int X = 0; X < 640; X += 2) {
            double rx = (X + .5 - 320) / F, ry = (Y + .5 - 180) / F;
            double dx = c * rx - s * sp * ry + s * cp, dy = -cp * ry - sp, dz = s * rx + c * sp * ry - c * cp;
            if (dy >= -1e-6) continue;
            double t = (PLANE - cy) / dy;
            double X3 = cx + t * dx, Z3 = cz + t * dz;
            if (hypot(X3 - cx, Z3 - cz) > dmax) continue;
            int ix = (int)floor(X3), iz = (int)floor(Z3);
            if (ix < 1 || iz < 1 || ix > WW - 2 || iz > HH - 2) continue;
            votes[iz * WW + ix]++;
            int open = DEPTHD < 0 ? !solid_at(a, ix, PLANE - 1, iz) : top_at(a, ix, iz) <= PLANE - 1 - DEPTHD;
            lab[iz * WW + ix] = open ? 1 : 0;
        }
    }
}

static void write_obs(const char *path, const ObsSet *O, int ref) {
    FILE *f = fopen(path, "w");
    fprintf(f, "# synthetic full-height (fullobs), relative to floor of reset %d spawn\nR %d\n", ref, ref);
    for (int i = 0; i < O->n; i++) fprintf(f, "O %d %d %d %d\n", O->o[i].x, O->o[i].z, O->o[i].lab, O->o[i].comp);
    for (int c = 0; c < O->ncomp; c++) fprintf(f, "C %.4f %.4f %d\n", O->cx[c], O->cz[c], O->csize[c]);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 10) { fprintf(stderr, "see header\n"); return 2; }
    static uint8_t a[WW * HH * DD];
    FILE *f = fopen(argv[1], "rb"); if (!f || fread(a, 1, sizeof a, f) != sizeof a) { fprintf(stderr, "bad world\n"); return 2; } fclose(f);
    PLANE = atoi(argv[2]);
    DEPTHD = !strcmp(argv[3], "air") ? -1 : atoi(argv[3] + 5);
    double x2 = atof(argv[4]), z2 = atof(argv[5]), x1 = atof(argv[6]), z1 = atof(argv[7]);
    int minhole = argc > 10 ? atoi(argv[10]) : 4;
    static int votes[WW * HH]; static int8_t lab[WW * HH]; static ObsSet OA, OB;
    memset(votes, 0, sizeof votes); memset(lab, 0, sizeof lab);
    observe_full(a, x2, z2, R2, 4, 32.0, votes, lab); build_obs(votes, lab, 3, minhole, -(int)floor(x2), -(int)floor(z2), &OA);
    memset(votes, 0, sizeof votes); memset(lab, 0, sizeof lab);
    observe_full(a, x1, z1, PR1, 4, 32.0, votes, lab); build_obs(votes, lab, 3, minhole, -(int)floor(x1), -(int)floor(z1), &OB);
    write_obs(argv[8], &OA, 2); write_obs(argv[9], &OB, 1);
    printf("{\"nobs2\":%d,\"nopen2\":%d,\"ncomp2\":%d,\"nobs1\":%d,\"nopen1\":%d,\"ncomp1\":%d}\n", OA.n, OA.nopen, OA.ncomp, OB.n, OB.nopen, OB.ncomp);
    return 0;
}
