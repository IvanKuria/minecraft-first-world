/* C5 lane: seed-placement-only early rejection, synthetic power study.
 *
 * Re-uses GEN's generator verbatim (#include of proof/gen/src/gen.c) so worlds and seed draws are bit-identical
 * to the validated `gen` binary.  HYPOTHESIS FAMILY ONLY (GEN's surviving "multi-type growth into UNSET" family).
 *
 * Modes
 *   c5sim validate <variant> <state> <seedlog>        compare seeds_from_state() with GEN_SEEDLOG of `gen run`
 *   c5sim sim <variant> <nworlds> <nfalse> <sigma> <minhole> <dmax> <base_state> [views]
 *        For each synthetic world: grow it from a random state, place cameras at random plain spawns with the
 *        footage's measured poses (second reset: 4 views, optionally opening: 3 views), extract the surface
 *        open/closed column map exactly as the footage would give it (ray/plane crossing at the y=43 grass top),
 *        degrade it (rigid registration shift ~ N(0,sigma) per spawn, small holes < minhole dropped, rims dropped),
 *        then score the TRUE seed list and <nfalse> random seed lists with the seeds-only statistics.
 *        One JSON line per world on stdout.
 * Statistics (seed list only, no growth):
 *   BA   = balanced accuracy of the nearest-kept-seed (Voronoi) prediction of each observed column
 *   A_m  = stage A: each observed open component has an air seed within r_A (horizontal) whose y >= 42-dyA
 */
#define main gen_main_unused
#include "../gen/src/gen.c"
#undef main
#include <math.h>

#define WW 256
#define HH 256
#define DD 64
#define MAXK 16384
#define MAXOBS 20000
#define MAXCOMP 512

typedef struct { int16_t x, y, z; uint8_t t, kept; } Seed;

/* exact replica of generate()'s seed phase (same PICK macro, same draw order) */
static int seeds_from_state(const Variant *v, uint64_t s0, Seed *S, uint64_t *state_after) {
    JRand rr = {s0, 0, 0}; JRand *r = &rr;
    const int W = WW, H = HH, D = DD; const long long N = (long long)W * H * D; (void)N;
    int capy = v->capy >= 0 ? v->capy : D * 2 / 3;
    for (long long k = 0; k < v->K; k++) {
        int x, y, z;
        PICK(x, y, z);
        uint8_t t = 1;
        if (v->seedtype == ST_AIR) t = 0;
        else if (v->seedtype == ST_RAND2) t = (uint8_t)draw(r, v, 2);
        else if (v->seedtype == ST_RAND3) t = (uint8_t)draw(r, v, 3);
        if (v->ntypes > 1) t = draw(r, v, v->ntypes) == 0 ? 0 : 1;
        S[k].x = x; S[k].y = y; S[k].z = z; S[k].t = t;
        S[k].kept = !(v->cap == CAP_FIXED && y > capy);
    }
    if (state_after) *state_after = r->s;
    return (int)v->K;
}

/* ---------------- footage poses (proof/link/scorer.py inputs) ---------------- */
typedef struct { double yaw, pitch, h; } Pose;
static const Pose R2[4] = {{0.9629822982451468, 0.6294091009185605, 9.547240479548899},
                           {1.4196994674044625, 0.8099710792327597, 8.64972879972648},
                           {2.2520413984319965, 0.7357062633000372, 7.151904068461688},
                           {2.2515223125511685, 0.7344982514935497, 5.894047108821224}};
static const Pose OP[3] = {{5.5156435916846895, -0.05957701054393033, 0.6021374681527347},
                           {5.52859959936068, 0.06648304883975528, 0.6399302251589015},
                           {5.535698189890221, 0.06618968452410817, 0.6398424094365687}};

typedef struct { int x, z; int8_t lab; int comp; } Obs;
typedef struct { int n; Obs o[MAXOBS]; int ncomp; double cx[MAXCOMP], cz[MAXCOMP]; int csize[MAXCOMP]; int nopen, nclosed; } ObsSet;

static inline int solid_at(const uint8_t *a, int x, int y, int z) { return a[((long)y * HH + z) * WW + x] != 0; }
static int top_of(const uint8_t *a, int x, int z) { for (int y = DD - 1; y >= 0; y--) if (solid_at(a, x, y, z)) return y; return -1; }

/* Footage-like observation of the y=42 surface layer from one spawn with a pose list.
 * votes: per column count of rays crossing y=43 there (pixel footprint), label from the world. */
static void observe_spawn(const uint8_t *a, double px, double pz, const Pose *P, int np, double dmax,
                          int *votes, int8_t *lab) {
    const double F = 180.0 / tan(35.0 * M_PI / 180.0);
    for (int f = 0; f < np; f++) {
        double s = sin(P[f].yaw), c = cos(P[f].yaw), sp = sin(P[f].pitch), cp = cos(P[f].pitch);
        double cx = px - .3 * s * cp, cz = pz + .3 * c * cp, cy = 43.0 + P[f].h;
        for (int Y = 0; Y < 360; Y += 2) for (int X = 0; X < 640; X += 2) {
            double rx = (X + .5 - 320) / F, ry = (Y + .5 - 180) / F;
            double dx = c * rx - s * sp * ry + s * cp, dy = -cp * ry - sp, dz = s * rx + c * sp * ry - c * cp;
            if (dy >= -1e-6) continue;
            double t = (43.0 - cy) / dy;
            double X3 = cx + t * dx, Z3 = cz + t * dz;
            if (hypot(X3 - cx, Z3 - cz) > dmax) continue;
            int ix = (int)floor(X3), iz = (int)floor(Z3);
            if (ix < 1 || iz < 1 || ix > WW - 2 || iz > HH - 2) continue;
            votes[iz * WW + ix]++;
            lab[iz * WW + ix] = solid_at(a, ix, 42, iz) ? 0 : 1;
        }
    }
}

/* turn votes/labels into an ObsSet: drop columns with < minvotes, drop open components smaller than minhole,
   drop closed columns adjacent (8-nbr) to any open column (rim), shift by (sx,sz). */
static void build_obs(const int *votes, const int8_t *lab, int minvotes, int minhole, int sx, int sz, ObsSet *O) {
    static int comp[WW * HH]; static int stackv[WW * HH];
    O->n = 0; O->ncomp = 0; O->nopen = O->nclosed = 0;
    for (int i = 0; i < WW * HH; i++) comp[i] = -1;
    int nc = 0; static int csz[WW * HH / 2 + 1]; static double ccx[WW * HH / 2 + 1], ccz[WW * HH / 2 + 1];
    for (int i = 0; i < WW * HH; i++) {
        if (votes[i] < minvotes || lab[i] != 1 || comp[i] >= 0) continue;
        int sp = 0; stackv[sp++] = i; comp[i] = nc; csz[nc] = 0; ccx[nc] = ccz[nc] = 0;
        while (sp) {
            int j = stackv[--sp]; int x = j % WW, z = j / WW; csz[nc]++; ccx[nc] += x + .5; ccz[nc] += z + .5;
            const int dxs[4] = {1, -1, 0, 0}, dzs[4] = {0, 0, 1, -1};
            for (int d = 0; d < 4; d++) {
                int xx = x + dxs[d], zz = z + dzs[d]; if (xx < 0 || zz < 0 || xx >= WW || zz >= HH) continue;
                int k = zz * WW + xx; if (votes[k] >= minvotes && lab[k] == 1 && comp[k] < 0) { comp[k] = nc; stackv[sp++] = k; }
            }
        }
        nc++;
    }
    int remap[MAXCOMP * 8]; int kept = 0;
    for (int c = 0; c < nc && c < MAXCOMP * 8; c++) {
        remap[c] = -1;
        if (csz[c] >= minhole && kept < MAXCOMP) { remap[c] = kept; O->cx[kept] = ccx[c] / csz[c] + sx; O->cz[kept] = ccz[c] / csz[c] + sz; O->csize[kept] = csz[c]; kept++; }
    }
    O->ncomp = kept;
    for (int i = 0; i < WW * HH && O->n < MAXOBS; i++) {
        if (votes[i] < minvotes) continue;
        int x = i % WW, z = i / WW;
        if (lab[i] == 1) {
            int c = comp[i]; if (c >= MAXCOMP * 8 || remap[c] < 0) continue;
            O->o[O->n++] = (Obs){x + sx, z + sz, 1, remap[c]}; O->nopen++;
        } else {
            int rim = 0;
            for (int dz = -1; dz <= 1 && !rim; dz++) for (int dx = -1; dx <= 1; dx++) {
                int xx = x + dx, zz = z + dz; if (xx < 0 || zz < 0 || xx >= WW || zz >= HH) continue;
                if (lab[zz * WW + xx] == 1 && votes[zz * WW + xx] > 0) { rim = 1; break; }
            }
            if (rim) continue;
            O->o[O->n++] = (Obs){x + sx, z + sz, 0, -1}; O->nclosed++;
        }
    }
}

/* ---------------- seeds-only statistics ---------------- */
typedef struct { double BA, tpr, tnr, iou; int centHit; int compHit, ncomp; int stageA; double cdist[MAXCOMP]; int cdy[MAXCOMP]; } Stat;
static double RA = 5.0; static int DYA = 12; /* stage A window (set by caller) */

static void stats(const Seed *S, int K, const ObsSet *O, int minx, int maxx, int minz, int maxz, Stat *st) {
    /* candidate seeds: kept, within bbox + 30 */
    static float sx[MAXK], sy[MAXK], sz[MAXK]; static uint8_t stt[MAXK]; int m = 0;
    for (int k = 0; k < K; k++) {
        if (!S[k].kept) continue;
        if (S[k].x < minx - 30 || S[k].x > maxx + 30 || S[k].z < minz - 30 || S[k].z > maxz + 30) continue;
        sx[m] = S[k].x + .5f; sy[m] = S[k].y + .5f; sz[m] = S[k].z + .5f; stt[m] = S[k].t; m++;
    }
    /* bucket index (counting sort by 8x8 column bucket); within a bucket, seed order = draw order */
    static int bstart[32 * 32 + 1], border[MAXK], bcnt[32 * 32];
    memset(bcnt, 0, sizeof bcnt);
    for (int j = 0; j < m; j++) bcnt[((int)sz[j] >> 3) * 32 + ((int)sx[j] >> 3)]++;
    bstart[0] = 0; for (int b = 0; b < 1024; b++) bstart[b + 1] = bstart[b] + bcnt[b];
    memset(bcnt, 0, sizeof bcnt);
    for (int j = 0; j < m; j++) { int b = ((int)sz[j] >> 3) * 32 + ((int)sx[j] >> 3); border[bstart[b] + bcnt[b]++] = j; }
    int tp = 0, tn = 0, fp = 0; int chit[MAXCOMP]; memset(chit, 0, sizeof chit);
    for (int i = 0; i < O->n; i++) {
        float px = O->o[i].x + .5f, py = 42.5f, pz = O->o[i].z + .5f; float best = 1e30f; int bt = 1, bj = 1 << 30;
        /* exact nearest kept seed: 8x8-column buckets, window +-3 buckets (>= 24 blocks), brute-force fallback */
        int bx0 = O->o[i].x >> 3, bz0 = O->o[i].z >> 3;
        for (int bz = bz0 - 3; bz <= bz0 + 3; bz++) for (int bx = bx0 - 3; bx <= bx0 + 3; bx++) {
            if (bx < 0 || bz < 0 || bx >= 32 || bz >= 32) continue;
            for (int q = bstart[bz * 32 + bx]; q < bstart[bz * 32 + bx + 1]; q++) { int j = border[q];
                float dx = sx[j] - px, dy = sy[j] - py, dz = sz[j] - pz; float d = dx * dx + dy * dy + dz * dz;
                if (d < best || (d == best && j < bj)) { best = d; bt = stt[j]; bj = j; } }
        }
        if (best >= 24.0f * 24.0f) { best = 1e30f; for (int j = 0; j < m; j++) {
            float dx = sx[j] - px, dy = sy[j] - py, dz = sz[j] - pz; float d = dx * dx + dy * dy + dz * dz;
            if (d < best) { best = d; bt = stt[j]; } } }
        int pred = bt == 0 ? 1 : 0;
        if (O->o[i].lab == 1) { if (pred) { tp++; chit[O->o[i].comp] = 1; } }
        else if (!pred) tn++; else fp++;
    }
    st->iou = (O->nopen + fp) ? (double)tp / (O->nopen + fp) : 1;
    st->tpr = O->nopen ? (double)tp / O->nopen : 1; st->tnr = O->nclosed ? (double)tn / O->nclosed : 1;
    st->BA = .5 * (st->tpr + st->tnr);
    st->ncomp = O->ncomp; st->compHit = 0; for (int c = 0; c < O->ncomp; c++) st->compHit += chit[c];
    for (int c = 0; c < O->ncomp; c++) { /* nearest air seed (3D, from centroid at y=42.5) */
        float best = 1e30f; int by = -1;
        for (int j = 0; j < m; j++) { if (stt[j]) continue; float dx = sx[j] - O->cx[c], dy = sy[j] - 42.5f, dz = sz[j] - O->cz[c]; float d = dx*dx+dy*dy+dz*dz; if (d < best) { best = d; by = (int)sy[j]; } }
        st->cdist[c] = sqrt(best); st->cdy[c] = by;
    }
    st->centHit = 0;
    for (int c = 0; c < O->ncomp; c++) { /* stage-1 statistic: nearest kept seed to the component centroid (y=42.5) is air */
        float best = 1e30f; int bt = 1;
        for (int j = 0; j < m; j++) { float dx = sx[j] - O->cx[c], dy = sy[j] - 42.5f, dz = sz[j] - O->cz[c]; float d = dx*dx+dy*dy+dz*dz; if (d < best) { best = d; bt = stt[j]; } }
        if (bt == 0) st->centHit++;
    }
    /* stage A */
    int okA = 1;
    for (int c = 0; c < O->ncomp && okA; c++) {
        int f = 0;
        for (int j = 0; j < m; j++) {
            if (stt[j] != 0 || sy[j] < 42.5f - DYA) continue;
            float dx = sx[j] - O->cx[c], dz = sz[j] - O->cz[c];
            if (dx * dx + dz * dz <= RA * RA) { f = 1; break; }
        }
        okA = f;
    }
    st->stageA = okA;
}

static uint64_t splitmix(uint64_t *x) { uint64_t z = (*x += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static double gauss(uint64_t *g) { double u = ((splitmix(g) >> 11) + 1.0) / 9007199254740993.0, w = (splitmix(g) >> 11) / 9007199254740992.0; return sqrt(-2 * log(u)) * cos(2 * M_PI * w); }

static int pick_spawn(const uint8_t *a, uint64_t *g, double *px, double *pz) {
    for (int tries = 0; tries < 100000; tries++) {
        double x = 30 + (splitmix(g) % 196000) / 1000.0, z = 30 + (splitmix(g) % 196000) / 1000.0;
        int mt = -1;
        for (int ix = (int)floor(x - .3); ix <= (int)floor(x + .3); ix++) for (int iz = (int)floor(z - .3); iz <= (int)floor(z + .3); iz++) { int t = top_of(a, ix, iz); if (t > mt) mt = t; }
        if (mt == 42) { *px = x; *pz = z; return 1; }
    }
    return 0;
}

#ifndef C5_NO_MAIN
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "see header\n"); return 2; }
    static Seed S[MAXK];
    if (!strcmp(argv[1], "validate")) {
        Variant v; if (vload(&v, argv[2])) return 2;
        uint64_t s0 = strtoull(argv[3], 0, 0) & JR_MASK; uint64_t after;
        int K = seeds_from_state(&v, s0, S, &after);
        FILE *f = fopen(argv[4], "r"); long long k; int x, y, z, t, bad = 0, n = 0;
        while (fscanf(f, "%lld %d %d %d %d", &k, &x, &y, &z, &t) == 5) { if (k >= K || S[k].x != x || S[k].y != y || S[k].z != z || S[k].t != t) bad++; n++; }
        fclose(f);
        printf("{\"variant\":\"%s\",\"state\":%llu,\"seeds\":%d,\"lines\":%d,\"mismatches\":%d,\"state_after_seed_phase\":%llu}\n", v.id, (unsigned long long)s0, K, n, bad, (unsigned long long)after);
        return bad != 0 || n != K;
    }
    if (!strcmp(argv[1], "sim")) {
        Variant v; if (vload(&v, argv[2])) return 2;
        int nworlds = atoi(argv[3]); long nfalse = atol(argv[4]); double sigma = atof(argv[5]);
        int minhole = atoi(argv[6]); double dmax = atof(argv[7]); uint64_t g = strtoull(argv[8], 0, 0);
        int views = argc > 9 ? atoi(argv[9]) : 2; /* 1 = second reset only, 2 = + opening */
        if (argc > 10) RA = atof(argv[10]);
        if (argc > 11) DYA = atoi(argv[11]);
        uint8_t *a = malloc((size_t)WW * HH * DD);
        static int votes[WW * HH]; static int8_t lab[WW * HH]; static ObsSet O; static Seed F[MAXK];
        for (int w = 0; w < nworlds; w++) {
            uint64_t st = splitmix(&g) & JR_MASK;
            JRand r = {st, 0, 0}; Stats sts; generate(&v, &r, a, WW, HH, DD, &sts);
            int K = seeds_from_state(&v, st, S, NULL);
            memset(votes, 0, sizeof votes); memset(lab, 0, sizeof lab);
            double px, pz, qx, qz; int ok = pick_spawn(a, &g, &px, &pz);
            observe_spawn(a, px, pz, R2, 4, dmax, votes, lab);
            int sx = (int)lround(sigma * gauss(&g)), sz = (int)lround(sigma * gauss(&g)); /* one rigid shift (both spawns share it here) */
            if (views >= 2 && pick_spawn(a, &g, &qx, &qz)) observe_spawn(a, qx, qz, OP, 3, dmax, votes, lab);
            build_obs(votes, lab, 3, minhole, sx, sz, &O);
            if (!ok || O.n == 0) { w--; continue; }
            int minx = 1 << 30, maxx = -1, minz = 1 << 30, maxz = -1;
            for (int i = 0; i < O.n; i++) { if (O.o[i].x < minx) minx = O.o[i].x; if (O.o[i].x > maxx) maxx = O.o[i].x; if (O.o[i].z < minz) minz = O.o[i].z; if (O.o[i].z > maxz) maxz = O.o[i].z; }
            Stat ts; stats(S, K, &O, minx, maxx, minz, maxz, &ts);
            /* false seeds */
            static long hist[201]; memset(hist, 0, sizeof hist); static long histI[201]; memset(histI, 0, sizeof histI); static long hist2[201][21]; memset(hist2, 0, sizeof hist2); static long histC[201][33]; memset(histC, 0, sizeof histC); long passA = 0; static long histA[201]; memset(histA, 0, sizeof histA);
            long cmh[MAXCOMP + 1]; memset(cmh, 0, sizeof cmh);
            for (long f = 0; f < nfalse; f++) {
                uint64_t fs = splitmix(&g) & JR_MASK;
                seeds_from_state(&v, fs, F, NULL);
                Stat fsx; stats(F, K, &O, minx, maxx, minz, maxz, &fsx);
                int b = (int)floor(fsx.BA * 200); if (b > 200) b = 200; if (b < 0) b = 0;
                int bi = (int)floor(fsx.iou * 200); if (bi > 200) bi = 200; histI[bi]++;
                histC[bi][fsx.centHit > 32 ? 32 : fsx.centHit]++;
                hist2[bi][(int)floor(fsx.BA * 20) > 20 ? 20 : (int)floor(fsx.BA * 20)]++;
                hist[b]++; if (fsx.stageA) { passA++; histA[b]++; }
                cmh[fsx.compHit > MAXCOMP ? MAXCOMP : fsx.compHit]++;
            }
            printf("{\"variant\":\"%s\",\"K\":%lld,\"world_state\":%llu,\"sigma\":%.2f,\"shift\":[%d,%d],\"minhole\":%d,\"dmax\":%.1f,\"views\":%d,\"RA\":%.1f,\"DYA\":%d,"
                   "\"air\":%lld,\"nobs\":%d,\"nopen\":%d,\"nclosed\":%d,\"ncomp\":%d,\"true\":{\"BA\":%.4f,\"tpr\":%.4f,\"tnr\":%.4f,\"iou\":%.4f,\"compHit\":%d,\"stageA\":%d},"
                   "\"nfalse\":%ld,\"false_passA\":%ld,\"false_hist\":[",
                   v.id, v.K, (unsigned long long)st, sigma, sx, sz, minhole, dmax, views, RA, DYA, sts.air, O.n, O.nopen, O.nclosed, O.ncomp,
                   ts.BA, ts.tpr, ts.tnr, ts.iou, ts.compHit, ts.stageA, nfalse, passA);
            for (int b = 0; b <= 200; b++) printf("%s%ld", b ? "," : "", hist[b]);
            printf("],\"false_histIoU\":[");
            for (int b = 0; b <= 200; b++) printf("%s%ld", b ? "," : "", histI[b]);
            printf("],\"false_IoU_BA\":[");
            for (int b = 0; b <= 200; b++) { printf("%s[", b ? "," : ""); for (int q = 0; q <= 20; q++) printf("%s%ld", q ? "," : "", hist2[b][q]); printf("]"); }
            printf("],\"false_IoU_cent\":[");
            for (int b = 0; b <= 200; b++) { printf("%s[", b ? "," : ""); for (int q = 0; q <= 32; q++) printf("%s%ld", q ? "," : "", histC[b][q]); printf("]"); }
            printf("],\"true_centHit\":%d", ts.centHit);
            printf(",\"true_comp\":[");
            for (int c = 0; c < O.ncomp; c++) printf("%s[%d,%.2f,%d]", c ? "," : "", O.csize[c], ts.cdist[c], ts.cdy[c]);
            printf("],\"false_histA\":[");
            for (int b = 0; b <= 200; b++) printf("%s%ld", b ? "," : "", histA[b]);
            printf("],\"false_compHit\":[");
            for (int b = 0; b <= O.ncomp && b <= MAXCOMP; b++) printf("%s%ld", b ? "," : "", cmh[b]);
            printf("]}\n"); fflush(stdout);
        }
        free(a); return 0;
    }
    if (!strcmp(argv[1], "obs")) { /* obs <variant> <gen_state> <x2> <z2> <out.obs> [minhole]: synthetic footage-like observation */
        Variant v; if (vload(&v, argv[2])) return 2;
        uint64_t st = strtoull(argv[3], 0, 0) & JR_MASK; double px = atof(argv[4]), pz = atof(argv[5]);
        int minhole = argc > 7 ? atoi(argv[7]) : 4;
        uint8_t *a = malloc((size_t)WW * HH * DD); JRand r = {st, 0, 0}; Stats sts; generate(&v, &r, a, WW, HH, DD, &sts);
        static int votes[WW * HH]; static int8_t lab[WW * HH]; static ObsSet O;
        observe_spawn(a, px, pz, R2, 4, 32.0, votes, lab);
        build_obs(votes, lab, 3, minhole, 0, 0, &O);
        int fx = (int)floor(px), fz = (int)floor(pz);
        FILE *f = fopen(argv[6], "w");
        fprintf(f, "# relative to floor(x2),floor(z2) of the second-reset spawn; lines: O dx dz lab comp | C cdx cdz size\n");
        for (int i = 0; i < O.n; i++) fprintf(f, "O %d %d %d %d\n", O.o[i].x - fx, O.o[i].z - fz, O.o[i].lab, O.o[i].comp);
        for (int c = 0; c < O.ncomp; c++) fprintf(f, "C %.3f %.3f %d\n", O.cx[c] - fx, O.cz[c] - fz, O.csize[c]);
        fclose(f);
        printf("{\"nobs\":%d,\"nopen\":%d,\"ncomp\":%d,\"fnv_state_out\":%llu}\n", O.n, O.nopen, O.ncomp, (unsigned long long)r.s);
        free(a); return 0;
    }
    fprintf(stderr, "unknown mode\n"); return 2;
}
#endif
