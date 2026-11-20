/* R1MAP: two-view synthetic power study + planted fixtures for the two-map C5 search.
 * Includes c5/c5sim.c (and through it GEN's gen.c verbatim), so worlds, seed lists and the observation pipeline are
 * identical to C5's.  Adds the FIRST-reset poses (r1map/obs_r1.py frames n1662, n1663, n1666, n1668; mode 0).
 *
 *   r1sim sim <variant> <nworlds> <nfalse> <sigma2> <sigma1> <minhole> <base_state> [npairs]
 *       npairs spawn pairs per world (one JSON line each; random-seed false lists only for the first pair).
 *       per world: grow from a random state; spawn A (plain) observed with the 4 second-reset poses, spawn B (plain,
 *       >= 60 blocks from A) observed with the 4 first-reset poses; each map is built separately (C5 build_obs:
 *       >= 3 votes, holes >= minhole, rims dropped) and shifted by its OWN rigid registration error ~ N(0, sigma).
 *       Scores the true seed list on both maps (centHit, IoU, ncomp) and <nfalse> random seed lists (per-map
 *       histograms IoU x cent, plus joint counts for loose rules as an independence check).  One JSON line per world.
 *   r1sim obs2 <variant> <gen_state> <x2> <z2> <x1> <z1> <out2.obs> <out1.obs> [minhole]
 *       planted fixture: both maps of one world, relative to floor of the respective spawn, with R lines.
 */
#define C5_NO_MAIN
#include "../c5/c5sim.c"

static const Pose R1[4] = {{0.8723395869759836, 0.7476627651893996, 8.902646463378982},
                           {0.8197652187813413, 0.7967635967042872, 8.452703654933146},
                           {0.4609524263042897, 0.8749450911851742, 7.19692649388464},
                           {-0.02853240416465555, 0.9835284426226765, 6.279740322656448}};

static void bbox(const ObsSet *O, int *a, int *b, int *c, int *d) {
    *a = 1 << 30; *b = -1; *c = 1 << 30; *d = -1;
    for (int i = 0; i < O->n; i++) { if (O->o[i].x < *a) *a = O->o[i].x; if (O->o[i].x > *b) *b = O->o[i].x; if (O->o[i].z < *c) *c = O->o[i].z; if (O->o[i].z > *d) *d = O->o[i].z; }
}

static void write_obs(const ObsSet *O, double px, double pz, int ref, const char *path) {
    int fx = (int)floor(px), fz = (int)floor(pz);
    FILE *f = fopen(path, "w");
    fprintf(f, "# synthetic (r1sim obs2), relative to floor of reset %d spawn; O dx dz lab comp | C cdx cdz size | R ref\nR %d\n", ref, ref);
    for (int i = 0; i < O->n; i++) fprintf(f, "O %d %d %d %d\n", O->o[i].x - fx, O->o[i].z - fz, O->o[i].lab, O->o[i].comp);
    for (int c = 0; c < O->ncomp; c++) fprintf(f, "C %.3f %.3f %d\n", O->cx[c] - fx, O->cz[c] - fz, O->csize[c]);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "see header\n"); return 2; }
    static Seed S[MAXK], F[MAXK];
    static int votes[WW * HH]; static int8_t lab[WW * HH]; static ObsSet OA, OB;
    if (!strcmp(argv[1], "obs2")) {
        Variant v; if (vload(&v, argv[2])) return 2;
        uint64_t st = strtoull(argv[3], 0, 0) & JR_MASK;
        double x2 = atof(argv[4]), z2 = atof(argv[5]), x1 = atof(argv[6]), z1 = atof(argv[7]);
        int minhole = argc > 10 ? atoi(argv[10]) : 4;
        uint8_t *a = malloc((size_t)WW * HH * DD); JRand r = {st, 0, 0}; Stats sts; generate(&v, &r, a, WW, HH, DD, &sts);
        memset(votes, 0, sizeof votes); memset(lab, 0, sizeof lab);
        observe_spawn(a, x2, z2, R2, 4, 32.0, votes, lab); build_obs(votes, lab, 3, minhole, 0, 0, &OA);
        memset(votes, 0, sizeof votes); memset(lab, 0, sizeof lab);
        observe_spawn(a, x1, z1, R1, 4, 32.0, votes, lab); build_obs(votes, lab, 3, minhole, 0, 0, &OB);
        write_obs(&OA, x2, z2, 2, argv[8]); write_obs(&OB, x1, z1, 1, argv[9]);
        printf("{\"nobs2\":%d,\"nopen2\":%d,\"ncomp2\":%d,\"nobs1\":%d,\"nopen1\":%d,\"ncomp1\":%d,\"fnv_state_out\":%llu}\n",
               OA.n, OA.nopen, OA.ncomp, OB.n, OB.nopen, OB.ncomp, (unsigned long long)r.s);
        free(a); return 0;
    }
    if (!strcmp(argv[1], "sim")) {
        Variant v; if (vload(&v, argv[2])) return 2;
        int nworlds = atoi(argv[3]); long nfalse = atol(argv[4]); double sig2 = atof(argv[5]), sig1 = atof(argv[6]);
        int minhole = atoi(argv[7]); uint64_t g = strtoull(argv[8], 0, 0); int npairs = argc > 9 ? atoi(argv[9]) : 1;
        uint8_t *a = malloc((size_t)WW * HH * DD);
        for (int w = 0; w < nworlds; w++) {
            uint64_t st = splitmix(&g) & JR_MASK;
            JRand r = {st, 0, 0}; Stats sts; generate(&v, &r, a, WW, HH, DD, &sts);
            int K = seeds_from_state(&v, st, S, NULL);
            for (int pr = 0; pr < npairs; pr++) {
            double px, pz, qx = 0, qz = 0; int ok = pick_spawn(a, &g, &px, &pz), okb = 0;
            for (int t = 0; t < 200 && ok; t++) { if (!pick_spawn(a, &g, &qx, &qz)) break; if (hypot(qx - px, qz - pz) >= 60) { okb = 1; break; } }
            if (!ok || !okb) continue;
            int s2x = (int)lround(sig2 * gauss(&g)), s2z = (int)lround(sig2 * gauss(&g));
            int s1x = (int)lround(sig1 * gauss(&g)), s1z = (int)lround(sig1 * gauss(&g));
            memset(votes, 0, sizeof votes); memset(lab, 0, sizeof lab);
            observe_spawn(a, px, pz, R2, 4, 32.0, votes, lab); build_obs(votes, lab, 3, minhole, s2x, s2z, &OA);
            memset(votes, 0, sizeof votes); memset(lab, 0, sizeof lab);
            observe_spawn(a, qx, qz, R1, 4, 32.0, votes, lab); build_obs(votes, lab, 3, minhole, s1x, s1z, &OB);
            if (OA.n == 0 || OB.n == 0) continue;
            int ax0, ax1, az0, az1, bx0, bx1, bz0, bz1; bbox(&OA, &ax0, &ax1, &az0, &az1); bbox(&OB, &bx0, &bx1, &bz0, &bz1);
            Stat ta, tb; stats(S, K, &OA, ax0, ax1, az0, az1, &ta); stats(S, K, &OB, bx0, bx1, bz0, bz1, &tb);
            long nf = pr == 0 ? nfalse : 0;
            /* false: per-map [iou bin 0..20][cent 0..32] + joint counts for loose rules (independence check) */
            static long HA[21][33], HB[21][33]; memset(HA, 0, sizeof HA); memset(HB, 0, sizeof HB);
            long jA = 0, jB = 0, jAB = 0;
            int nA = (OA.ncomp + 1) / 2, nB = (OB.ncomp + 1) / 2; /* loose: cent >= ceil(n/2), IoU >= .3 */
            for (long f = 0; f < nf; f++) {
                uint64_t fs = splitmix(&g) & JR_MASK; seeds_from_state(&v, fs, F, NULL);
                Stat fa, fb; stats(F, K, &OA, ax0, ax1, az0, az1, &fa); stats(F, K, &OB, bx0, bx1, bz0, bz1, &fb);
                int ia = (int)floor(fa.iou * 20), ib = (int)floor(fb.iou * 20); if (ia > 20) ia = 20; if (ib > 20) ib = 20; if (ia < 0) ia = 0; if (ib < 0) ib = 0;
                HA[ia][fa.centHit > 32 ? 32 : fa.centHit]++; HB[ib][fb.centHit > 32 ? 32 : fb.centHit]++;
                int pa = fa.centHit >= nA && fa.iou >= .3, pb = fb.centHit >= nB && fb.iou >= .3;
                jA += pa; jB += pb; jAB += pa && pb;
            }
            printf("{\"variant\":\"%s\",\"K\":%lld,\"world\":%d,\"pair\":%d,\"world_state\":%llu,\"sigma2\":%.2f,\"sigma1\":%.2f,\"shift2\":[%d,%d],\"shift1\":[%d,%d],\"minhole\":%d,"
                   "\"spawn2\":[%.3f,%.3f],\"spawn1\":[%.3f,%.3f],\"air\":%lld,"
                   "\"m2\":{\"nobs\":%d,\"nopen\":%d,\"ncomp\":%d,\"cent\":%d,\"iou\":%.4f},\"m1\":{\"nobs\":%d,\"nopen\":%d,\"ncomp\":%d,\"cent\":%d,\"iou\":%.4f},"
                   "\"nfalse\":%ld",
                   v.id, v.K, w, pr, (unsigned long long)st, sig2, sig1, s2x, s2z, s1x, s1z, minhole, px, pz, qx, qz, sts.air,
                   OA.n, OA.nopen, OA.ncomp, ta.centHit, ta.iou, OB.n, OB.nopen, OB.ncomp, tb.centHit, tb.iou, nf);
            if (nf) {
                printf(",\"loose_pass\":[%ld,%ld,%ld],\"false_m2\":[", jA, jB, jAB);
                for (int b = 0; b <= 20; b++) { printf("%s[", b ? "," : ""); for (int q = 0; q <= 32; q++) printf("%s%ld", q ? "," : "", HA[b][q]); printf("]"); }
                printf("],\"false_m1\":[");
                for (int b = 0; b <= 20; b++) { printf("%s[", b ? "," : ""); for (int q = 0; q <= 32; q++) printf("%s%ld", q ? "," : "", HB[b][q]); printf("]"); }
                printf("]");
            }
            printf("}\n"); fflush(stdout);
            }
        }
        free(a); return 0;
    }
    fprintf(stderr, "unknown mode\n"); return 2;
}
