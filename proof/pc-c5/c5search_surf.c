/* C5 CPU reference search: enumerate generator seeds from the fingerprint lane's clock window, test each by its
 * SEED BLOCKS ONLY (first 4K draws), keep the few that pass for full regrowth + LINK's frozen scorer.
 *
 * Hypothesis H3 (LINK) / U*-int variants (GEN): the generator owned `new Random()` created before Math.random's.
 *   JDK6:  new Random() -> seed64 = (U + k) + System.nanoTime(),  U = 8682522807148012, k = k-th Random in the JVM
 *          internal state = (seed64 ^ 0x5DEECE66D) & (2^48-1)
 *   HotSpot6 Windows nanoTime = (jlong)(((double)qpc / (double)f) * 1e9)
 * Each candidate row (Math.random's state before the opening x draw, number of earlier Math.random calls c, k_m, and
 * the QPC count q_m that the fingerprint grid test recovered) fixes Math's Random creation time.  The generator's
 * Random was created Delta = [dmin,dmax] seconds earlier (generation ~20 s + startup) with k_g = k_m - j, j = 1..J.
 *   q_g in [q_m - dmax*f, q_m - dmin*f],  G0 = ((U + k_g + hs_nanos(q_g)) ^ MULT) & MASK
 * Early test per G0 (seed list only, see NOTES.txt for its measured power), applied to EVERY map (R1MAP, Oct 15):
 *   stage 1: count observed open components whose centroid's nearest kept seed (3D, y = 42.5) is AIR  >= cmin[m]
 *   stage 2: Voronoi IoU of predicted-open vs observed-open columns                                >= imin[m]
 * A seed passes iff it passes both stages on all maps.
 *
 * MAPS.  <obs> is one obs file or a comma-separated list.  Each file is registered to one reset of the candidate:
 *   a line "R 2" (default when absent: second reset, x2 z2) or "R 1" (first observed reset, x1 z1).
 * usage: c5search <variant> <cands.txt> <obs[,obs2..]> <f_hz> <dmin_s> <dmax_s> <J> <cmin[,..]> <imin[,..]> <threads> [q_lo q_hi]
 *   cands.txt lines: s_open calls k_m q_m x2 z2 [x1 z1]   (Java-float reset positions of that candidate; x1 z1 are
 *                    required iff some map has R 1)
 *   obs lines:       O dx dz lab comp  |  C cdx cdz size  |  R 1|2     (relative to floor(x_R), floor(z_R))
 *   cmin / imin:     one value for all maps, or one per map in map order
 *   optional q_lo q_hi: override the q_g window (validation with a planted seed)
 * Output: "PASS cand j q_g G0 hits0 iou0 [hits1 iou1 ...]" per pass (map order), then a SUMMARY line.
 * Env C5_ANY=1 (null studies only): log a seed when it passes on ANY map; all maps' (hits, iou) are printed, with
 *   iou = -1 for a map whose stage 1 failed.  Per-map s1/s2 counters in SUMMARY are marginal only under C5_ANY
 *   (otherwise maps are tested in order with early exit).  C5_RANDNULL=1: replace G0 by an unstructured random value (control).
 */
#define C5_NO_MAIN
#include "c5sim.c"
#include <pthread.h>

#define U0 8682522807148012LL
#define MAXMAP 8
static inline int64_t hs_nanos(int64_t q, int64_t f) { return (int64_t)(((double)q / (double)f) * 1000000000.0); }

typedef struct { uint64_t s_open; int calls, km; int64_t qm; double x2, z2, x1, z1; int has1; } Cand;
typedef struct { int ref; int NO; int *odx, *odz, *olab, *ocomp; int NCOMP; double *ccx, *ccz; int cmin; double imin; } Map;
static Variant V; static Cand *CA; static int NC;
static Map MP[MAXMAP]; static int NM;
static int64_t FHZ; static double DMIN, DMAX; static int J; static int64_t QLO = -1, QHI = -1;
static int RANDNULL, ANY;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long TESTED, PASS1, PASS2, PASS1M[MAXMAP], PASS2M[MAXMAP];

typedef struct { int tid, nth; } Arg;
typedef struct { float *sx, *sy, *sz; uint8_t *stt; } Roi;

static int roi(const Seed *S, int K, int fx, int fz, Roi *R) {
    int m = 0;
    for (int k = 0; k < K; k++) {
        if (!S[k].kept) continue;
        int rx = S[k].x - fx, rz = S[k].z - fz;
        if (rx < -62 || rx > 62 || rz < -62 || rz > 62) continue;  /* observations lie within 32 of the spawn; +30 margin */
        R->sx[m] = rx + .5f; R->sy[m] = S[k].y + .5f; R->sz[m] = rz + .5f; R->stt[m] = S[k].t; m++;
    }
    return m;
}

static int cent_hits(const Map *M, const Roi *R, int m) {
    int hits = 0;
    for (int cc = 0; cc < M->NCOMP; cc++) {
        float best = 1e30f; int bt = 1;
        for (int i = 0; i < m; i++) { float dx = R->sx[i] - (float)M->ccx[cc], dy = R->sy[i] - C5_SURFF, dz = R->sz[i] - (float)M->ccz[cc]; float d = dx*dx+dy*dy+dz*dz; if (d < best) { best = d; bt = R->stt[i]; } }
        if (bt == 0) hits++;
    }
    return hits;
}

static double map_iou(const Map *M, const Roi *R, int m) {
    int tp = 0, fp = 0, nopen = 0;
    for (int o = 0; o < M->NO; o++) {
        float px = M->odx[o] + .5f, pz = M->odz[o] + .5f; float best = 1e30f; int bt = 1;
        for (int i = 0; i < m; i++) { float dx = R->sx[i] - px, dy = R->sy[i] - C5_SURFF, dz = R->sz[i] - pz; float d = dx*dx+dy*dy+dz*dz; if (d < best) { best = d; bt = R->stt[i]; } }
        int pred = bt == 0;
        if (M->olab[o]) { nopen++; if (pred) tp++; } else if (pred) fp++;
    }
    return (nopen + fp) ? (double)tp / (nopen + fp) : 1.0;
}

static void test_seed(uint64_t G0, int ci, int j, int64_t q, Seed *S, Roi *R, unsigned long long *t1, unsigned long long *t2,
                      unsigned long long *t1m, unsigned long long *t2m) {
    const Cand *c = &CA[ci];
    int K = seeds_from_state(&V, G0, S, NULL);
    int hits[MAXMAP], mm[MAXMAP], ok1 = 1, any1 = 0; double iou[MAXMAP];
    for (int k = 0; k < NM; k++) {
        const Map *M = &MP[k];
        int fx = (int)floor(M->ref == 1 ? c->x1 : c->x2), fz = (int)floor(M->ref == 1 ? c->z1 : c->z2);
        mm[k] = roi(S, K, fx, fz, &R[k]);
        hits[k] = cent_hits(M, &R[k], mm[k]); iou[k] = -1;
        if (hits[k] >= M->cmin) { any1 = 1; t1m[k]++; } else { ok1 = 0; if (!ANY) return; }
    }
    if (!any1) return;
    if (ok1) (*t1)++;
    int ok2 = ok1, any2 = 0;
    for (int k = 0; k < NM; k++) {
        if (hits[k] < MP[k].cmin) continue;
        iou[k] = map_iou(&MP[k], &R[k], mm[k]);
        if (iou[k] >= MP[k].imin) { any2 = 1; t2m[k]++; } else { ok2 = 0; if (!ANY) return; }
    }
    if (ok2) (*t2)++;
    if (!(ANY ? any2 : ok2)) return;
    char buf[512]; int n = snprintf(buf, sizeof buf, "PASS %d %d %lld %llu", ci, j, (long long)q, (unsigned long long)G0);
    for (int k = 0; k < NM; k++) n += snprintf(buf + n, sizeof buf - n, " %d %.4f", hits[k], iou[k]);
    pthread_mutex_lock(&mu); puts(buf); fflush(stdout); pthread_mutex_unlock(&mu);
}

static void *worker(void *p) {
    Arg *a = p;
    Seed *S = malloc(sizeof(Seed) * MAXK); Roi R[MAXMAP];
    for (int k = 0; k < NM; k++) { R[k].sx = malloc(4 * MAXK); R[k].sy = malloc(4 * MAXK); R[k].sz = malloc(4 * MAXK); R[k].stt = malloc(MAXK); }
    unsigned long long n = 0, t1 = 0, t2 = 0, t1m[MAXMAP] = {0}, t2m[MAXMAP] = {0};
    for (int ci = 0; ci < NC; ci++) for (int j = 1; j <= J; j++) {
        int64_t lo = QLO >= 0 ? QLO : CA[ci].qm - (int64_t)(DMAX * FHZ), hi = QHI >= 0 ? QHI : CA[ci].qm - (int64_t)(DMIN * FHZ);
        long long kg = CA[ci].km - j; if (kg < 1) continue;
        for (int64_t q = lo + a->tid; q <= hi; q += a->nth) {
            int64_t t = hs_nanos(q, FHZ);
            uint64_t G0 = ((uint64_t)(U0 + kg + t) ^ JR_MULT) & JR_MASK;
            if (RANDNULL) { uint64_t g = (uint64_t)q * 0x9E3779B97F4A7C15ULL; G0 = splitmix(&g) & JR_MASK; } /* control: unstructured random G0 */
            test_seed(G0, ci, j, q, S, R, &t1, &t2, t1m, t2m); n++;
        }
    }
    pthread_mutex_lock(&mu); TESTED += n; PASS1 += t1; PASS2 += t2;
    for (int k = 0; k < NM; k++) { PASS1M[k] += t1m[k]; PASS2M[k] += t2m[k]; }
    pthread_mutex_unlock(&mu);
    free(S); for (int k = 0; k < NM; k++) { free(R[k].sx); free(R[k].sy); free(R[k].sz); free(R[k].stt); }
    return NULL;
}

static int load_map(Map *M, const char *path) {
    FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    char line[256]; int capo = 1 << 16;
    M->ref = 2; M->NO = M->NCOMP = 0;
    M->odx = malloc(4 * capo); M->odz = malloc(4 * capo); M->olab = malloc(4 * capo); M->ocomp = malloc(4 * capo);
    M->ccx = malloc(8 * 4096); M->ccz = malloc(8 * 4096);
    while (fgets(line, sizeof line, f)) {
        int a1, a2, a3, a4; double c1, c2; int sz;
        if (line[0] == 'O' && sscanf(line + 1, "%d %d %d %d", &a1, &a2, &a3, &a4) == 4) { if (M->NO == capo) { fprintf(stderr, "too many obs\n"); return 1; } M->odx[M->NO] = a1; M->odz[M->NO] = a2; M->olab[M->NO] = a3; M->ocomp[M->NO] = a4; M->NO++; }
        else if (line[0] == 'C' && sscanf(line + 1, "%lf %lf %d", &c1, &c2, &sz) == 3) { M->ccx[M->NCOMP] = c1; M->ccz[M->NCOMP] = c2; M->NCOMP++; }
        else if (line[0] == 'R' && sscanf(line + 1, "%d", &a1) == 1) { if (a1 != 1 && a1 != 2) { fprintf(stderr, "bad R line in %s\n", path); return 1; } M->ref = a1; }
    }
    fclose(f);
    return 0;
}

static int split_list(const char *s, char out[][1024], int max) {
    int n = 0; const char *p = s;
    while (n < max) { const char *e = strchr(p, ','); size_t L = e ? (size_t)(e - p) : strlen(p); if (L >= 1024) return -1; memcpy(out[n], p, L); out[n][L] = 0; n++; if (!e) break; p = e + 1; }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 11) { fprintf(stderr, "see header\n"); return 2; }
    if (vload(&V, argv[1])) return 2;
    if (V.rng != RNG_INT) { fprintf(stderr, "C5 search is for own-Random (rng=int) variants\n"); return 2; }
    static char parts[MAXMAP][1024];
    NM = split_list(argv[3], parts, MAXMAP); if (NM < 1) return 2;
    int need1 = 0;
    for (int k = 0; k < NM; k++) { if (load_map(&MP[k], parts[k])) return 2; need1 |= MP[k].ref == 1; }
    char cl[MAXMAP][1024]; int nc = split_list(argv[8], cl, MAXMAP); char il[MAXMAP][1024]; int ni = split_list(argv[9], il, MAXMAP);
    if ((nc != 1 && nc != NM) || (ni != 1 && ni != NM)) { fprintf(stderr, "cmin/imin: give 1 value or one per map\n"); return 2; }
    for (int k = 0; k < NM; k++) { MP[k].cmin = atoi(cl[nc == 1 ? 0 : k]); MP[k].imin = atof(il[ni == 1 ? 0 : k]); }
    FILE *f = fopen(argv[2], "r"); if (!f) { fprintf(stderr, "cannot open cands\n"); return 2; }
    int cap = 1024; CA = malloc(sizeof(Cand) * cap); NC = 0; char line[512];
    while (fgets(line, sizeof line, f)) {
        unsigned long long so; int calls, km; long long qm; double x2, z2, x1 = 0, z1 = 0;
        int got = sscanf(line, "%llu %d %d %lld %lf %lf %lf %lf", &so, &calls, &km, &qm, &x2, &z2, &x1, &z1);
        if (got < 6) continue;
        if (need1 && got < 8) { fprintf(stderr, "a map is registered to R 1 but a candidate line lacks x1 z1\n"); return 2; }
        if (NC == cap) { cap *= 2; CA = realloc(CA, sizeof(Cand) * cap); }
        CA[NC++] = (Cand){so, calls, km, qm, x2, z2, x1, z1, got >= 8};
    }
    fclose(f);
    FHZ = atoll(argv[4]); DMIN = atof(argv[5]); DMAX = atof(argv[6]); J = atoi(argv[7]);
    int nth = atoi(argv[10]); RANDNULL = getenv("C5_RANDNULL") != NULL; ANY = getenv("C5_ANY") != NULL;
    if (argc > 12) { QLO = atoll(argv[11]); QHI = atoll(argv[12]); }
    pthread_t th[256]; Arg ar[256];
    for (int i = 0; i < nth; i++) { ar[i] = (Arg){i, nth}; pthread_create(&th[i], NULL, worker, &ar[i]); }
    for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
    int totobs = 0, totc = 0; for (int k = 0; k < NM; k++) { totobs += MP[k].NO; totc += MP[k].NCOMP; }
    printf("SUMMARY variant=%s cands=%d obs=%d comps=%d tested=%llu pass_stage1=%llu pass_stage2=%llu maps=%d", V.id, NC, totobs, totc, TESTED, PASS1, PASS2, NM);
    for (int k = 0; k < NM; k++) printf(" map%d:R%d,obs=%d,comps=%d,cmin=%d,imin=%.3f,s1=%llu,s2=%llu", k, MP[k].ref, MP[k].NO, MP[k].NCOMP, MP[k].cmin, MP[k].imin, PASS1M[k], PASS2M[k]);
    printf("\n");
    return 0;
}
