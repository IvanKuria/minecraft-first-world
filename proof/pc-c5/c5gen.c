/* Extended seed-and-grow generator for the C5 regrowth (PC lane). HYPOTHESIS FAMILY ONLY.
 * It covers GEN's surviving family (init=unset, rng=int, pick xyz, rule mono, cap fixed 42, unsetfill 1) and adds the
 * lead's rd-160052-idiom seed-draw variants that GEN's CLI cannot express:
 *   m   : coordinate margin (GEN: 1 = edge-skipping nextInt(254)+1; V-full / V-full42: 0)
 *   yb  : y bound for picks: y = nextInt(yb) + m  (GEN: D - 2m = 62; V-full: 64; V-full42: 43)
 *   tf  : 1 = type drawn BEFORE x,y,z (type-first order), 0 = after (GEN)
 *   nt  : ntypes; air iff nextInt(nt) == 0
 *   nb  : scan (first set neighbour in Tile.render order y-,y+,z-,z+,x-,x+) | rand (nextInt(6) drawn every iteration)
 *   until=1: iterate until no pickable UNSET cell remains (else sweeps)
 *   edge: 1 = after growth force x=0/255, z=0/255, y=0 (y <= capy) solid ("edges forced solid")
 * Growth picks use the SAME (m, yb) bounds as the seeds. The draw order, cap test position and neighbour-draw position
 * are copied from GEN's generate(), so with m=1, yb=62, tf=0, edge=0 it reproduces `gen run U6s:...` bit-for-bit
 * (checked by `c5gen check`, see README).
 *
 * usage: c5gen run   <spec> <state48> <out.blocks|->   -> JSON {steps, state_out, iterations, unset_left, fnv64, solid}
 *        c5gen seeds <spec> <state48> <n>               -> first n seeds "x y z t kept"
 * spec: K=..,nt=..,m=..,yb=..,tf=..,nb=scan|rand,until=0|1,sweeps=..,edge=0|1
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../gen/src/jrand.h"

#define W 256
#define H 256
#define D 64
#define CAPY 42
#define UNSET 255
typedef struct { long long K; int nt, m, yb, tf, nbrand, until, edge; double sweeps; int capy, surf; } Spec;   /* capy: seeds kept iff y <= capy (42 = GEN/old; 63 = full-height, no discard); surf: C5 surface layer */

int c5g_parse(const char *s, Spec *p) {
    p->K = 1024; p->nt = 8; p->m = 1; p->yb = 62; p->tf = 0; p->nbrand = 0; p->until = 1; p->edge = 0; p->sweeps = 32; p->capy = CAPY; p->surf = CAPY;
    char buf[256]; strncpy(buf, s, 255); buf[255] = 0;
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
        char *e = strchr(t, '='); if (!e) return -1; *e = 0; const char *v = e + 1;
        if (!strcmp(t, "K")) p->K = atoll(v); else if (!strcmp(t, "nt")) p->nt = atoi(v); else if (!strcmp(t, "m")) p->m = atoi(v);
        else if (!strcmp(t, "yb")) p->yb = atoi(v); else if (!strcmp(t, "tf")) p->tf = atoi(v);
        else if (!strcmp(t, "nb")) p->nbrand = !strcmp(v, "rand"); else if (!strcmp(t, "until")) p->until = atoi(v);
        else if (!strcmp(t, "edge")) p->edge = atoi(v); else if (!strcmp(t, "capy")) p->capy = atoi(v); else if (!strcmp(t, "surf")) p->surf = atoi(v); else if (!strcmp(t, "sweeps")) p->sweeps = atof(v);
        else return -1;
    }
    return 0;
}
static inline void pick(JRand *r, const Spec *p, int *x, int *y, int *z) {
    *x = jr_nextInt(r, W - 2 * p->m) + p->m; *y = jr_nextInt(r, p->yb) + p->m; *z = jr_nextInt(r, H - 2 * p->m) + p->m;
}
/* one seed; returns kept */
int c5g_seed(JRand *r, const Spec *p, int *x, int *y, int *z, int *t) {
    if (p->tf) { *t = jr_nextInt(r, p->nt) == 0 ? 0 : 1; pick(r, p, x, y, z); }
    else { pick(r, p, x, y, z); *t = jr_nextInt(r, p->nt) == 0 ? 0 : 1; }
    return *y <= p->capy;
}
static const int O[6][3] = {{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}};

static void grow(const Spec *p, JRand *r, uint8_t *a, long long *iters, long long *left) {
    for (long long i = 0; i < (long long)W * H * D; i++) a[i] = (i / (W * H)) > CAPY ? 0 : UNSET;
    for (long long k = 0; k < p->K; k++) {
        int x, y, z, t;
        if (!c5g_seed(r, p, &x, &y, &z, &t)) continue;
        a[((long long)y * H + z) * W + x] = (uint8_t)t;
    }
    /* pickable unset count: cells reachable by pick() with y <= CAPY */
    long long unsetCount = 0;
    int ymax = p->m + p->yb - 1;
    if (p->until) for (int y = p->m; y <= ymax && y <= CAPY; y++) for (int z = p->m; z < H - p->m; z++) for (int x = p->m; x < W - p->m; x++)
        if (a[((long long)y * H + z) * W + x] == UNSET) unsetCount++;
    long long M = (long long)(p->sweeps * (double)W * H * D + 0.5), it = 0;
    for (; p->until ? (unsetCount > 0 && it < 40000000000LL) : it < M; it++) {
        int x, y, z; pick(r, p, &x, &y, &z);
        long long i = ((long long)y * H + z) * W + x;
        int dsel = p->nbrand ? jr_nextInt(r, 6) : -1;
        if (y > CAPY) continue;
        if (a[i] != UNSET) continue;
        if (dsel >= 0) {
            int xx = x + O[dsel][0], yy = y + O[dsel][1], zz = z + O[dsel][2];
            if (xx < 0 || yy < 0 || zz < 0 || xx >= W || yy >= D || zz >= H) continue;
            uint8_t s = a[((long long)yy * H + zz) * W + xx];
            if (s != UNSET && yy <= CAPY) { a[i] = s; unsetCount--; }
        } else {
            for (int d = 0; d < 6; d++) {
                int xx = x + O[d][0], yy = y + O[d][1], zz = z + O[d][2];
                if (xx < 0 || yy < 0 || zz < 0 || xx >= W || yy >= D || zz >= H) continue;
                uint8_t s = a[((long long)yy * H + zz) * W + xx];
                if (yy > CAPY) continue;
                if (s != UNSET) { a[i] = s; unsetCount--; break; }
            }
        }
    }
    *iters = it; *left = 0;
    for (long long i = 0; i < (long long)W * H * D; i++) if (a[i] == UNSET) { (*left)++; a[i] = 1; }
    if (p->edge) for (int y = 0; y <= CAPY; y++) for (int z = 0; z < H; z++) for (int x = 0; x < W; x++)
        if (x == 0 || x == W - 1 || z == 0 || z == H - 1 || y == 0) a[((long long)y * H + z) * W + x] = 1;
}

#ifndef C5GEN_NO_MAIN
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "see header\n"); return 2; }
    Spec p; if (c5g_parse(argv[2], &p)) { fprintf(stderr, "bad spec\n"); return 2; }
    uint64_t s0 = strtoull(argv[3], NULL, 0) & JR_MASK;
    JRand r = {s0, 0, 0};
    if (!strcmp(argv[1], "seeds")) {
        int n = argc > 4 ? atoi(argv[4]) : (int)p.K;
        for (int k = 0; k < n; k++) { int x, y, z, t, kept = c5g_seed(&r, &p, &x, &y, &z, &t); printf("%d %d %d %d %d\n", x, y, z, t, kept); }
        return 0;
    }
    if (!strcmp(argv[1], "run")) {
        uint8_t *a = malloc((size_t)W * H * D); long long it, left;
        grow(&p, &r, a, &it, &left);
        if (argc > 4 && strcmp(argv[4], "-")) { FILE *f = fopen(argv[4], "wb"); fwrite(a, 1, (size_t)W * H * D, f); fclose(f); }
        uint64_t h = 1469598103934665603ULL; long long solid = 0;
        for (long long i = 0; i < (long long)W * H * D; i++) { h ^= a[i]; h *= 1099511628211ULL; solid += a[i] != 0; }
        printf("{\"spec\":\"%s\",\"state_in\":%llu,\"state_out\":%llu,\"steps\":%llu,\"iterations_run\":%lld,\"unset_left\":%lld,\"solid\":%lld,\"fnv64\":\"%016llx\"}\n",
               argv[2], (unsigned long long)s0, (unsigned long long)r.s, (unsigned long long)r.steps, it, left, solid, (unsigned long long)h);
        free(a); return 0;
    }
    return 2;
}
#endif
