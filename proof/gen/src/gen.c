/* GEN lane: parameterised family of "growth" generators for Notch's 13 May 2009 description.
 * HYPOTHESIS FAMILY ONLY - not recovered Minecraft source.
 *
 * Usage:
 *   gen run    <variant> <state48> <out.blocks|-> [W H D]   -> JSON on stdout (steps consumed, stats)
 *   gen steps  <variant> [W H D]                           -> closed-form LCG step count (if fixed)
 *   gen rewind <state48> <n>      / gen advance <state48> <n>
 *   gen list                                                 -> registered variant ids
 * <variant> is either a registered id (see registry below) or a spec string "k=v,k=v,...".
 * <state48> is the INTERNAL java.util.Random state (already scrambled), decimal or 0x-hex.
 * Block array layout is Notch's: index = (y*H + z)*W + x  (W=width x, H="height" = z extent, D=depth = y extent).
 * Values: 0 air, 1 rock (solid), 2 grass-type (only for typed variants).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "jrand.h"

enum { INIT_EMPTY = 0, INIT_UNSET = 1, INIT_SLAB = 2 };
enum { ST_ROCK = 0, ST_AIR = 1, ST_RAND2 = 2, ST_RAND3 = 3 };
enum { RNG_INT = 0, RNG_MATH = 1 };
enum { PICK_XYZ = 0, PICK_IDX = 1, PICK_XZY = 2 };
enum { NB_RAND = 0, NB_SCAN = 1 };
enum { RULE_MONO = 0, RULE_VOTER = 1 };
enum { CAP_NONE = 0, CAP_POST = 1, CAP_FIXED = 2 };
#define UNSET 255

typedef struct {
    char id[64];
    int init, seedtype, rng, pick, nb, order, rule, cap, capy, edgeair, slaby, unsetfill, lazynb, margin, until, ntypes;
    long long K;      /* number of initial random blocks */
    double sweeps;    /* iterations as multiple of W*H*D (used when M<0) */
    long long M;      /* explicit iteration count, overrides sweeps if >=0 */
} Variant;

/* neighbour orders: 0 = Notch's Tile.render face order (y-,y+,z-,z+,x-,x+); 1 = x-,x+,y-,y+,z-,z+ */
static const int ORD[2][6][3] = {
    {{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},{-1,0,0},{1,0,0}},
    {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}}};

static void vdefault(Variant *v) {
    memset(v, 0, sizeof *v);
    strcpy(v->id, "custom");
    v->init = INIT_EMPTY; v->seedtype = ST_ROCK; v->rng = RNG_INT; v->pick = PICK_XYZ; v->nb = NB_RAND;
    v->order = 0; v->rule = RULE_MONO; v->cap = CAP_NONE; v->capy = -1; v->edgeair = 0; v->slaby = -1;
    v->unsetfill = 0; v->K = 1000; v->sweeps = 32; v->M = -1;
}

static int parse_kv(Variant *v, const char *spec) {
    char buf[512]; strncpy(buf, spec, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        char *eq = strchr(tok, '='); if (!eq) { fprintf(stderr, "bad token %s\n", tok); return -1; }
        *eq = 0; const char *k = tok, *val = eq + 1;
#define IS(s) (strcmp(val, s) == 0)
        if (!strcmp(k, "init")) v->init = IS("empty") ? INIT_EMPTY : IS("unset") ? INIT_UNSET : IS("slab") ? INIT_SLAB : -1;
        else if (!strcmp(k, "seed")) v->seedtype = IS("rock") ? ST_ROCK : IS("air") ? ST_AIR : IS("rand2") ? ST_RAND2 : IS("rand3") ? ST_RAND3 : -1;
        else if (!strcmp(k, "rng")) v->rng = IS("int") ? RNG_INT : IS("math") ? RNG_MATH : -1;
        else if (!strcmp(k, "pick")) v->pick = IS("xyz") ? PICK_XYZ : IS("idx") ? PICK_IDX : IS("xzy") ? PICK_XZY : -1;
        else if (!strcmp(k, "nb")) v->nb = IS("rand") ? NB_RAND : IS("scan") ? NB_SCAN : -1;
        else if (!strcmp(k, "order")) v->order = atoi(val);
        else if (!strcmp(k, "rule")) v->rule = IS("mono") ? RULE_MONO : IS("voter") ? RULE_VOTER : -1;
        else if (!strcmp(k, "cap")) v->cap = IS("none") ? CAP_NONE : IS("post") ? CAP_POST : IS("fixed") ? CAP_FIXED : -1;
        else if (!strcmp(k, "capy")) v->capy = atoi(val);
        else if (!strcmp(k, "slaby")) v->slaby = atoi(val);
        else if (!strcmp(k, "edge")) v->edgeair = IS("air") ? 1 : 0;
        else if (!strcmp(k, "nbdraw")) v->lazynb = IS("iffree") ? 1 : 0;
        else if (!strcmp(k, "ntypes")) v->ntypes = atoi(val);
        else if (!strcmp(k, "margin")) v->margin = atoi(val);
        else if (!strcmp(k, "until")) v->until = atoi(val);
        else if (!strcmp(k, "unsetfill")) v->unsetfill = atoi(val);
        else if (!strcmp(k, "K")) v->K = atoll(val);
        else if (!strcmp(k, "sweeps")) { v->sweeps = atof(val); v->M = -1; }
        else if (!strcmp(k, "M")) v->M = atoll(val);
        else if (!strcmp(k, "id")) { strncpy(v->id, val, 63); }
        else { fprintf(stderr, "unknown key %s\n", k); return -1; }
#undef IS
    }
    if (v->init < 0 || v->seedtype < 0 || v->rng < 0 || v->pick < 0 || v->nb < 0 || v->rule < 0 || v->cap < 0) { fprintf(stderr, "bad value\n"); return -1; }
    return 0;
}

/* ---- registry of principled variants (justification in REPORT.md) ---- */
typedef struct { const char *id; const char *spec; } Reg;
static const Reg REG[] = {
    /* G1: literal reading. empty world, K rock seeds, pick random tile, if empty & next to a block copy it (Eden growth). */
    {"G1a", "init=empty,seed=rock,rng=int,pick=xyz,nb=scan,rule=mono,K=1000,sweeps=32"},
    {"G1b", "init=empty,seed=rock,rng=math,pick=xyz,nb=scan,rule=mono,K=1000,sweeps=32"},
    {"G1c", "init=empty,seed=rock,rng=int,pick=xyz,nb=rand,rule=mono,K=1000,sweeps=32"},
    /* G2: typed seeds (air and rock are both 'tiles'); unset cells adopt a set neighbour's type -> space partition into air/rock domains */
    {"G2a", "init=unset,seed=rand2,rng=int,pick=xyz,nb=scan,rule=mono,K=1000,sweeps=32"},
    {"G2b", "init=unset,seed=rand2,rng=int,pick=xyz,nb=rand,rule=mono,K=1000,sweeps=32"},
    {"G2c", "init=unset,seed=rand2,rng=math,pick=xyz,nb=rand,rule=mono,K=1000,sweeps=32"},
    {"G2d", "init=unset,seed=rand2,rng=int,pick=idx,nb=rand,rule=mono,K=1000,sweeps=32"},
    /* G3: rd-132211 slab constructor (y<=d*2/3 rock) as starting state, random blocks of random type, then voter copying */
    {"G3a", "init=slab,seed=rand2,rng=int,pick=xyz,nb=rand,rule=voter,K=1000,sweeps=32"},
    {"G3b", "init=slab,seed=air,rng=int,pick=xyz,nb=rand,rule=voter,K=1000,sweeps=32"},
    {"G3c", "init=slab,seed=rand2,rng=math,pick=xyz,nb=rand,rule=voter,K=1000,sweeps=32"},
    /* G4: binary voter from random noise (every 'random block' copies a random neighbour) */
    {"G4a", "init=empty,seed=rock,rng=int,pick=xyz,nb=rand,rule=voter,K=2097152,sweeps=32"},
    /* G5: typed partition with cap: cells above the slab level are fixed air (domain = slab) */
    {"G5a", "init=unset,seed=rand2,rng=int,pick=xyz,nb=rand,rule=mono,K=1000,sweeps=32,cap=post,capy=42"},
    /* U: multi-type growth into UNSET cells, boundary never picked (margin=1, x=nextInt(W-2)+1), leftover unset -> rock,
       growth domain capped at the rd-132211 slab level d*2/3=42. Survives the frozen look-alike tests (REPORT.md). */
    {"U2s", "init=unset,ntypes=2,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,sweeps=32,unsetfill=1,margin=1,cap=fixed,capy=42"},
    {"U4s", "init=unset,ntypes=4,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,sweeps=32,unsetfill=1,margin=1,cap=fixed,capy=42"},
    {"U6s", "init=unset,ntypes=6,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,sweeps=32,unsetfill=1,margin=1,cap=fixed,capy=42"},
    {"U6sm", "init=unset,ntypes=6,rng=math,pick=xyz,nb=scan,rule=mono,K=1024,sweeps=32,unsetfill=1,margin=1,cap=fixed,capy=42"},
    {"U6r", "init=unset,ntypes=6,rng=int,pick=xyz,nb=rand,rule=mono,K=4096,sweeps=64,unsetfill=1,margin=1,cap=fixed,capy=42"},
    {"U6rm", "init=unset,ntypes=6,rng=math,pick=xyz,nb=rand,rule=mono,K=4096,sweeps=64,unsetfill=1,margin=1,cap=fixed,capy=42"},
    {"U8s", "init=unset,ntypes=8,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,sweeps=32,unsetfill=1,margin=1,cap=fixed,capy=42"},
    {NULL, NULL}};

static int vload(Variant *v, const char *s) {
    vdefault(v);
    for (int i = 0; REG[i].id; i++)
        if (!strcmp(REG[i].id, s)) { int r = parse_kv(v, REG[i].spec); strcpy(v->id, REG[i].id); return r; }
    /* allow "ID:overrides" */
    const char *colon = strchr(s, ':');
    if (colon) {
        char base[64]; size_t n = colon - s; if (n > 63) n = 63; memcpy(base, s, n); base[n] = 0;
        for (int i = 0; REG[i].id; i++)
            if (!strcmp(REG[i].id, base)) {
                if (parse_kv(v, REG[i].spec)) return -1;
                snprintf(v->id, 64, "%s", s);
                return parse_kv(v, colon + 1);
            }
    }
    if (strchr(s, '=')) return parse_kv(v, s);
    fprintf(stderr, "unknown variant %s\n", s); return -1;
}

static inline int draw(JRand *r, const Variant *v, int n) {
    if (v->rng == RNG_INT) return jr_nextInt(r, n);
    return (int)(jr_nextDouble(r) * n); /* (int)(Math.random()*n) */
}
static inline int isPow2(long long n) { return n > 0 && (n & (n - 1)) == 0; }

static long long iters(const Variant *v, int W, int H, int D) {
    return v->M >= 0 ? v->M : (long long)(v->sweeps * (double)W * H * D + 0.5);
}

/* closed-form step count (LCG steps). returns -1 if not fixed (non-pow2 nextInt) - then also gives the
   'no-rejection' base value through *base. */
static long long steps_formula(const Variant *v, int W, int H, int D, long long *base, char *expl, size_t el) {
    int u = v->rng == RNG_MATH ? 2 : 1;
    int seedDraws = (v->pick == PICK_IDX ? 1 : 3) + ((v->seedtype >= ST_RAND2 || v->ntypes > 1) ? 1 : 0);
    int pickDraws = v->pick == PICK_IDX ? 1 : 3;
    int nbDraws = v->nb == NB_RAND ? 1 : 0;
    long long M = iters(v, W, H, D);
    long long b = (long long)u * (v->K * seedDraws + M * (pickDraws + nbDraws));
    *base = b;
    int fixed = 1;
    if (v->rng == RNG_INT) {
        if (!isPow2(W - 2*v->margin) || !isPow2(H - 2*v->margin) || !isPow2(D - 2*v->margin)) fixed = 0;
        if (v->nb == NB_RAND) fixed = 0; /* nextInt(6) rejection: p = 2/2^31 per call */
        if (v->seedtype == ST_RAND3) fixed = 0;
        if (v->ntypes > 1 && !isPow2(v->ntypes)) fixed = 0;
    }
    if (v->until) fixed = 0; /* iteration count itself is data-dependent */
    if (v->lazynb && v->nb == NB_RAND && v->rule == RULE_MONO) {
        fixed = 0; /* data-dependent: neighbour draw only when picked cell is free */
        if (v->pick == PICK_IDX && !isPow2((long long)W * H * D)) fixed = 0;
    }
    snprintf(expl, el, "N = %d*(K*%d + M*%d) = %d*(%lld*%d + %lld*%d) = %lld%s", u, seedDraws, pickDraws + nbDraws,
             u, v->K, seedDraws, M, pickDraws + nbDraws, b,
             fixed ? "" : " + R  (R = nextInt rejection redraws for non-power-of-two bounds, P per call = (2^31 mod n)/2^31: n=254 -> 3.7e-9, n=62 or 6 -> 9.3e-10; or data-dependent if until=1 or nbdraw=iffree)");
    return fixed ? b : -1;
}

typedef struct { long long solid, air, rock, typed2, unset_left, copies, iters; } Stats;

#define PICK(x, y, z) do { const int m_ = v->margin; \
    if (v->pick == PICK_IDX) { long long i_ = draw(r, v, (int)N); x = i_ % W; z = (i_ / W) % H; y = i_ / ((long long)W * H); } \
    else if (v->pick == PICK_XZY) { x = draw(r, v, W - 2*m_) + m_; z = draw(r, v, H - 2*m_) + m_; y = draw(r, v, D - 2*m_) + m_; } \
    else { x = draw(r, v, W - 2*m_) + m_; y = draw(r, v, D - 2*m_) + m_; z = draw(r, v, H - 2*m_) + m_; } } while (0)
static FILE *SEEDLOG = NULL;
static uint16_t *OWN = NULL; /* optional territory tracking (does not touch RNG) */
static void generate(const Variant *v, JRand *r, uint8_t *a, int W, int H, int D, Stats *st) {
    const long long N = (long long)W * H * D;
    int slaby = v->slaby >= 0 ? v->slaby : D * 2 / 3;
    int capy = v->capy >= 0 ? v->capy : D * 2 / 3;
    /* initial fill */
    for (int y = 0; y < D; y++)
        for (int z = 0; z < H; z++)
            for (int x = 0; x < W; x++) {
                long long i = ((long long)y * H + z) * W + x;
                uint8_t val = 0;
                if (v->init == INIT_UNSET) val = UNSET;
                else if (v->init == INIT_SLAB) val = (y <= slaby) ? 1 : 0;
                if (v->cap == CAP_FIXED && y > capy) val = 0;
                a[i] = val;
            }
    /* random blocks */
    for (long long k = 0; k < v->K; k++) {
        int x, y, z;
        PICK(x, y, z);
        uint8_t t = 1;
        if (v->seedtype == ST_AIR) t = 0;
        else if (v->seedtype == ST_RAND2) t = (uint8_t)draw(r, v, 2);       /* 0 air / 1 rock */
        else if (v->seedtype == ST_RAND3) t = (uint8_t)draw(r, v, 3);       /* 0 air / 1 rock / 2 grass-type */
        if (v->ntypes > 1) t = draw(r, v, v->ntypes) == 0 ? 0 : 1;           /* one draw: tile id in [0,n), id 0 = air, others solid */
        if (SEEDLOG) fprintf(SEEDLOG, "%lld %d %d %d %d\n", k, x, y, z, t);
        if (v->cap == CAP_FIXED && y > capy) continue;
        a[((long long)y * H + z) * W + x] = t;
        if (OWN) OWN[((long long)y * H + z) * W + x] = (uint16_t)(k + 1 > 65535 ? 65535 : k + 1);
    }
    /* growth */
    const long long M = iters(v, W, H, D);
    const int (*O)[3] = ORD[v->order & 1];
    const uint8_t FREE = v->init == INIT_UNSET ? UNSET : 0; /* mono rule: only free cells change, only from non-free */
    long long copies = 0, unsetCount = 0;
    const int mg = v->pick == PICK_IDX ? 0 : v->margin;
    #define PICKABLE(x, y, z) ((x) >= mg && (x) < W - mg && (y) >= mg && (y) < D - mg && (z) >= mg && (z) < H - mg && !(v->cap == CAP_FIXED && (y) > capy))
    if (v->until) for (int y = 0; y < D; y++) for (int z = 0; z < H; z++) for (int x = 0; x < W; x++)
        if (a[((long long)y * H + z) * W + x] == UNSET && PICKABLE(x, y, z)) unsetCount++;
    long long itDone = 0;
    for (long long it = 0; v->until ? (unsetCount > 0 && it < (long long)4e10) : it < M; it++) {
        itDone = it + 1;
        int x, y, z;
        PICK(x, y, z);
        long long i = ((long long)y * H + z) * W + x;
        int dsel = -1;
        if (v->nb == NB_RAND && !(v->lazynb && v->rule == RULE_MONO)) dsel = draw(r, v, 6);
        if (v->cap == CAP_FIXED && y > capy) continue;
        if (v->rule == RULE_MONO) {
            if (a[i] != FREE) continue;
            if (v->nb == NB_RAND && v->lazynb) dsel = draw(r, v, 6); /* draw only for free cells: N data-dependent */
            if (dsel >= 0) {
                int xx = x + O[dsel][0], yy = y + O[dsel][1], zz = z + O[dsel][2];
                if (xx < 0 || yy < 0 || zz < 0 || xx >= W || yy >= D || zz >= H) continue;
                uint8_t s = a[((long long)yy * H + zz) * W + xx];
                if (s != FREE && !(v->cap == CAP_FIXED && yy > capy)) { a[i] = s; if (OWN) OWN[i] = OWN[((long long)yy * H + zz) * W + xx]; copies++; if (FREE == UNSET) unsetCount--; }
            } else {
                for (int d = 0; d < 6; d++) {
                    int xx = x + O[d][0], yy = y + O[d][1], zz = z + O[d][2];
                    if (xx < 0 || yy < 0 || zz < 0 || xx >= W || yy >= D || zz >= H) continue;
                    uint8_t s = a[((long long)yy * H + zz) * W + xx];
                    if (v->cap == CAP_FIXED && yy > capy) continue;
                    if (s != FREE) { a[i] = s; if (OWN) OWN[i] = OWN[((long long)yy * H + zz) * W + xx]; copies++; if (FREE == UNSET) unsetCount--; break; }
                }
            }
        } else { /* voter: copy the chosen neighbour whatever it is */
            int d = dsel >= 0 ? dsel : 0;
            int xx = x + O[d][0], yy = y + O[d][1], zz = z + O[d][2];
            uint8_t s;
            if (xx < 0 || yy < 0 || zz < 0 || xx >= W || yy >= D || zz >= H) { if (!v->edgeair) continue; s = 0; }
            else s = a[((long long)yy * H + zz) * W + xx];
            if (a[i] != s) { a[i] = s; copies++; }
        }
    }
    /* finalise */
    memset(st, 0, sizeof *st);
    st->copies = copies; st->iters = itDone;
    for (long long i = 0; i < N; i++) {
        if (a[i] == UNSET) { st->unset_left++; a[i] = (uint8_t)v->unsetfill; }
        int y = (int)(i / ((long long)W * H));
        if (v->cap == CAP_POST && y > capy) a[i] = 0;
        if (a[i]) st->solid++; else st->air++;
        if (a[i] == 1) st->rock++;
        if (a[i] == 2) st->typed2++;
    }
}

static uint64_t parse_u64(const char *s) { return strtoull(s, NULL, 0); }

static void print_variant_json(const Variant *v, int W, int H, int D) {
    long long base; char ex[512];
    long long f = steps_formula(v, W, H, D, &base, ex, sizeof ex);
    printf("\"variant\":\"%s\",\"init\":%d,\"seedtype\":%d,\"rng\":\"%s\",\"pick\":%d,\"nb\":\"%s\",\"order\":%d,\"rule\":\"%s\","
           "\"cap\":%d,\"K\":%lld,\"M\":%lld,\"W\":%d,\"H\":%d,\"D\":%d,\"steps_formula\":%lld,\"steps_base\":%lld,\"steps_formula_text\":\"%s\"",
           v->id, v->init, v->seedtype, v->rng ? "math" : "int", v->pick, v->nb ? "scan" : "rand", v->order,
           v->rule ? "voter" : "mono", v->cap, v->K, iters(v, W, H, D), W, H, D, f, base, ex);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "see header\n"); return 2; }
    if (!strcmp(argv[1], "list")) { for (int i = 0; REG[i].id; i++) printf("%s %s\n", REG[i].id, REG[i].spec); return 0; }
    if (!strcmp(argv[1], "rewind") || !strcmp(argv[1], "advance")) {
        if (argc < 4) return 2;
        uint64_t s = parse_u64(argv[2]) & JR_MASK, n = parse_u64(argv[3]);
        printf("%llu\n", (unsigned long long)(argv[1][0] == 'r' ? jr_rewind(s, n) : jr_advance(s, n)));
        return 0;
    }
    if (!strcmp(argv[1], "scramble")) { printf("%llu\n", (unsigned long long)jr_scramble(strtoll(argv[2], NULL, 0))); return 0; }
    if (!strcmp(argv[1], "steps")) {
        Variant v; if (vload(&v, argv[2])) return 2;
        int W = argc > 5 ? atoi(argv[3]) : 256, H = argc > 5 ? atoi(argv[4]) : 256, D = argc > 5 ? atoi(argv[5]) : 64;
        printf("{"); print_variant_json(&v, W, H, D); printf("}\n"); return 0;
    }
    if (!strcmp(argv[1], "run")) {
        if (argc < 5) return 2;
        Variant v; if (vload(&v, argv[2])) return 2;
        uint64_t s0 = parse_u64(argv[3]) & JR_MASK;
        int W = argc > 7 ? atoi(argv[5]) : 256, H = argc > 7 ? atoi(argv[6]) : 256, D = argc > 7 ? atoi(argv[7]) : 64;
        long long N = (long long)W * H * D;
        uint8_t *a = malloc(N);
        JRand r = {s0, 0, 0};
        if (getenv("GEN_OWNER")) OWN = calloc(N, 2);
        if (getenv("GEN_SEEDLOG")) SEEDLOG = fopen(getenv("GEN_SEEDLOG"), "w");
        Stats st; clock_t t0 = clock();
        generate(&v, &r, a, W, H, D, &st);
        double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        if (strcmp(argv[4], "-")) { FILE *f = fopen(argv[4], "wb"); fwrite(a, 1, N, f); fclose(f); }
        /* FNV-1a 64 of array for cross-checks */
        uint64_t h = 1469598103934665603ULL; for (long long i = 0; i < N; i++) { h ^= a[i]; h *= 1099511628211ULL; }
        printf("{"); print_variant_json(&v, W, H, D);
        printf(",\"state_in\":%llu,\"state_out\":%llu,\"steps\":%llu,\"rejections\":%llu,\"solid\":%lld,\"air\":%lld,\"rock\":%lld,\"type2\":%lld,"
               "\"unset_left\":%lld,\"copies\":%lld,\"iterations_run\":%lld,\"fnv64\":\"%016llx\",\"seconds\":%.3f}\n",
               (unsigned long long)s0, (unsigned long long)r.s, (unsigned long long)r.steps, (unsigned long long)r.rejections,
               st.solid, st.air, st.rock, st.typed2, st.unset_left, st.copies, st.iters, (unsigned long long)h, secs);
        if (SEEDLOG) fclose(SEEDLOG);
        if (OWN) { FILE *fo = fopen(getenv("GEN_OWNER"), "wb"); fwrite(OWN, 2, N, fo); fclose(fo); free(OWN); }
        free(a); return 0;
    }
    fprintf(stderr, "unknown command\n"); return 2;
}
