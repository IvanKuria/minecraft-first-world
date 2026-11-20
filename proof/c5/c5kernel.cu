// C5 search on the GPU: a CUDA port of c5search.c (CPU reference). Same CLI, same PASS/SUMMARY lines.
// Built and validated on the Windows PC (RTX 5070 Ti, sm_120, CUDA 13.4); see ../pc-c5/README.md.
//
// usage: c5gpu <variant> <cands.txt> <obs[,obs2..]> <f_hz> <dmin_s> <dmax_s> <J> <cmin[,..]> <imin[,..]> <threads(ignored)> [q_lo q_hi]
//   identical meaning to the multi-map c5search (R1MAP, Oct 15): each obs file registered to its "R 1|2" reset (x1,z1 |
//   x2,z2); a seed passes iff both stages pass on every map. Output: "PASS cand j q_g G0 hits0 iou0 [hits1 iou1 ..]"
//   sorted by (cand, j, q), then the same SUMMARY line (per-map s1/s2 counters with the CPU's in-order early exit).
//   Env C5_SELFTEST=n : compare the device seed list with the CPU reference (seeds_from_state) on n random states first.
//
// Semantics reproduced exactly (bit-for-bit) from c5search.c:
//   t  = (int64)(((double)q / (double)f) * 1e9)       IEEE div/mul, no FMA (compile with -fmad=false)
//   G0 = ((U + k_g + t) ^ 0x5DEECE66D) & (2^48-1),    k_g = k_m - j (skipped if < 1)
//   seed phase = GEN's PICK + type draws with java.util.Random.nextInt (rejection loop exact), kept iff y <= capy
//   ROI: |x - floor(x2)| <= 62 and |z - floor(z2)| <= 62, seed coordinates as floats rx+.5, y+.5, rz+.5
//   nearest kept ROI seed by squared float distance ((dx*dx + dy*dy) + dz*dz), strict <, first in draw order wins
//   stage 1: #centroids whose nearest seed is air (t == 0) >= cmin ; stage 2: IoU = tp / (nopen + fp) >= imin (double)
//
// Build (vcvars64 shell):
//   cl /c /O2 /std:c11 /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS ..\pc-c5\c5variant.c
//   cl /c /O2 /std:c11 /DC5GEN_NO_MAIN /D_CRT_SECURE_NO_WARNINGS ..\pc-c5\c5gen.c
//   nvcc -O3 -arch=sm_120 -fmad=false -o c5gpu.exe c5kernel.cu c5variant.obj c5gen.obj
// Extended seed-draw specs "X:K=..,nt=..,m=..,yb=..,tf=0|1" (V-full: m=0,yb=64; V-full42: m=0,yb=43) use c5gen.c's
// definition; the CPU reference for their seed lists is c5gen (C5_SELFTEST compares against it).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>

extern "C" int c5_variant(const char *spec, long long *K, int *pick, int *margin, int *ntypes, int *seedtype, int *rng, int *cap, int *capy);
// X specs (PC lane, rd-160052 idiom variants): parsed/drawn by ../pc-c5/c5gen.c (CPU reference, validated vs gen)
struct C5GSpec { long long K; int nt, m, yb, tf, nbrand, until, edge; double sweeps; int capy, surf; };
struct C5GRand { uint64_t s, steps, rejections; };
extern "C" int c5g_parse(const char *s, C5GSpec *p);
extern "C" int c5g_seed(C5GRand *r, const C5GSpec *p, int *x, int *y, int *z, int *t);
extern "C" int c5_seeds(const char *spec, unsigned long long s0, short *x, short *y, short *z, unsigned char *t, unsigned char *kept);

#define MASK48 0xFFFFFFFFFFFFULL
#define MULT 0x5DEECE66DULL
#define ADDC 0xBULL
#define U0 8682522807148012LL
#define MAXC 32
#define MAXMAP 4
#define ROI 62
#define MAXROI 3072        // shared-memory seed list per stage-2 block (K <= ~16k at the real ROI fraction)

struct VarP { int K, pick, margin, ntypes, seedtype, cap, capy, yb, tf; float surfc; };   // surfc: C5 surface plane centre (42.5 old; 63.5 full-height)   // yb: y bound of pick (GEN: 64-2m); tf: type drawn first (X specs)   // pick: 0 xyz, 2 xzy; seedtype: 0 rock 1 air 2 rand2 3 rand3
struct Seg { long long qlo, count, start; int ci, j, kg, raw, fx[MAXMAP], fz[MAXMAP]; };   // raw: 0 JDK6 new Random(), 1 new Random(q), 2 new Random(nanoTime(q))

__constant__ VarP c_v;
__constant__ float c_cx[MAXC], c_cz[MAXC];
__constant__ int c_cmap[MAXC], c_ncomp, c_nmap, c_cminm[MAXMAP], c_nseg;
__constant__ long long c_fhz;

__device__ __forceinline__ int nextInt(uint64_t &s, int n) {
    if ((n & -n) == n) { s = (s * MULT + ADDC) & MASK48; return (int)(((long long)n * (long long)(int)(s >> 17)) >> 31); }
    for (;;) {
        s = (s * MULT + ADDC) & MASK48;
        int bits = (int)(s >> 17), val = bits % n;
        if ((int)((unsigned)bits - (unsigned)val + (unsigned)(n - 1)) >= 0) return val;
    }
}
// one seed of GEN's seed phase (generate()/seeds_from_state order); returns kept
__device__ __forceinline__ bool draw_seed(uint64_t &s, int &x, int &y, int &z, int &t) {
    const int m = c_v.margin;
    if (c_v.tf) t = nextInt(s, c_v.ntypes) == 0 ? 0 : 1;
    if (c_v.pick == 2) { x = nextInt(s, 256 - 2 * m) + m; z = nextInt(s, 256 - 2 * m) + m; y = nextInt(s, 64 - 2 * m) + m; }
    else { x = nextInt(s, 256 - 2 * m) + m; y = nextInt(s, c_v.yb) + m; z = nextInt(s, 256 - 2 * m) + m; }
    if (c_v.tf) return !(c_v.cap == 2 && y > c_v.capy);
    t = 1;
    if (c_v.seedtype == 1) t = 0;
    else if (c_v.seedtype == 2) t = nextInt(s, 2);
    else if (c_v.seedtype == 3) t = nextInt(s, 3);
    if (c_v.ntypes > 1) t = nextInt(s, c_v.ntypes) == 0 ? 0 : 1;
    return !(c_v.cap == 2 && y > c_v.capy);
}
__device__ __forceinline__ uint64_t g0_of(long long q, int kg) {
    long long t = (long long)__dmul_rn(__ddiv_rn((double)q, (double)c_fhz), 1000000000.0);
    return ((uint64_t)(U0 + (long long)kg + t) ^ MULT) & MASK48;
}
__device__ __forceinline__ uint64_t seed_of(const Seg &sg, long long q) {
    if (sg.raw == 1) return ((uint64_t)q ^ MULT) & MASK48;
    if (sg.raw == 2) { long long t = (long long)__dmul_rn(__ddiv_rn((double)q, (double)c_fhz), 1000000000.0); return ((uint64_t)t ^ MULT) & MASK48; }
    return g0_of(q, sg.kg);
}

__global__ void selftest_kernel(const uint64_t *states, int n, short *ox, short *oy, short *oz, unsigned char *ot, unsigned char *ok) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint64_t s = states[i];
    for (int k = 0; k < c_v.K; k++) {
        int x, y, z, t; bool kept = draw_seed(s, x, y, z, t);
        long long o = (long long)i * c_v.K + k; ox[o] = x; oy[o] = y; oz[o] = z; ot[o] = t; ok[o] = kept;
    }
}


struct Rec { int seg; int hits[MAXMAP]; long long q; };

__global__ void stage1(const Seg *segs, long long base, long long n, Rec *queue, unsigned int *qn, unsigned int qcap,
                       unsigned long long *s1m) {
    long long gid = base + blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (gid >= base + n) return;
    int si = 0; while (si + 1 < c_nseg && segs[si + 1].start <= gid) si++;
    const Seg sg = segs[si];
    long long q = sg.qlo + (gid - sg.start);
    uint64_t s = seed_of(sg, q);
    float best[MAXC]; int bt[MAXC];
    for (int c = 0; c < c_ncomp; c++) { best[c] = 1e30f; bt[c] = 1; }
    for (int k = 0; k < c_v.K; k++) {
        int x, y, z, t;
        if (!draw_seed(s, x, y, z, t)) continue;
        for (int m = 0; m < c_nmap; m++) {
            int rx = x - sg.fx[m], rz = z - sg.fz[m];
            if (rx < -ROI || rx > ROI || rz < -ROI || rz > ROI) continue;
            float sx = rx + .5f, sy = y + .5f, sz = rz + .5f;
            for (int c = 0; c < c_ncomp; c++) {
                if (c_cmap[c] != m) continue;
                float dx = sx - c_cx[c], dy = sy - c_v.surfc, dz = sz - c_cz[c];
                float d = __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
                if (d < best[c]) { best[c] = d; bt[c] = t; }
            }
        }
    }
    int hits[MAXMAP];
    for (int m = 0; m < c_nmap; m++) hits[m] = 0;
    for (int c = 0; c < c_ncomp; c++) hits[c_cmap[c]] += (bt[c] == 0);
    for (int m = 0; m < c_nmap; m++) {           // CPU order: maps tested in order, early exit at the first failure
        if (hits[m] < c_cminm[m]) return;
        atomicAdd(&s1m[m], 1ULL);
    }
    unsigned int slot = atomicAdd(qn, 1u);
    if (slot < qcap) { queue[slot].seg = si; queue[slot].q = q; for (int m = 0; m < MAXMAP; m++) queue[slot].hits[m] = m < c_nmap ? hits[m] : 0; }
}

// one block per stage-1 survivor; IoU on every map (host applies the per-map thresholds in map order)
__global__ void stage2(const Seg *segs, const Rec *queue, const short2 *obs, const unsigned char *olab, const int *obase, const int *ocount,
                       int *tp_out, int *fp_out) {
    __shared__ float sx[MAXROI], sy[MAXROI], sz[MAXROI];
    __shared__ unsigned char st[MAXROI];
    __shared__ int mroi, overflow, tp_s, fp_s;
    const Rec r = queue[blockIdx.x];
    const Seg sg = segs[r.seg];
    for (int m = 0; m < c_nmap; m++) {
        if (threadIdx.x == 0) {
            uint64_t s = seed_of(sg, r.q);
            int mm = 0, of = 0;
            for (int k = 0; k < c_v.K; k++) {
                int x, y, z, t;
                if (!draw_seed(s, x, y, z, t)) continue;
                int rx = x - sg.fx[m], rz = z - sg.fz[m];
                if (rx < -ROI || rx > ROI || rz < -ROI || rz > ROI) continue;
                if (mm >= MAXROI) { of = 1; continue; }
                sx[mm] = rx + .5f; sy[mm] = y + .5f; sz[mm] = rz + .5f; st[mm] = (unsigned char)t; mm++;
            }
            mroi = mm; overflow = of; tp_s = 0; fp_s = 0;
        }
        __syncthreads();
        if (overflow) { if (threadIdx.x == 0) { tp_out[blockIdx.x * MAXMAP + m] = -1; fp_out[blockIdx.x * MAXMAP + m] = -1; } __syncthreads(); continue; }
        int tp = 0, fp = 0;
        for (int o = obase[m] + threadIdx.x; o < obase[m] + ocount[m]; o += blockDim.x) {
            float px = obs[o].x + .5f, pz = obs[o].y + .5f; float best = 1e30f; int b = 1;
            for (int i = 0; i < mroi; i++) {
                float dx = sx[i] - px, dy = sy[i] - c_v.surfc, dz = sz[i] - pz;
                float d = __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
                if (d < best) { best = d; b = st[i]; }
            }
            int pred = b == 0;
            if (olab[o]) { if (pred) tp++; } else if (pred) fp++;
        }
        atomicAdd(&tp_s, tp); atomicAdd(&fp_s, fp);
        __syncthreads();
        if (threadIdx.x == 0) { tp_out[blockIdx.x * MAXMAP + m] = tp_s; fp_out[blockIdx.x * MAXMAP + m] = fp_s; }
        __syncthreads();
    }
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(3); } } while (0)

static int selftest(const char *spec, const VarP &vp, int n) {
    std::vector<uint64_t> st(n); uint64_t g = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < n; i++) { g ^= g << 13; g ^= g >> 7; g ^= g << 17; st[i] = g & MASK48; }
    long long tot = (long long)n * vp.K;
    uint64_t *ds; short *dx, *dy, *dz; unsigned char *dt, *dk;
    CK(cudaMalloc(&ds, 8 * n)); CK(cudaMalloc(&dx, 2 * tot)); CK(cudaMalloc(&dy, 2 * tot)); CK(cudaMalloc(&dz, 2 * tot)); CK(cudaMalloc(&dt, tot)); CK(cudaMalloc(&dk, tot));
    CK(cudaMemcpy(ds, st.data(), 8 * n, cudaMemcpyHostToDevice));
    selftest_kernel<<<(n + 127) / 128, 128>>>(ds, n, dx, dy, dz, dt, dk); CK(cudaDeviceSynchronize());
    std::vector<short> hx(tot), hy(tot), hz(tot); std::vector<unsigned char> ht(tot), hk(tot);
    CK(cudaMemcpy(hx.data(), dx, 2 * tot, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hy.data(), dy, 2 * tot, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hz.data(), dz, 2 * tot, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ht.data(), dt, tot, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hk.data(), dk, tot, cudaMemcpyDeviceToHost));
    std::vector<short> rx(vp.K), ry(vp.K), rz(vp.K); std::vector<unsigned char> rt(vp.K), rk(vp.K);
    long long bad = 0;
    for (int i = 0; i < n; i++) {
        if (!strncmp(spec, "X:", 2)) {
            C5GSpec g; c5g_parse(spec + 2, &g); C5GRand r = {st[i], 0, 0};
            for (int k = 0; k < vp.K; k++) { int x, y, z, t; rk[k] = (unsigned char)c5g_seed(&r, &g, &x, &y, &z, &t); rx[k] = x; ry[k] = y; rz[k] = z; rt[k] = t; }
        } else c5_seeds(spec, st[i], rx.data(), ry.data(), rz.data(), rt.data(), rk.data());
        for (int k = 0; k < vp.K; k++) {
            long long o = (long long)i * vp.K + k;
            bad += hx[o] != rx[k] || hy[o] != ry[k] || hz[o] != rz[k] || ht[o] != rt[k] || hk[o] != rk[k];
        }
    }
    fprintf(stderr, "SELFTEST variant=%s states=%d seeds=%lld mismatches=%lld\n", spec, n, tot, bad);
    cudaFree(ds); cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dt); cudaFree(dk);
    return bad != 0;
}

static int split_list(const char *s, std::vector<std::string> &out) {
    out.clear(); const char *p = s;
    for (;;) { const char *e = strchr(p, ','); out.push_back(e ? std::string(p, e - p) : std::string(p)); if (!e) break; p = e + 1; }
    return (int)out.size();
}

static uint64_t host_seed(int raw, long long q, int kg, long long FHZ) {
    if (raw == 1) return ((uint64_t)q ^ MULT) & MASK48;                                   // new Random(literal long)
    long long t = (long long)(((double)q / (double)FHZ) * 1000000000.0);
    if (raw == 2) return ((uint64_t)t ^ MULT) & MASK48;                                   // new Random(System.nanoTime())
    return ((uint64_t)(U0 + kg + t) ^ MULT) & MASK48;                                      // JDK6 new Random()
}

int main(int argc, char **argv) {
    if (argc < 11) { fprintf(stderr, "see header\n"); return 2; }
    const char *spec = argv[1];
    long long K; VarP vp; int rng;
    bool xspec = !strncmp(spec, "X:", 2);
    if (xspec) {
        C5GSpec g; if (c5g_parse(spec + 2, &g)) { fprintf(stderr, "bad X spec%c", 10); return 2; }
        K = g.K; vp.pick = 0; vp.margin = g.m; vp.ntypes = g.nt; vp.seedtype = 0; rng = 0; vp.cap = 2; vp.capy = g.capy; vp.yb = g.yb; vp.tf = g.tf; vp.surfc = g.surf + .5f;
    } else {
        if (c5_variant(spec, &K, &vp.pick, &vp.margin, &vp.ntypes, &vp.seedtype, &rng, &vp.cap, &vp.capy)) return 2;
        vp.yb = 64 - 2 * vp.margin; vp.tf = 0; vp.surfc = getenv("C5_SURF") ? (float)atof(getenv("C5_SURF")) + .5f : 42.5f;
    }
    if (rng != 0) { fprintf(stderr, "C5 search is for own-Random (rng=int) variants\n"); return 2; }
    if (vp.pick == 1) { fprintf(stderr, "pick=idx not supported on GPU\n"); return 2; }
    vp.K = (int)K;
    CK(cudaMemcpyToSymbol(c_v, &vp, sizeof vp));
    if (getenv("C5_SELFTEST") && selftest(spec, vp, atoi(getenv("C5_SELFTEST")))) return 4;

    // maps
    std::vector<std::string> mp, cl, il; int NM = split_list(argv[3], mp);
    if (NM < 1 || NM > MAXMAP) { fprintf(stderr, "1..%d maps\n", MAXMAP); return 2; }
    int nc = split_list(argv[8], cl), ni = split_list(argv[9], il);
    if ((nc != 1 && nc != NM) || (ni != 1 && ni != NM)) { fprintf(stderr, "cmin/imin: give 1 value or one per map\n"); return 2; }
    std::vector<int> mref(NM, 2), mcmin(NM), obase(NM), ocount(NM), mncomp(NM), mnopen(NM); std::vector<double> mimin(NM);
    std::vector<short2> obs; std::vector<unsigned char> olab; std::vector<float> ccx, ccz; std::vector<int> cmap;
    int need1 = 0;
    for (int m = 0; m < NM; m++) {
        mcmin[m] = atoi(cl[nc == 1 ? 0 : m].c_str()); mimin[m] = atof(il[ni == 1 ? 0 : m].c_str());
        FILE *f = fopen(mp[m].c_str(), "r"); if (!f) { fprintf(stderr, "cannot open %s\n", mp[m].c_str()); return 2; }
        obase[m] = (int)obs.size(); mncomp[m] = 0; mnopen[m] = 0;
        char line[256];
        while (fgets(line, sizeof line, f)) {
            int a1, a2, a3, a4; double c1, c2; int sz;
            if (line[0] == 'O' && sscanf(line + 1, "%d %d %d %d", &a1, &a2, &a3, &a4) == 4) { obs.push_back(make_short2((short)a1, (short)a2)); olab.push_back((unsigned char)a3); mnopen[m] += a3 != 0; }
            else if (line[0] == 'C' && sscanf(line + 1, "%lf %lf %d", &c1, &c2, &sz) == 3) { ccx.push_back((float)c1); ccz.push_back((float)c2); cmap.push_back(m); mncomp[m]++; }
            else if (line[0] == 'R' && sscanf(line + 1, "%d", &a1) == 1) { if (a1 != 1 && a1 != 2) return 2; mref[m] = a1; }
        }
        fclose(f);
        ocount[m] = (int)obs.size() - obase[m]; need1 |= mref[m] == 1;
    }
    int NCOMP = (int)ccx.size();
    if (NCOMP > MAXC) { fprintf(stderr, "too many components\n"); return 2; }

    struct Cand { unsigned long long so; int calls, km; long long qm; double x2, z2, x1, z1; };
    std::vector<Cand> CA;
    FILE *f = fopen(argv[2], "r"); if (!f) return 2;
    { char line[512];
      while (fgets(line, sizeof line, f)) {
          Cand c; c.x1 = c.z1 = 0;
          int got = sscanf(line, "%llu %d %d %lld %lf %lf %lf %lf", &c.so, &c.calls, &c.km, &c.qm, &c.x2, &c.z2, &c.x1, &c.z1);
          if (got < 6) continue;
          if (need1 && got < 8) { fprintf(stderr, "a map is registered to R 1 but a candidate line lacks x1 z1\n"); return 2; }
          CA.push_back(c);
      } }
    fclose(f);
    long long FHZ = atoll(argv[4]); double DMIN = atof(argv[5]), DMAX = atof(argv[6]);
    int J = atoi(argv[7]);
    long long QLO = -1, QHI = -1; if (argc > 12) { QLO = atoll(argv[11]); QHI = atoll(argv[12]); }

    // Segments in the same (ci, j) order as the CPU worker loop. PC-lane extensions (env, reported as j = 0):
    //   C5_AFTER=s        Math's Random created first: k_g = k_m + 1, q_g in [q_m, q_m + s*f]
    //   C5_SEEDMODE=nano  generator = new Random(System.nanoTime()) (no uniquifier), same q window as the default
    //   C5_SEEDMODE=lit:LO:HI  generator = new Random(L) for every long L in [LO, HI] (q column = L), e.g. currentTimeMillis
    const char *after = getenv("C5_AFTER"); const char *smode = getenv("C5_SEEDMODE");
    int raw = 0; long long LLO = 0, LHI = -1;
    if (smode && !strcmp(smode, "nano")) raw = 2;
    else if (smode && !strncmp(smode, "lit:", 4)) { raw = 1; if (sscanf(smode + 4, "%lld:%lld", &LLO, &LHI) != 2) return 2; }
    bool single = after || raw;
    std::vector<Seg> segs; long long total = 0;
    for (int ci = 0; ci < (int)CA.size(); ci++) for (int j = single ? 0 : 1; j <= (single ? 0 : J); j++) {
        long long lo = QLO >= 0 ? QLO : CA[ci].qm - (long long)(DMAX * FHZ), hi = QHI >= 0 ? QHI : CA[ci].qm - (long long)(DMIN * FHZ);
        long long kg = CA[ci].km - j;
        if (after) { kg = CA[ci].km + 1; lo = CA[ci].qm; hi = CA[ci].qm + (long long)(atof(after) * FHZ); }
        if (raw == 1) { lo = LLO; hi = LHI; kg = 1; }
        if (raw == 2) kg = 1;
        if (kg < 1) continue;
        if (hi < lo) continue;
        Seg s; s.qlo = lo; s.count = hi - lo + 1; s.start = total; s.ci = ci; s.j = j; s.kg = (int)kg; s.raw = raw;
        for (int m = 0; m < MAXMAP; m++) { s.fx[m] = 0; s.fz[m] = 0; }
        for (int m = 0; m < NM; m++) { s.fx[m] = (int)floor(mref[m] == 1 ? CA[ci].x1 : CA[ci].x2); s.fz[m] = (int)floor(mref[m] == 1 ? CA[ci].z1 : CA[ci].z2); }
        segs.push_back(s); total += s.count;
    }
    int nseg = (int)segs.size();
    float hcx[MAXC] = {0}, hcz[MAXC] = {0}; int hcm[MAXC] = {0}, hcmin[MAXMAP] = {0};
    for (int c = 0; c < NCOMP; c++) { hcx[c] = ccx[c]; hcz[c] = ccz[c]; hcm[c] = cmap[c]; }
    // Ranked mode (PC lane): env C5_TOPN=N keeps, per candidate row, the N best stage-1 survivors by the combined score
    // sum_m (hits_m / ncomp_m + iou_m), even below the PASS thresholds; stage 1 then runs at C5_RANK_CMIN (default 1 per
    // map) and PASS lines still apply cmin/imin. In this mode the SUMMARY s1/s2 counters refer to the rank cmin.
    const int TOPN = getenv("C5_TOPN") ? atoi(getenv("C5_TOPN")) : 0;
    std::vector<int> rcmin(NM, 1);
    if (TOPN && getenv("C5_RANK_CMIN")) { std::vector<std::string> rl; int nr = split_list(getenv("C5_RANK_CMIN"), rl); for (int m = 0; m < NM; m++) rcmin[m] = atoi(rl[nr == 1 ? 0 : m].c_str()); }
    for (int m = 0; m < NM; m++) hcmin[m] = TOPN ? std::min(rcmin[m], mcmin[m]) : mcmin[m];
    CK(cudaMemcpyToSymbol(c_cx, hcx, sizeof hcx)); CK(cudaMemcpyToSymbol(c_cz, hcz, sizeof hcz)); CK(cudaMemcpyToSymbol(c_cmap, hcm, sizeof hcm));
    CK(cudaMemcpyToSymbol(c_ncomp, &NCOMP, sizeof(int))); CK(cudaMemcpyToSymbol(c_nmap, &NM, sizeof(int)));
    CK(cudaMemcpyToSymbol(c_cminm, hcmin, sizeof hcmin));
    CK(cudaMemcpyToSymbol(c_nseg, &nseg, sizeof(int))); CK(cudaMemcpyToSymbol(c_fhz, &FHZ, sizeof(long long)));

    Seg *dseg; short2 *dobs; unsigned char *dlab; Rec *dq; unsigned int *dqn; int *dtp, *dfp, *dob, *doc; unsigned long long *ds1m;
    const long long CHUNK = 1LL << 26; const unsigned int QCAP = 1u << 24;
    int NO = (int)obs.size();
    CK(cudaMalloc(&dseg, sizeof(Seg) * (nseg ? nseg : 1))); CK(cudaMemcpy(dseg, segs.data(), sizeof(Seg) * nseg, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dobs, sizeof(short2) * (NO ? NO : 1))); CK(cudaMemcpy(dobs, obs.data(), sizeof(short2) * NO, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dlab, NO ? NO : 1)); CK(cudaMemcpy(dlab, olab.data(), NO, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dob, 4 * NM)); CK(cudaMemcpy(dob, obase.data(), 4 * NM, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&doc, 4 * NM)); CK(cudaMemcpy(doc, ocount.data(), 4 * NM, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&ds1m, 8 * MAXMAP)); CK(cudaMemset(ds1m, 0, 8 * MAXMAP));
    CK(cudaMalloc(&dq, sizeof(Rec) * QCAP)); CK(cudaMalloc(&dqn, 4));
    CK(cudaMalloc(&dtp, 4 * MAXMAP * (size_t)QCAP)); CK(cudaMalloc(&dfp, 4 * MAXMAP * (size_t)QCAP));

    struct Out { int ci, j; long long q; unsigned long long g0; int hits[MAXMAP]; double iou[MAXMAP]; };
    std::vector<Out> passes; unsigned long long P1 = 0, P2 = 0, OVF = 0, S2M[MAXMAP] = {0};
    struct Rk { double score; Out o; };
    auto rkcmp = [](const Rk &a, const Rk &b) { return a.score > b.score; };          // min-heap on score
    std::vector<std::vector<Rk>> heaps(CA.size());
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Rec> hq; std::vector<int> htp, hfp;
    for (long long base = 0; base < total; base += CHUNK) {
        long long n = std::min(CHUNK, total - base);
        CK(cudaMemset(dqn, 0, 4));
        stage1<<<(unsigned)((n + 255) / 256), 256>>>(dseg, base, n, dq, dqn, QCAP, ds1m);
        CK(cudaDeviceSynchronize());
        unsigned int qn; CK(cudaMemcpy(&qn, dqn, 4, cudaMemcpyDeviceToHost));
        if (qn > QCAP) { fprintf(stderr, "stage-1 queue overflow (%u > %u)\n", qn, QCAP); return 5; }
        P1 += qn;
        if (!qn) continue;
        stage2<<<qn, 128>>>(dseg, dq, dobs, dlab, dob, doc, dtp, dfp);
        CK(cudaDeviceSynchronize());
        hq.resize(qn); htp.resize((size_t)qn * MAXMAP); hfp.resize((size_t)qn * MAXMAP);
        CK(cudaMemcpy(hq.data(), dq, sizeof(Rec) * qn, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(htp.data(), dtp, 4 * MAXMAP * (size_t)qn, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hfp.data(), dfp, 4 * MAXMAP * (size_t)qn, cudaMemcpyDeviceToHost));
        for (unsigned int i = 0; i < qn; i++) {
            Out o; bool ok = true, ovf = false;
            if (TOPN) {                                         // ranked mode: all maps' IoU, pass test on thresholds
                double sc = 0; bool pass = true; const Seg &sg = segs[hq[i].seg];
                for (int m = 0; m < NM; m++) {
                    int tp = htp[(size_t)i * MAXMAP + m], fp = hfp[(size_t)i * MAXMAP + m];
                    if (tp < 0) { ovf = true; break; }
                    o.iou[m] = (mnopen[m] + fp) ? (double)tp / (mnopen[m] + fp) : 1.0; o.hits[m] = hq[i].hits[m];
                    sc += (mncomp[m] ? (double)o.hits[m] / mncomp[m] : 0) + o.iou[m];
                    if (o.hits[m] < mcmin[m] || o.iou[m] < mimin[m]) pass = false; else S2M[m]++;
                }
                if (ovf) { OVF++; continue; }
                o.ci = sg.ci; o.j = sg.j; o.q = hq[i].q; o.g0 = host_seed(sg.raw, hq[i].q, sg.kg, FHZ);
                if (pass) { passes.push_back(o); P2++; }
                auto &h = heaps[sg.ci];
                if ((int)h.size() < TOPN) { h.push_back({sc, o}); std::push_heap(h.begin(), h.end(), rkcmp); }
                else if (sc > h.front().score) { std::pop_heap(h.begin(), h.end(), rkcmp); h.back() = {sc, o}; std::push_heap(h.begin(), h.end(), rkcmp); }
                continue;
            }
            for (int m = 0; m < NM && ok; m++) {               // CPU: maps in order, early exit
                int tp = htp[(size_t)i * MAXMAP + m], fp = hfp[(size_t)i * MAXMAP + m];
                if (tp < 0) { ovf = true; ok = false; break; }
                double iou = (mnopen[m] + fp) ? (double)tp / (mnopen[m] + fp) : 1.0;
                o.iou[m] = iou; o.hits[m] = hq[i].hits[m];
                if (iou >= mimin[m]) S2M[m]++; else ok = false;
            }
            if (ovf) { OVF++; continue; }
            if (!ok) continue;
            const Seg &sg = segs[hq[i].seg];
            o.ci = sg.ci; o.j = sg.j; o.q = hq[i].q; o.g0 = host_seed(sg.raw, hq[i].q, sg.kg, FHZ);
            passes.push_back(o); P2++;
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    unsigned long long S1M[MAXMAP] = {0}; CK(cudaMemcpy(S1M, ds1m, 8 * MAXMAP, cudaMemcpyDeviceToHost));
    std::sort(passes.begin(), passes.end(), [](const Out &a, const Out &b) { return a.ci != b.ci ? a.ci < b.ci : a.j != b.j ? a.j < b.j : a.q < b.q; });
    for (auto &p : passes) {
        printf("PASS %d %d %lld %llu", p.ci, p.j, p.q, p.g0);
        for (int m = 0; m < NM; m++) printf(" %d %.4f", p.hits[m], p.iou[m]);
        printf("\n");
    }
    for (size_t ci = 0; ci < heaps.size(); ci++) {
        auto &h = heaps[ci]; std::sort(h.begin(), h.end(), [](const Rk &a, const Rk &b) { return a.score > b.score; });
        for (auto &r : h) {
            printf("RANK %d %d %lld %llu %.4f", r.o.ci, r.o.j, r.o.q, r.o.g0, r.score);
            for (int m = 0; m < NM; m++) printf(" %d %.4f", r.o.hits[m], r.o.iou[m]);
            putchar(10);
        }
    }
    printf("SUMMARY variant=%s cands=%d obs=%d comps=%d tested=%llu pass_stage1=%llu pass_stage2=%llu maps=%d", spec, (int)CA.size(), NO, NCOMP,
           (unsigned long long)total, P1, P2, NM);
    for (int m = 0; m < NM; m++) printf(" map%d:R%d,obs=%d,comps=%d,cmin=%d,imin=%.3f,s1=%llu,s2=%llu", m, mref[m], ocount[m], mncomp[m], mcmin[m], mimin[m], S1M[m], S2M[m]);
    printf("\n");
    fprintf(stderr, "GPU seconds=%.3f seeds_per_s=%.3e roi_overflow=%llu\n", secs, total / (secs > 0 ? secs : 1), OVF);
    return OVF ? 6 : 0;
}
