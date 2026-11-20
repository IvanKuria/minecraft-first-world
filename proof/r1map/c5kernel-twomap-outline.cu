// C5 CUDA kernel OUTLINE (not compiled or run here: no NVIDIA GPU on the Mac).
// Reference semantics = c5search.c (CPU, validated end-to-end against GEN's generator by validate_search.py).
// Any GPU build must reproduce c5search's PASS lines bit-for-bit on the planted-seed fixtures before a real run.
//
// One thread = one generator seed G0 (one QPC tick q_g of one candidate row r and one uniquifier offset j).
// Stage 1 is a pure streaming pass over the first 4K draws: no world, no seed storage.
//
//   t   = (long long)(__dmul_rn(__ddiv_rn((double)q, (double)f), 1e9))   // HotSpot6 Windows nanoTime, IEEE exact;
//                                                                         // compile with -fmad=false (no FMA contraction)
//   G0  = ((U + k_g + t) ^ 0x5DEECE66D) & (2^48-1)                       // JDK6 new Random()
//   for k in 0..Kmax-1:  x = nextInt(254)+1, y = nextInt(62)+1, z = nextInt(254)+1, ty = nextInt(nt)   (U*s/U*r order)
//       kept iff y <= 42; relevant iff |x - x2f| <= 62 and |z - z2f| <= 62 (observations lie within 32 of the spawn)
//       for each observed open-component centroid c (<= 16, constant memory):
//           d = (x+.5-cx)^2 + (y+.5-42.5)^2 + (z+.5-cz)^2 ; keep running (best d, raw type bits) per c,
//           snapshot the running best at K checkpoints {256,512,1024,2048,4096} (seed lists are prefixes)
//   at the end, for each (K checkpoint, air share nt in {2,3,4,6,8,12}):  hits = #c whose best seed's type == 0
//   pass1 iff hits >= cmin  -> atomicAdd a (r, j, q, subvariant) record to the stage-2 queue.
//
// Stage 2 kernel (one block per queued record, ~1e-3..5e-3 of seeds on the real map): re-draw the K seeds into shared
// memory (<= ~700 ROI seeds for K=4096 -> 700*4*4 B = 11 KB), threads stride over the ~1.4k observed columns, exact
// nearest kept seed (ties -> lowest draw index), block-reduce TP/FP/nOpen -> IoU; pass2 iff IoU >= imin -> host list.
// Host: dedupe, then full regrowth (gen run) + LINK frozen scorer on a CPU farm.
//
// TWO MAPS (R1MAP, Oct 15): centroids carry a map id (c_cmap[c]); each map is registered to its own reset of the row
// (map R2 -> x2f,z2f; map R1 -> x1f,z1f, both floors of that candidate's Java-float reset).  Each kept seed is tested
// against each map's +-62 ROI; stage 1 needs hits_m >= cmin_m on EVERY map; stage 2 computes one IoU per map.
// Reference semantics = c5search.c with obs 'footage.obs,../r1map/footage-r1.obs' (validated by
// r1map/validate_search2.py: two-map PASS set == intersection of the single-map PASS sets).  Untested on GPU.
//
// Cost per stage-1 thread (K=1024): 4096 LCG steps (48-bit mul+add as 64-bit: ~4 IMAD) + ~165 ROI seeds x 6 centroids
// x ~7 FP32 ops  ->  ~5e4 integer + ~7e3 FP ops  ->  ~2e8 seeds/s peak on an RTX 5070 Ti (~44 TFLOP/s FP32, INT32 on
// the same cores); plan on ~5e7..1e8 seeds/s after occupancy/branch losses. Kmax=4096 is 4x the LCG work.

#include <stdint.h>
#define MASK48 0xFFFFFFFFFFFFULL
#define MULT 0x5DEECE66DULL
#define ADDC 0xBULL
#define U0 8682522807148012ULL
#define MAXC 16   // total centroids over both maps (real: 6 + 6)
#define NKCP 5
#define NNT 6

__constant__ float  c_cx[MAXC], c_cz[MAXC];
__constant__ int    c_ncomp, c_kmax, c_cmin;   // c_cmin: map 0 (R2) threshold
__constant__ int    c_cmap[MAXC];            // centroid -> map (0 = R2, 1 = R1)
__constant__ int    c_cmin1;                 // map 1 (R1) threshold
__constant__ int    c_kcp[NKCP];          // {256,512,1024,2048,4096}
__constant__ int    c_nt[NNT];            // {2,3,4,6,8,12}

struct Row { long long q_lo, q_hi; int k_g; int x2f, z2f, x1f, z1f; };

__device__ __forceinline__ uint32_t next31(uint64_t &s) { s = (s * MULT + ADDC) & MASK48; return (uint32_t)(s >> 17); }
__device__ __forceinline__ int nextIntNP2(uint64_t &s, int n) {           // java.util.Random.nextInt(n), n not 2^k
    for (;;) { int bits = (int)next31(s); int val = bits % n; if (bits - val + (n - 1) >= 0) return val; }  // int32 overflow test
}

__global__ void stage1(const Row *rows, int nrows, long long per_row, unsigned long long *queue, unsigned int *qn, unsigned int qcap) {
    long long gid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int r = (int)(gid / per_row); if (r >= nrows) return;
    long long q = rows[r].q_lo + gid % per_row; if (q > rows[r].q_hi) return;
    long long t = (long long)__dmul_rn(__ddiv_rn((double)q, 3579545.0 /* f, per scenario */), 1e9);
    uint64_t s = ((U0 + (uint64_t)rows[r].k_g + (uint64_t)t) ^ MULT) & MASK48;
    float best[MAXC]; uint32_t braw[MAXC];
    float bcp[NKCP][MAXC]; uint32_t rcp[NKCP][MAXC];  // snapshots (registers/local; MAXC can be set to the real 6)
    for (int c = 0; c < MAXC; c++) { best[c] = 1e30f; braw[c] = 0; }
    int cp = 0;
    for (int k = 0; k < c_kmax; k++) {
        int x = nextIntNP2(s, 254) + 1, y = nextIntNP2(s, 62) + 1, z = nextIntNP2(s, 254) + 1;
        uint32_t raw = next31(s);   // type draw: nextInt(nt) is derived per nt below (rejection for nt=3,6,12 has
                                    // P ~ 1e-9 per call; a rejected draw shifts the stream -> the CPU stage re-checks exactly)
        for (int m = 0; m < 2; m++) {
            int ox = m ? rows[r].x1f : rows[r].x2f, oz = m ? rows[r].z1f : rows[r].z2f;
            if (!(y <= 42 && abs(x - ox) <= 62 && abs(z - oz) <= 62)) continue;
            float fx = x + .5f - ox, fy = y + .5f - 42.5f, fz = z + .5f - oz;
            for (int c = 0; c < c_ncomp; c++) {
                if (c_cmap[c] != m) continue;
                float dx = fx - c_cx[c], dz = fz - c_cz[c]; float d = dx * dx + fy * fy + dz * dz;
                if (d < best[c]) { best[c] = d; braw[c] = raw; }
            }
        }
        if (cp < NKCP && k + 1 == c_kcp[cp]) { for (int c = 0; c < c_ncomp; c++) { bcp[cp][c] = best[c]; rcp[cp][c] = braw[c]; } cp++; }
    }
    for (int i = 0; i < cp; i++) for (int n = 0; n < NNT; n++) {
        int nt = c_nt[n], hits[2] = {0, 0};
        for (int c = 0; c < c_ncomp; c++) {
            uint32_t raw = rcp[i][c];
            int v = ((nt & -nt) == nt) ? (int)(((uint64_t)nt * raw) >> 31) : (int)(raw % (uint32_t)nt);
            hits[c_cmap[c]] += (v == 0);
        }
        if (hits[0] >= c_cmin && hits[1] >= c_cmin1) {
            unsigned int slot = atomicAdd(qn, 1u);
            if (slot < qcap) queue[slot] = ((unsigned long long)r << 48) | ((unsigned long long)(q - rows[r].q_lo) << 8) | (i << 4) | n;
        }
    }
}
// NOTE 1: the outline encodes q offset in 40 bits (per-row windows <= 30 s * f <= 4.3e8 for HPET: fits).
// NOTE 2: the real file must parameterise f, the draw order (xyz / xzy / type-first) and the uniquifier offset j.
// NOTE 3: the U*r (random-neighbour) variants share the seed phase; only stage 3 (regrowth) differs.
