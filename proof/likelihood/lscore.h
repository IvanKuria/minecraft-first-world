/* LIKELIHOOD lane: probabilistic surface likelihood of a generator seed list (seed positions/types + T + parameters only).
 * HYPOTHESIS FAMILY ONLY: the full-height shell model (lgen FH*), not recovered Minecraft source.
 *
 * Model M1 (fitted by calib2.py on lgen ground truth; FPP picture: the scan rule is site first-passage percolation with
 * Exp(1) site waits, the random-neighbour rule is edge FPP with Exp(1/6) edge waits):
 *   cell (X, ytop-i, Z), i = 0..L-1:   dr = min(nearest rock seed, kap * shell distance), da = nearest air seed
 *   m_r = g(dr), m_a = g(da), g(d) = a + b d^p
 *   rock wins iff m_r - m_a < v ;  solid iff rock wins and m_r + u + tau_i < T,  tau_i ~ Exp(lam) independent
 *   u ~ N(0, su^2) and v ~ N(0, sv^2) are shared by the column (5x5 Gauss-Hermite nodes)
 *   D = number of non-solid cells from the top (0..L-1, L = deep)
 * Observation: each observed column has a class c; P(c | D) is a measured confusion table (lobs synthetic truth).
 * Map log-likelihood = log sum_shift w(shift) prod_cols sum_D P(c|D) P_model(D | column + shift).
 *
 * GPU notes: everything below is plain C on fixed-size arrays; distances are between integer cell centres so g() is a
 * table over d^2 <= RMAX^2; one expf per cell; no allocation, recursion or library calls other than expf/logf.
 */
#ifndef LSCORE_H
#define LSCORE_H
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "jrand.h"
#ifdef __CUDACC__
#define LS_FN __host__ __device__ static inline
#else
#define LS_FN static inline
#endif

#define LS_W 256
#define LS_H 256
#define LS_D 64
#define LS_L 8              /* depths 0..7, class 8 = deep */
#define LS_NQ 5             /* Gauss-Hermite nodes per noise */
#define LS_RMAX 24          /* distances are capped at RMAX (beyond it g() is far above any fitted T) */
#define LS_MAXK 16384
#define LS_NB 32            /* 32 x 32 column buckets of 8 x 8 */
#define LS_MAXCLS 32
#define LS_MAXOBS 4096
#define LS_MAXSHIFT 2

typedef struct { int16_t x, y, z; uint8_t t; } LSeed;
typedef struct {
    float a, b, p, su, sv, lam, kap, T;
    int ytop, margin, K, nt, tf;          /* seed draw: x=nextInt(W-2m)+m, y=nextInt(D-2m)+m, z=..., type nextInt(nt)==0 -> air */
    float gtab[LS_RMAX * LS_RMAX + 1];   /* g(sqrt(d2)) */
    float gsh[LS_W];                      /* g(kap * d_shell) */
    float eu[LS_NQ], uw[LS_NQ], vv[LS_NQ], vw[LS_NQ];
    float eu3[3], uw3[3], vv3[3], vw3[3];   /* stage-1 (fast) 3x3 nodes */
    int rmax1;                              /* stage-1 search radius (distances beyond it are treated as RMAX) */
} LParams;

static const double LS_GHX[LS_NQ] = {-2.8569700138728056, -1.3556261799742657, 0.0, 1.3556261799742657, 2.8569700138728056};
static const double LS_GHW[LS_NQ] = {0.011257411327720691, 0.22207592200561266, 0.5333333333333333, 0.22207592200561266, 0.011257411327720691};

LS_FN void ls_prepare(LParams *P) {
    if (P->rmax1 <= 0 || P->rmax1 > LS_RMAX) P->rmax1 = 16;
    for (int d2 = 0; d2 <= LS_RMAX * LS_RMAX; d2++) { double d = sqrt((double)d2); if (d < 0.5) d = 0.5; P->gtab[d2] = (float)(P->a + P->b * pow(d, P->p)); }
    for (int s = 0; s < LS_W; s++) { double d = P->kap * s; if (d < 0.5) d = 0.5; if (d > LS_RMAX) d = LS_RMAX; P->gsh[s] = (float)(P->a + P->b * pow(d, P->p)); }
    for (int q = 0; q < LS_NQ; q++) { P->eu[q] = (float)exp(P->lam * P->su * LS_GHX[q]); P->uw[q] = (float)LS_GHW[q]; P->vv[q] = (float)(P->sv * LS_GHX[q]); P->vw[q] = (float)LS_GHW[q]; }
    const double x3[3] = {-1.7320508075688772, 0.0, 1.7320508075688772}, w3[3] = {1.0 / 6, 2.0 / 3, 1.0 / 6};
    for (int q = 0; q < 3; q++) { P->eu3[q] = (float)exp(P->lam * P->su * x3[q]); P->uw3[q] = (float)w3[q]; P->vv3[q] = (float)(P->sv * x3[q]); P->vw3[q] = (float)w3[q]; }
}

/* bit-exact lgen seed phase for pick=xyz, rng=int, typed seeds (ntypes); FH1 never discards (y <= 62) */
LS_FN int ls_seeds(const LParams *P, uint64_t state, LSeed *S) {
    JRand r = {state & JR_MASK, 0, 0};
    const int m = P->margin;
    for (int k = 0; k < P->K; k++) {
        int x, y, z, t = 1;
        if (P->tf && P->nt > 1) { t = jr_nextInt(&r, P->nt) == 0 ? 0 : 1; }
        x = jr_nextInt(&r, LS_W - 2 * m) + m; y = jr_nextInt(&r, LS_D - 2 * m) + m; z = jr_nextInt(&r, LS_H - 2 * m) + m;
        if (!P->tf && P->nt > 1) t = jr_nextInt(&r, P->nt) == 0 ? 0 : 1;
        S[k].x = (int16_t)x; S[k].y = (int16_t)y; S[k].z = (int16_t)z; S[k].t = (uint8_t)t;
    }
    return P->K;
}

/* bucket the seeds that can be within RMAX of the top L layers */
typedef struct { int start[LS_NB * LS_NB + 1]; LSeed s[LS_MAXK]; int n; } LBuckets;
LS_FN void ls_bucket(const LParams *P, const LSeed *S, int K, LBuckets *B) {
    int cnt[LS_NB * LS_NB]; memset(cnt, 0, sizeof cnt);
    const int ymin = P->ytop - (LS_L - 1) - LS_RMAX;
    for (int k = 0; k < K; k++) if (S[k].y >= ymin && S[k].y <= P->ytop) cnt[(S[k].z >> 3) * LS_NB + (S[k].x >> 3)]++;
    B->start[0] = 0; for (int b = 0; b < LS_NB * LS_NB; b++) B->start[b + 1] = B->start[b] + cnt[b];
    memset(cnt, 0, sizeof cnt);
    for (int k = 0; k < K; k++) if (S[k].y >= ymin && S[k].y <= P->ytop) { int b = (S[k].z >> 3) * LS_NB + (S[k].x >> 3); B->s[B->start[b] + cnt[b]++] = S[k]; }
    B->n = B->start[LS_NB * LS_NB];
}

/* P(D = 0..L-1, deep) for world column (X, Z); out-of-world columns get a fixed 'deep' distribution */
LS_FN void ls_column_q(const LParams *P, const LBuckets *B, int X, int Z, float *pd, int fast) {
    if (X < 0 || Z < 0 || X >= LS_W || Z >= LS_H) { for (int k = 0; k < LS_L; k++) pd[k] = 0.f; pd[LS_L] = 1.f; return; }
    const int RR = fast ? P->rmax1 : LS_RMAX;
    int da2[LS_L], dr2[LS_L]; const int R2 = RR * RR, RC = LS_RMAX * LS_RMAX;
    for (int i = 0; i < LS_L; i++) { da2[i] = RC; dr2[i] = RC; }
    int bx0 = (X - RR) >> 3, bx1 = (X + RR) >> 3, bz0 = (Z - RR) >> 3, bz1 = (Z + RR) >> 3;
    if (bx0 < 0) bx0 = 0; if (bz0 < 0) bz0 = 0; if (bx1 >= LS_NB) bx1 = LS_NB - 1; if (bz1 >= LS_NB) bz1 = LS_NB - 1;
    for (int bz = bz0; bz <= bz1; bz++) for (int bx = bx0; bx <= bx1; bx++) {
        for (int q = B->start[bz * LS_NB + bx]; q < B->start[bz * LS_NB + bx + 1]; q++) {
            const LSeed s = B->s[q]; int dx = s.x - X, dz = s.z - Z, dh = dx * dx + dz * dz; if (dh >= R2) continue;
            int *dd = s.t ? dr2 : da2;
            for (int i = 0; i < LS_L; i++) { int dy = s.y - (P->ytop - i); int d2 = dh + dy * dy; if (d2 < dd[i]) dd[i] = d2; }
        }
    }
    int dsh = X; if (LS_W - 1 - X < dsh) dsh = LS_W - 1 - X; if (Z < dsh) dsh = Z; if (LS_H - 1 - Z < dsh) dsh = LS_H - 1 - Z;
    float mr[LS_L], ma[LS_L], E[LS_L];
    for (int i = 0; i < LS_L; i++) {
        float r = P->gtab[dr2[i]], s = P->gsh[dsh]; mr[i] = r < s ? r : s; ma[i] = P->gtab[da2[i]];
        E[i] = expf(-P->lam * (P->T - mr[i]));
    }
    for (int k = 0; k <= LS_L; k++) pd[k] = 0.f;
    const int nq = fast ? 3 : LS_NQ; const float *eu = fast ? P->eu3 : P->eu, *uw = fast ? P->uw3 : P->uw, *vv = fast ? P->vv3 : P->vv, *vw = fast ? P->vw3 : P->vw;
    for (int a = 0; a < nq; a++) for (int b = 0; b < nq; b++) {
        float w = uw[a] * vw[b], cum = 1.f;
        for (int i = 0; i < LS_L; i++) {
            float pf = 1.f - E[i] * eu[a]; if (pf < 0.f) pf = 0.f;
            float ps = (mr[i] - ma[i] < vv[b]) ? pf : 0.f;
            pd[i] += w * cum * ps; cum *= 1.f - ps;
        }
        pd[LS_L] += w * cum;
    }
}
LS_FN void ls_column(const LParams *P, const LBuckets *B, int X, int Z, float *pd) { ls_column_q(P, B, X, Z, pd, 0); }

/* one observed map registered to a reset: columns (dx, dz, class) relative to floor(reset) */
typedef struct { int n, ref, ncls; int16_t dx[LS_MAXOBS], dz[LS_MAXOBS]; uint8_t c[LS_MAXOBS], s1[LS_MAXOBS]; float conf[LS_MAXCLS][LS_L + 1]; float lnull[LS_MAXCLS]; } LMap;

/* log-likelihood ratio of the map vs the null (per-class null = sum_D conf * P0(D)), marginalised over integer shifts
 * |sx|,|sz| <= S with Gaussian prior sigma; work buffer pd must hold (2S+1)^2 ... computed on the fly per (column, shift). */
LS_FN double ls_map_llr(const LParams *P, const LBuckets *B, const LMap *M, int fx, int fz, int S, float sigma, double *best_shift_llr) {
    double ll[(2 * LS_MAXSHIFT + 1) * (2 * LS_MAXSHIFT + 1)]; int ns = 0; double wsum = 0, lw[(2 * LS_MAXSHIFT + 1) * (2 * LS_MAXSHIFT + 1)];
    for (int sz = -S; sz <= S; sz++) for (int sx = -S; sx <= S; sx++) { lw[ns] = -(sx * sx + sz * sz) / (2.0 * sigma * sigma); wsum += exp(lw[ns]); ll[ns++] = 0; }
    for (int j = 0; j < M->n; j++) {
        const float *cf = M->conf[M->c[j]]; const double l0 = M->lnull[M->c[j]];
        int s = 0;
        for (int sz = -S; sz <= S; sz++) for (int sx = -S; sx <= S; sx++, s++) {
            float pd[LS_L + 1]; ls_column(P, B, fx + M->dx[j] + sx, fz + M->dz[j] + sz, pd);
            double q = 0; for (int k = 0; k <= LS_L; k++) q += cf[k] * pd[k];
            ll[s] += log(q > 1e-30 ? q : 1e-30) - l0;
        }
    }
    double mx = -1e300; for (int s = 0; s < ns; s++) if (ll[s] + lw[s] > mx) mx = ll[s] + lw[s];
    double acc = 0; for (int s = 0; s < ns; s++) acc += exp(ll[s] + lw[s] - mx);
    if (best_shift_llr) { double b = -1e300; for (int s = 0; s < ns; s++) if (ll[s] > b) b = ll[s]; *best_shift_llr = b; }
    return mx + log(acc / wsum);
}

/* CPU variant with a per-map column cache (each distinct world column evaluated once for all shifts).
 * box: map bounding box (relative) dilated by S; pdbuf >= nx*nz*(L+1) floats, done >= nx*nz bytes. */
LS_FN void ls_map_bbox(const LMap *M, int S, int *x0, int *z0, int *nx, int *nz) {
    int a = 1 << 30, b = -(1 << 30), c = 1 << 30, d = -(1 << 30);
    for (int j = 0; j < M->n; j++) { if (M->dx[j] < a) a = M->dx[j]; if (M->dx[j] > b) b = M->dx[j]; if (M->dz[j] < c) c = M->dz[j]; if (M->dz[j] > d) d = M->dz[j]; }
    *x0 = a - S; *z0 = c - S; *nx = b - a + 1 + 2 * S; *nz = d - c + 1 + 2 * S;
}
LS_FN double ls_map_llr_cached(const LParams *P, const LBuckets *B, const LMap *M, int fx, int fz, int S, float sigma,
                                float *pdbuf, uint8_t *done, int x0, int z0, int nx, int nz, double *best_shift_llr) {
    double ll[(2 * LS_MAXSHIFT + 1) * (2 * LS_MAXSHIFT + 1)], lw[(2 * LS_MAXSHIFT + 1) * (2 * LS_MAXSHIFT + 1)]; int ns = 0; double wsum = 0;
    for (int sz = -S; sz <= S; sz++) for (int sx = -S; sx <= S; sx++) { lw[ns] = -(sx * sx + sz * sz) / (2.0 * sigma * sigma); wsum += exp(lw[ns]); ll[ns++] = 0; }
    memset(done, 0, (size_t)nx * nz);
    for (int j = 0; j < M->n; j++) {
        const float *cf = M->conf[M->c[j]]; const double l0 = M->lnull[M->c[j]];
        int s = 0;
        for (int sz = -S; sz <= S; sz++) for (int sx = -S; sx <= S; sx++, s++) {
            int bx = M->dx[j] + sx - x0, bz = M->dz[j] + sz - z0, bi = bz * nx + bx;
            float *pd = pdbuf + (size_t)bi * (LS_L + 1);
            if (!done[bi]) { ls_column(P, B, fx + M->dx[j] + sx, fz + M->dz[j] + sz, pd); done[bi] = 1; }
            double q = 0; for (int k = 0; k <= LS_L; k++) q += cf[k] * pd[k];
            ll[s] += log(q > 1e-30 ? q : 1e-30) - l0;
        }
    }
    double mx = -1e300; for (int s = 0; s < ns; s++) if (ll[s] + lw[s] > mx) mx = ll[s] + lw[s];
    double acc = 0; for (int s = 0; s < ns; s++) acc += exp(ll[s] + lw[s] - mx);
    if (best_shift_llr) { double b = -1e300; for (int s = 0; s < ns; s++) if (ll[s] > b) b = ll[s]; *best_shift_llr = b; }
    return mx + log(acc / wsum);
}

/* STAGE 1 (screen): no shift, 3x3 nodes, only the flagged column subset (all near-field sweep/U columns plus a fixed
 * 1-in-STRIDE sample of the far map columns; set by ls_flag_stage1).  Same model, same tables. */
LS_FN void ls_flag_stage1(LMap *M, int nearmax_cls, int stride) {
    for (int j = 0; j < M->n; j++) { unsigned h = (unsigned)(M->dx[j] * 73856093) ^ (unsigned)(M->dz[j] * 19349663); M->s1[j] = M->c[j] <= nearmax_cls || (h % (unsigned)stride) == 0; }
}
LS_FN double ls_map_llr_fast(const LParams *P, const LBuckets *B, const LMap *M, int fx, int fz) {
    double ll = 0;
    for (int j = 0; j < M->n; j++) {
        if (!M->s1[j]) continue;
        float pd[LS_L + 1]; ls_column_q(P, B, fx + M->dx[j], fz + M->dz[j], pd, 1);
        const float *cf = M->conf[M->c[j]]; float q = 0; for (int k = 0; k <= LS_L; k++) q += cf[k] * pd[k];
        ll += logf(q > 1e-30f ? q : 1e-30f) - M->lnull[M->c[j]];
    }
    return ll;
}
#endif
