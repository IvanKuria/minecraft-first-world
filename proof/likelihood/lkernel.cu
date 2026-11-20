/* LIKELIHOOD lane: CUDA OUTLINE of the stage-1 likelihood screen (NOT compiled here: no NVIDIA GPU on the Mac).
 * Semantics = lscore.h ls_map_llr_fast (stage 1: no shift, 3x3 nodes, stage-1 column subset, search radius rmax1),
 * which `lscore rank` uses on the CPU.  VALIDATION REQUIRED before use: run `LS_DUMP1=1 lscore rank ... <half> ... 1 1`
 * (1 thread, small window) and this kernel on the same params/maps/window; every S1 value must agree to ~1e-3 (float
 * summation order differs), and the planted fixtures (fixtures/<key>/test/fx*.json via fx_rank.py maps) must give the
 * same stage-1 rank of the planted L.
 * Build (PC): nvcc -O3 -arch=sm_120 -fmad=false -o lkernel lkernel.cu
 * usage: lkernel <params.txt> <map1,map2> <lit|nano> <lo> <hi> <x1> <z1> <x2> <z2> <thr> [kg f_hz]
 *   lit : G0 = (L ^ 0x5DEECE66D) mod 2^48 for L in [lo, hi]
 *   nano: G0 = ((U + kg + (jlong)((double)q / f * 1e9)) ^ 0x5DEECE66D) mod 2^48 for q in [lo, hi]  (c5search H3)
 *   prints "S1 <L|q> <score>" for every seed with stage-1 score >= thr (atomic append), then a SUMMARY line.
 * Design: one thread per seed.  Seeds are generated with the exact java.util.Random path (power-of-two bounds for the
 * full-height Notch idiom need no modulo), culled to the ROI (union of the two map boxes +- rmax1, y >= ytop-7-rmax1;
 * ~50 seeds for K=1024, ~200 for K=4096), bucketed into a per-thread 8x8-column grid over the ROI, then every stage-1
 * column (~600-900 for the two maps) gets the 8-depth nearest air/rock distances and the 3x3-node column model.
 * Cost ~= 4K LCG steps + n_col * (~25 seed visits * 8 + 72) flops; CPU reference 0.65-0.9 ms/seed/thread on an M2.
 */
#define LSCORE_NO_MAIN
#include "lscore.c"
#include <cuda_runtime.h>

#define ROI_MAX 1024
#define GX 24   /* ROI bucket grid (8x8 columns each): 24 x 24 buckets = 192 x 192 columns max */
__constant__ LParams cP;
__constant__ int cNM, cRef[4], cFx[4], cFz[4], cN1[4];
__constant__ float cConf[4][LS_MAXCLS][LS_L + 1], cLnull[4][LS_MAXCLS];
__constant__ int cRoiX0, cRoiZ0, cRoiX1, cRoiZ1;
__device__ int16_t gDx[4][LS_MAXOBS], gDz[4][LS_MAXOBS]; __device__ uint8_t gC[4][LS_MAXOBS];

__device__ static float col_fast(const LSeed *rs, const int *bst, int X, int Z, const float *conf) {
    const LParams *P = &cP; const int RR = P->rmax1, R2 = RR * RR, RC = LS_RMAX * LS_RMAX;
    int da2[LS_L], dr2[LS_L];
    for (int i = 0; i < LS_L; i++) { da2[i] = RC; dr2[i] = RC; }
    if (X < 0 || Z < 0 || X >= LS_W || Z >= LS_H) return conf[LS_L];      /* out of world: P(deep) = 1 */
    int bx0 = (X - RR - cRoiX0) >> 3, bx1 = (X + RR - cRoiX0) >> 3, bz0 = (Z - RR - cRoiZ0) >> 3, bz1 = (Z + RR - cRoiZ0) >> 3;
    if (bx0 < 0) bx0 = 0; if (bz0 < 0) bz0 = 0; if (bx1 >= GX) bx1 = GX - 1; if (bz1 >= GX) bz1 = GX - 1;
    for (int bz = bz0; bz <= bz1; bz++) for (int bx = bx0; bx <= bx1; bx++)
        for (int q = bst[bz * GX + bx]; q < bst[bz * GX + bx + 1]; q++) {
            LSeed s = rs[q]; int dx = s.x - X, dz = s.z - Z, dh = dx * dx + dz * dz; if (dh >= R2) continue;
            int *dd = s.t ? dr2 : da2;
            for (int i = 0; i < LS_L; i++) { int dy = s.y - (P->ytop - i), d2 = dh + dy * dy; if (d2 < dd[i]) dd[i] = d2; }
        }
    int dsh = X; if (LS_W - 1 - X < dsh) dsh = LS_W - 1 - X; if (Z < dsh) dsh = Z; if (LS_H - 1 - Z < dsh) dsh = LS_H - 1 - Z;
    float mr[LS_L], ma[LS_L], E[LS_L], pd[LS_L + 1];
    for (int i = 0; i < LS_L; i++) { float r = P->gtab[dr2[i]], sh = P->gsh[dsh]; mr[i] = r < sh ? r : sh; ma[i] = P->gtab[da2[i]]; E[i] = expf(-P->lam * (P->T - mr[i])); }
    for (int k = 0; k <= LS_L; k++) pd[k] = 0.f;
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        float w = P->uw3[a] * P->vw3[b], cum = 1.f;
        for (int i = 0; i < LS_L; i++) { float pf = 1.f - E[i] * P->eu3[a]; if (pf < 0.f) pf = 0.f; float ps = (mr[i] - ma[i] < P->vv3[b]) ? pf : 0.f; pd[i] += w * cum * ps; cum *= 1.f - ps; }
        pd[LS_L] += w * cum;
    }
    float q = 0; for (int k = 0; k <= LS_L; k++) q += conf[k] * pd[k];
    return q;
}
__global__ void screen(long long lo, long long hi, int nano, long long kg, long long fhz, float thr, long long *outL, float *outS, unsigned *nout, unsigned cap) {
    long long idx = lo + (long long)blockIdx.x * blockDim.x + threadIdx.x; if (idx > hi) return;
    uint64_t G0;
    if (!nano) G0 = ((uint64_t)idx ^ JR_MULT) & JR_MASK;
    else { long long t = (long long)(((double)idx / (double)fhz) * 1000000000.0); G0 = ((uint64_t)(8682522807148012LL + kg + t) ^ JR_MULT) & JR_MASK; }
    const LParams *P = &cP;
    LSeed rs[ROI_MAX]; int cnt[GX * GX + 1], bst[GX * GX + 1], n = 0;
    for (int b = 0; b <= GX * GX; b++) cnt[b] = 0;
    /* seeds: exact lgen draw order; ROI cull */
    JRand r = {G0, 0, 0}; const int m = P->margin, ymin = P->ytop - (LS_L - 1) - P->rmax1;
    LSeed tmp[ROI_MAX]; int nt_ = 0;
    for (int k = 0; k < P->K; k++) {
        int t = 1; if (P->tf) t = jr_nextInt(&r, P->nt) == 0 ? 0 : 1;
        int x = jr_nextInt(&r, LS_W - 2 * m) + m, y = jr_nextInt(&r, LS_D - 2 * m) + m, z = jr_nextInt(&r, LS_H - 2 * m) + m;
        if (!P->tf) t = jr_nextInt(&r, P->nt) == 0 ? 0 : 1;
        if (y < ymin || y > P->ytop || x < cRoiX0 || x > cRoiX1 || z < cRoiZ0 || z > cRoiZ1 || nt_ >= ROI_MAX) continue;
        tmp[nt_].x = x; tmp[nt_].y = y; tmp[nt_].z = z; tmp[nt_].t = t; nt_++;
        cnt[((z - cRoiZ0) >> 3) * GX + ((x - cRoiX0) >> 3)]++;
    }
    bst[0] = 0; for (int b = 0; b < GX * GX; b++) bst[b + 1] = bst[b] + cnt[b];
    for (int b = 0; b < GX * GX; b++) cnt[b] = 0;
    for (int i = 0; i < nt_; i++) { int b = ((tmp[i].z - cRoiZ0) >> 3) * GX + ((tmp[i].x - cRoiX0) >> 3); rs[bst[b] + cnt[b]++] = tmp[i]; }
    n = nt_; (void)n;
    float ll = 0;
    for (int mp = 0; mp < cNM; mp++) for (int j = 0; j < cN1[mp]; j++) {
        int c = gC[mp][j];
        float q = col_fast(rs, bst, cFx[mp] + gDx[mp][j], cFz[mp] + gDz[mp][j], cConf[mp][c]);
        ll += logf(q > 1e-30f ? q : 1e-30f) - cLnull[mp][c];
    }
    if (ll >= thr) { unsigned o = atomicAdd(nout, 1u); if (o < cap) { outL[o] = idx; outS[o] = ll; } }
}
/* host: load params/maps with lscore.c's loaders, keep only stage-1 columns (ls_flag_stage1, LS_STRIDE), set the ROI as the
 * union of the map boxes (+- rmax1) in world coordinates from (x1,z1,x2,z2), copy to constant/device memory, launch over
 * [lo,hi] in chunks of 2^24, print hits.  ROI must fit GX*8 = 192 columns per axis (true for the footage maps). */
int main(int argc, char **argv) {
    if (argc < 11) { fprintf(stderr, "see header\n"); return 2; }
    if (load_params(argv[1], &PRM)) return 2;
    char ml[4096]; strncpy(ml, argv[2], 4095); if (load_maps(ml)) return 2;
    int stride = getenv("LS_STRIDE") ? atoi(getenv("LS_STRIDE")) : 4;
    int nano = !strcmp(argv[3], "nano"); long long lo = atoll(argv[4]), hi = atoll(argv[5]);
    double pos[4] = {atof(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9])}; float thr = atof(argv[10]);
    long long kg = argc > 12 ? atoll(argv[11]) : 0, fhz = argc > 12 ? atoll(argv[12]) : 1;
    int ref[4], fx[4], fz[4], n1[4], rx0 = 1 << 30, rz0 = 1 << 30, rx1 = -(1 << 30), rz1 = -(1 << 30);
    static int16_t dx[4][LS_MAXOBS], dz[4][LS_MAXOBS]; static uint8_t cc[4][LS_MAXOBS]; static float conf[4][LS_MAXCLS][LS_L + 1], lnull[4][LS_MAXCLS];
    for (int m = 0; m < NM; m++) {
        ls_flag_stage1(&MP[m], 14, stride); ref[m] = MP[m].ref;
        fx[m] = (int)floor(MP[m].ref == 1 ? pos[0] : pos[2]); fz[m] = (int)floor(MP[m].ref == 1 ? pos[1] : pos[3]);
        n1[m] = 0; for (int j = 0; j < MP[m].n; j++) if (MP[m].s1[j]) { dx[m][n1[m]] = MP[m].dx[j]; dz[m][n1[m]] = MP[m].dz[j]; cc[m][n1[m]] = MP[m].c[j]; n1[m]++;
            int X = fx[m] + MP[m].dx[j], Z = fz[m] + MP[m].dz[j]; if (X - PRM.rmax1 < rx0) rx0 = X - PRM.rmax1; if (X + PRM.rmax1 > rx1) rx1 = X + PRM.rmax1; if (Z - PRM.rmax1 < rz0) rz0 = Z - PRM.rmax1; if (Z + PRM.rmax1 > rz1) rz1 = Z + PRM.rmax1; }
        memcpy(conf[m], MP[m].conf, sizeof conf[m]); memcpy(lnull[m], MP[m].lnull, sizeof lnull[m]);
    }
    if (rx1 - rx0 >= GX * 8 || rz1 - rz0 >= GX * 8) { fprintf(stderr, "ROI too large for GX\n"); return 2; }
    cudaMemcpyToSymbol(cP, &PRM, sizeof PRM); cudaMemcpyToSymbol(cNM, &NM, sizeof NM); cudaMemcpyToSymbol(cRef, ref, sizeof ref);
    cudaMemcpyToSymbol(cFx, fx, sizeof fx); cudaMemcpyToSymbol(cFz, fz, sizeof fz); cudaMemcpyToSymbol(cN1, n1, sizeof n1);
    cudaMemcpyToSymbol(cConf, conf, sizeof conf); cudaMemcpyToSymbol(cLnull, lnull, sizeof lnull);
    cudaMemcpyToSymbol(cRoiX0, &rx0, sizeof rx0); cudaMemcpyToSymbol(cRoiZ0, &rz0, sizeof rz0); cudaMemcpyToSymbol(cRoiX1, &rx1, sizeof rx1); cudaMemcpyToSymbol(cRoiZ1, &rz1, sizeof rz1);
    cudaMemcpyToSymbol(gDx, dx, sizeof dx); cudaMemcpyToSymbol(gDz, dz, sizeof dz); cudaMemcpyToSymbol(gC, cc, sizeof cc);
    const unsigned cap = 1u << 22; long long *dL; float *dS; unsigned *dn; cudaMalloc(&dL, 8ull * cap); cudaMalloc(&dS, 4ull * cap); cudaMalloc(&dn, 4); cudaMemset(dn, 0, 4);
    for (long long a = lo; a <= hi; a += (1LL << 24)) { long long b = a + (1LL << 24) - 1; if (b > hi) b = hi; long long nn = b - a + 1;
        screen<<<(unsigned)((nn + 127) / 128), 128>>>(a, b, nano, kg, fhz, thr, dL, dS, dn, cap); cudaDeviceSynchronize(); }
    unsigned nout; cudaMemcpy(&nout, dn, 4, cudaMemcpyDeviceToHost); if (nout > cap) nout = cap;
    long long *hL = (long long *)malloc(8ull * nout); float *hS = (float *)malloc(4ull * nout);
    cudaMemcpy(hL, dL, 8ull * nout, cudaMemcpyDeviceToHost); cudaMemcpy(hS, dS, 4ull * nout, cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < nout; i++) printf("S1 %lld %.6f\n", hL[i], hS[i]);
    printf("SUMMARY tested=%lld hits=%u thr=%.3f\n", hi - lo + 1, nout, thr);
    return 0;
}
