/* Host helper for the CUDA C5 search: parses a GEN variant exactly as c5search does (vload from the unmodified
 * GEN/C5 sources) and exposes the reference seed list for self-tests. Compiled as C with MSVC, linked into c5gpu. */
#define C5_NO_MAIN
#include "../c5/c5sim.c"

int c5_variant(const char *spec, long long *K, int *pick, int *margin, int *ntypes, int *seedtype, int *rng, int *cap, int *capy) {
    Variant v;
    if (vload(&v, spec)) return -1;
    *K = v.K; *pick = v.pick; *margin = v.margin; *ntypes = v.ntypes; *seedtype = v.seedtype; *rng = v.rng; *cap = v.cap;
    *capy = v.capy >= 0 ? v.capy : DD * 2 / 3;
    return 0;
}

/* reference seed list (x,y,z,t,kept per seed) for GPU self-test */
int c5_seeds(const char *spec, unsigned long long s0, short *x, short *y, short *z, unsigned char *t, unsigned char *kept) {
    static Seed S[MAXK];
    Variant v;
    if (vload(&v, spec)) return -1;
    int K = seeds_from_state(&v, s0, S, NULL);
    for (int k = 0; k < K; k++) { x[k] = S[k].x; y[k] = S[k].y; z[k] = S[k].z; t[k] = S[k].t; kept[k] = S[k].kept; }
    return K;
}
