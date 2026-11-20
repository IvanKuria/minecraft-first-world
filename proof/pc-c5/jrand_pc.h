/* PC copy of likelihood/jrand.h: functions marked host+device under nvcc (annotation only). */
#ifdef __CUDACC__
#define JR_HD __host__ __device__
#else
#define JR_HD
#endif
/* Bit-exact java.util.Random (JDK 6 = JDK 8 algorithm) operating on the INTERNAL 48-bit state
 * (i.e. the value of Random.seed AFTER the setSeed scramble).  Every next(bits) = one LCG step.
 * Validated against Zulu JDK 8 in proof/gen/validate/ (see REPORT.md). */
#ifndef JRAND_H
#define JRAND_H
#include <stdint.h>
#define JR_MULT 0x5DEECE66DULL
#define JR_ADD  0xBULL
#define JR_MASK ((1ULL << 48) - 1)
#define JR_INV  0xDFE05BCB1365ULL /* JR_MULT^-1 mod 2^48 */

typedef struct { uint64_t s; uint64_t steps; uint64_t rejections; } JRand;

JR_HD static inline int32_t jr_next(JRand *r, int bits) {
    r->s = (r->s * JR_MULT + JR_ADD) & JR_MASK;
    r->steps++;
    return (int32_t)(r->s >> (48 - bits));
}
/* Random.nextInt(int n) */
JR_HD static inline int32_t jr_nextInt(JRand *r, int32_t n) {
    if ((n & -n) == n) return (int32_t)(((int64_t)n * (int64_t)jr_next(r, 31)) >> 31);
    int32_t bits, val;
    for (;;) {
        bits = jr_next(r, 31);
        val = bits % n;
        if ((int32_t)((uint32_t)bits - (uint32_t)val + (uint32_t)(n - 1)) >= 0) return val;
        r->rejections++;
    }
}
/* Random.nextDouble(): ((long)next(26) << 27) + next(27)) * 2^-53 ; Math.random() is this on a global Random */
JR_HD static inline double jr_nextDouble(JRand *r) {
    int64_t hi = jr_next(r, 26);
    int64_t lo = jr_next(r, 27);
    return (double)((hi << 27) + lo) * (1.0 / (double)(1ULL << 53));
}
JR_HD static inline float jr_nextFloat(JRand *r) { return jr_next(r, 24) / (float)(1 << 24); }

/* affine jump: state after n steps forward (n may be "negative" via rewind) */
JR_HD static inline uint64_t jr_advance(uint64_t s, uint64_t n) {
    uint64_t a = JR_MULT, c = JR_ADD, A = 1, C = 0; /* accumulate x -> A x + C */
    while (n) {
        if (n & 1) { A = (A * a) & JR_MASK; C = (C * a + c) & JR_MASK; }
        c = (c * (a + 1)) & JR_MASK; a = (a * a) & JR_MASK; n >>= 1;
    }
    return (A * s + C) & JR_MASK;
}
JR_HD static inline uint64_t jr_rewind(uint64_t s, uint64_t n) {
    /* inverse step: s_prev = (s - ADD) * INV */
    uint64_t a = JR_INV, c = ((JR_MASK + 1 - JR_ADD) * JR_INV) & JR_MASK, A = 1, C = 0;
    while (n) {
        if (n & 1) { A = (A * a) & JR_MASK; C = (C * a + c) & JR_MASK; }
        c = (c * (a + 1)) & JR_MASK; a = (a * a) & JR_MASK; n >>= 1;
    }
    return (A * s + C) & JR_MASK;
}
JR_HD static inline uint64_t jr_scramble(int64_t seed) { return ((uint64_t)seed ^ JR_MULT) & JR_MASK; }
#endif
