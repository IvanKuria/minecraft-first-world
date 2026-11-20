/* fpscan: JDK6 "new Random()" seed-fingerprint scanner.
 *
 * Hypothesis under test (H): the JVM-global Random behind Math.random() was
 * created by JDK6 `new Random()`:
 *     seed64 = (8682522807148012 + k) + System.nanoTime()        (k = 1,2,...)
 *     X0     = (seed64 ^ 0x5DEECE66D) & (2^48-1)
 * and Windows HotSpot 6 nanoTime = (jlong)(((double)qpc/(double)f) * 1e9).
 * Then 2N LCG steps (N Math.random() calls) reach the state S immediately
 * before the opening resetPos x draw.
 *
 * For each input state S and each N we step back 2N steps, and for each k
 * ask whether t = X0^mask - U - k (mod 2^48, plus wraps j*2^48) is a value the
 * HotSpot formula can produce for SOME integer qpc with t in [tmin,tmax].
 *
 * Usage:
 *   fpscan STATES_FILE NLIST_FILE kmin kmax f tmin_ns tmax_ns U [maxprint]
 *   STATES_FILE: one decimal 48-bit state per line (state before opening x draw)
 *   NLIST_FILE : lines "N" or "N0 N1" (inclusive contiguous range)
 *   f          : QPC frequency in Hz (integer); f>=1e9 means effectively dense
 * Output: "PASS idx N k t qpc" lines (up to maxprint) and a final summary line
 *   "SUMMARY states=... trials=... passes=..." where trials = states*N*k.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define A 0x5DEECE66DULL
#define C 0xBULL
#define MASK ((1ULL<<48)-1)

static uint64_t AINV; /* inverse of A mod 2^48 */

static uint64_t inv48(uint64_t a){ uint64_t x=1; for(int i=0;i<7;i++) x = x*(2-a*x); return x & MASK; }

/* affine map x -> m*x+b (mod 2^48) for one BACKWARD step: x_prev = AINV*(x - C) */
typedef struct { uint64_t m,b; } aff;
static aff compose(aff f, aff g){ /* f after g */ aff r; r.m=(f.m*g.m)&MASK; r.b=(f.m*g.b+f.b)&MASK; return r; }
static aff back_pow(uint64_t steps){
  aff one={AINV,(AINV*((0-C)&MASK))&MASK}, r={1,0};
  while(steps){ if(steps&1) r=compose(one,r); one=compose(one,one); steps>>=1; }
  return r;
}

/* HotSpot 6 Windows javaTimeNanos for a counter value q */
static inline int64_t hs_nanos(int64_t q, int64_t f){
  double current=(double)q, freq=(double)f;
  return (int64_t)((current/freq)*1000000000.0);
}

/* is t producible by hs_nanos for some integer q>=0 ? returns q or -1 */
static int64_t grid_q(int64_t t, int64_t f){
  __int128 est = ((__int128)t * f) / 1000000000;
  for(int64_t d=-2; d<=3; d++){
    int64_t q=(int64_t)est+d; if(q<0) continue;
    if(hs_nanos(q,f)==t) return q;
  }
  return -1;
}

int main(int argc,char**argv){
  if(argc<9){fprintf(stderr,"usage\n");return 2;}
  AINV=inv48(A);
  if(((A*AINV)&MASK)!=1){fprintf(stderr,"inverse fail\n");return 3;}
  FILE*fs=fopen(argv[1],"r"); if(!fs){perror("states");return 2;}
  size_t cap=1<<16,n=0; uint64_t*S=malloc(cap*8);
  unsigned long long v; while(fscanf(fs,"%llu",&v)==1){ if(n==cap){cap*=2;S=realloc(S,cap*8);} S[n++]=v&MASK; }
  fclose(fs);
  int kmin=atoi(argv[3]),kmax=atoi(argv[4]);
  int64_t f=atoll(argv[5]), tmin=atoll(argv[6]), tmax=atoll(argv[7]);
  uint64_t U=strtoull(argv[8],0,10);
  long maxprint = argc>9? atol(argv[9]) : 1000;
  uint64_t *X=malloc(n*8);
  FILE*fn=fopen(argv[2],"r"); if(!fn){perror("nlist");return 2;}
  char line[256]; unsigned long long trials=0,passes=0; long printed=0;
  while(fgets(line,sizeof line,fn)){
    unsigned long long n0,n1; int got=sscanf(line,"%llu %llu",&n0,&n1);
    if(got<1) continue; if(got==1) n1=n0;
    aff j=back_pow(2*n0);
    for(size_t i=0;i<n;i++) X[i]=(j.m*S[i]+j.b)&MASK;
    aff two=back_pow(2);
    for(unsigned long long N=n0;;N++){
      for(size_t i=0;i<n;i++){
        uint64_t s48 = X[i] ^ (A & MASK);
        for(int k=kmin;k<=kmax;k++){
          trials++;
          uint64_t r = (s48 - ((U + (uint64_t)k)&MASK)) & MASK;
          for(int64_t t=(int64_t)r; t<=tmax; t+=(int64_t)1<<48){
            if(t<tmin) continue;
            int64_t q=grid_q(t,f);
            if(q>=0){ passes++; if(printed<maxprint){printf("PASS %zu %llu %d %lld %lld\n",i,N,k,(long long)t,(long long)q);printed++;} }
          }
        }
      }
      if(N==n1) break;
      for(size_t i=0;i<n;i++) X[i]=(two.m*X[i]+two.b)&MASK;
    }
  }
  printf("SUMMARY states=%zu trials=%llu passes=%llu rate=%.6g\n",n,trials,passes,trials?(double)passes/trials:0.0);
  return 0;
}
