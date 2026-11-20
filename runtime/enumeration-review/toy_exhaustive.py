"""Independent exhaustive test of the enclosing-box lattice enumeration method.

16-bit LCG, 12+13 output bits, float32 cast: every one of 65536 states is tested.
The smaller modulus permits exhaustive coverage checks; this is NOT a historical
RNG search, nor a numerical completeness proof for the separate 48-bit solver.
"""
import json, math, random, struct, time
from pathlib import Path
from fpylll import IntegerMatrix, LLL, GSO, Enumeration, EvaluatorStrategy, FPLLL

BITS, K, L = 16, 12, 13
M, A, C = 1 << BITS, 0x5DEECE66D % (1 << BITS), 11

def affine(steps):
    assert steps >= 0
    a, c = 1, 0
    for _ in range(steps):
        a, c = a*A % M, (c*A+C) % M
    return a, c

def advance(s, steps):
    a, c = affine(steps)
    return (a*s+c) % M

def coordinate(first):
    second = (A*first+C) % M
    number = (((first >> (BITS-K)) << L) + (second >> (BITS-L))) / 2**(K+L)
    return struct.unpack('f', struct.pack('f', number))[0] * 256

def bounds_for(obs, error):
    # Same formula as reviewed solver with bit widths substituted.
    allowance = error + 2**-15
    lo = max(0, math.floor((obs-allowance)*2**(K-8))) << (BITS-K)
    hi = min((1<<K)-1, math.ceil((obs+allowance)*2**(K-8)))
    return lo, ((hi+1) << (BITS-K))-1

def enumerate_toy(observations, error, indices, cap):
    n = len(observations)
    B = IntegerMatrix(n,n)
    bounds, offsets, target = [], [], []
    for i,(obs,index) in enumerate(zip(observations,indices)):
        lo,hi = bounds_for(obs,error)
        a,c = affine(2*(index-indices[0]))
        bounds.append((lo,hi)); offsets.append(c); target.append((lo+hi)//2-c)
        B[0,i] = a
        if i: B[i,i] = M
    LLL.reduction(B)
    FPLLL.set_precision(160)
    gs = GSO.Mat(B,float_type='mpfr'); gs.update_gso()
    radius = sum((max(abs(lo-(t+off)),abs(hi-(t+off)))+2)**2
                 for t,off,(lo,hi) in zip(target,offsets,bounds))
    enum = Enumeration(gs,nr_solutions=cap,strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS)
    try:
        solutions = enum.enumerate(0,n,float(radius)*(1+1e-9),0,
                                   target=gs.from_canonical(tuple(target)))
    except Exception as exc:
        # Record rather than silently treating numerical/internal errors as empty.
        return {'exception':type(exc).__name__+': '+str(exc)}, set()
    states=set()
    a,c=affine(2*indices[0]+1)
    inv=pow(a,-1,M)
    for _,coeff in solutions:
        v=B.multiply_left(tuple(int(round(x)) for x in coeff))
        if not all(lo<=v[i]+offsets[i]<=hi for i,(lo,hi) in enumerate(bounds)):
            continue
        initial=((v[0]%M-c)*inv)%M
        if all(abs(coordinate(advance(initial,2*i+1))-obs)<=error+1e-12
               for i,obs in zip(indices,observations)):
            states.add(initial)
    return {'sphere_points':len(solutions),'hit_cap':len(solutions)==cap,
            'cap':cap,'nodes':enum.get_nodes()},states

def main():
    started=time.monotonic()
    rng=random.Random(39385)
    tests=[]
    for i in range(36):
        indices=([0,1],[0,1,2],[3,4,9],[0,1,6,7],[5,5,7])[i%5]
        error=[.01,.25,1,4,8,16][i%6]
        seed=rng.randrange(M)
        values=[coordinate(advance(seed,2*j+1)) for j in indices]
        observations=[round(x/(2*error))*(2*error) for x in values]
        tests.append((f'quantized-{i}',observations,error,indices,250000))
    # Exactly measurable output, narrow edges, inconsistent observations, and cap.
    tests += [('left-boundary',[0,0],.2,[0,1],250000),
              ('right-boundary',[256,256],.2,[0,1],250000),
              ('inconsistent',[128,0,256],0,[0,1,2],250000),
              ('exact',[coordinate(advance(12345,1)),coordinate(advance(12345,3))],0,[0,1],250000),
              ('capped',[128,128],16,[0,1],5)]
    rows=[]
    for name,obs,error,indices,cap in tests:
        expected={s for s in range(M)
                  if all(abs(coordinate(advance(s,2*j+1))-o)<=error+1e-12
                         for j,o in zip(indices,obs))}
        info,found=enumerate_toy(obs,error,indices,cap)
        # Independently check the coarse box contains every true output's first state.
        bounds_valid=all(all(lo<=advance(s,2*j+1)<=hi
                            for (lo,hi),j in zip([bounds_for(o,error) for o in obs],indices))
                         for s in expected)
        row={'name':name,'indices':indices,'error':error,'observations':obs,
             'brute_force_count':len(expected),'enumerated_count':len(found),
             'missing':sorted(expected-found),'spurious':sorted(found-expected),
             'coarse_bounds_contain_all_exact_candidates':bounds_valid,**info}
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ['missing','spurious','observations']},sort_keys=True),flush=True)
    summary={'bits':BITS,'states_exhaustively_checked_per_case':M,
             'output_bits':[K,L],'float32_cast':True,'tests':rows,
             'seconds':time.monotonic()-started}
    Path(__file__).with_name('toy-results.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__': main()
