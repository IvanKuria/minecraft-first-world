"""Retain multiple plausible Java RNG states from coarse coordinate intervals.

Enclosing-sphere lattice enumeration, exact coordinate filtering, bounded output.
This improves on choosing a single nearest point, but numerical enumeration is
not represented as a formal completeness certificate. No historical input here.
"""
import json, math, time
from pathlib import Path
from fpylll import IntegerMatrix, LLL, GSO, Enumeration, EnumerationError, EvaluatorStrategy, FPLLL
from benchmark_spawn_recovery import A,C,M,affine,advance,coordinate

BASE=Path(__file__).resolve().parent

def enumerate_states(observations,error,indices=None,cap=250000):
    if indices is None:indices=list(range(len(observations)))
    if not observations or not math.isfinite(error) or error<0:
        raise ValueError('nonempty observations and finite nonnegative error required')
    if len(indices)!=len(observations) or any(not isinstance(i,int) or i<0 for i in indices):
        raise ValueError('one nonnegative integer index required per observation')
    if any(a>b for a,b in zip(indices,indices[1:])):
        raise ValueError('indices must be nondecreasing')
    if any(not math.isfinite(v) for v in observations) or not isinstance(cap,int) or cap<1:
        raise ValueError('finite observations and a positive integer cap required')
    n=len(observations);B=IntegerMatrix(n,n)
    target=[];offsets=[];bounds=[]
    for i,(value,index) in enumerate(zip(observations,indices)):
        allowance=error+2**-15
        lo=max(0,math.floor((value-allowance)*2**18))<<22
        hi=min((1<<26)-1,math.ceil((value+allowance)*2**18))
        hi=((hi+1)<<22)-1
        if hi<lo:
            return {'states':[],'sphere_points_visited':0,'output_cap':cap,
                    'hit_output_cap':False,'enumeration_nodes':0,'seconds':0.,
                    'reason':'An observation interval lies outside possible world coordinates'}
        mul,off=affine(2*(index-indices[0]))
        target.append((lo+hi)//2-off);offsets.append(off);bounds.append((lo,hi))
        B[0,i]=mul
        if i:B[i,i]=M
    LLL.reduction(B)
    FPLLL.set_precision(160)
    gs=GSO.Mat(B,float_type='mpfr');gs.update_gso()
    radius_sq=sum((max(abs(lo-(t+off)),abs(hi-(t+off)))+2)**2
                  for t,off,(lo,hi) in zip(target,offsets,bounds))
    started=time.monotonic()
    enum=Enumeration(gs,nr_solutions=cap,strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS)
    try:
        solutions=enum.enumerate(0,n,float(radius_sq)*(1+1e-9),0,
                                 target=gs.from_canonical(tuple(target)))
    except EnumerationError as exc:
        if str(exc)!='No solution found.':raise
        solutions=[]
    candidates=set()
    mul,off=affine(2*indices[0]+1);inverse=pow(mul,-1,M)
    for distance,coeffs in solutions:
        v=B.multiply_left(tuple(int(round(c)) for c in coeffs))
        if not all(lo<=v[i]+offsets[i]<=hi for i,(lo,hi) in enumerate(bounds)):continue
        initial=((v[0]%M-off)*inverse)%M
        if all(abs(coordinate(advance(initial,2*i+1))-obs)<=error+1e-12
               for i,obs in zip(indices,observations)):
            candidates.add(initial)
    return {'states':sorted(candidates),'sphere_points_visited':len(solutions),
            'output_cap':cap,'hit_output_cap':len(solutions)==cap,
            'enumeration_nodes':enum.get_nodes(),'seconds':time.monotonic()-started}

def main():
    fixtures=[json.loads(s) for s in (BASE/'spawn-fixtures.jsonl').read_text().splitlines()]
    result={'label':'Synthetic bounded candidate enumeration; no historical state recovered',
            'limits':['Known call spacing and absolute coordinate registration',
                      'Floating enumeration with160bit GSO and outward radius guard, no formal completeness certificate',
                      'Output cap explicitly reported; cap hit means incomplete search',
                      'Independent heldout coordinate is available only in synthetic controls'], 'tests':[]}
    for error in [.5,1.,2.]:
        for sample,fixture in enumerate(fixtures[:8]):
            observations=[round(v/(2*error))*(2*error) for v in fixture['coordinates'][:6]]
            row=enumerate_states(observations,error)
            actual=(fixture['seed']^A)%M
            heldout=fixture['coordinates'][6]
            survivors=[s for s in row['states'] if abs(coordinate(advance(s,13))-heldout)<1e-12]
            row.update({'fixture':sample,'error_blocks':error,'observations':observations,
                        'true_state_retained':actual in row['states'],
                        'candidate_count':len(row['states']),
                        'heldout_survivors':survivors})
            result['tests'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ['states','observations']}),flush=True)
    (BASE/'spawn-state-enumeration.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
