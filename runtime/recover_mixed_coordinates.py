"""Joint coarse absolute + precise fractional Java spawn-coordinate constraints.

Each observed coordinate contributes two lattice columns: modulus2^48 for
its absolute interval and modulus2^40 for its position within a voxel. The
common first row forces both to arise from the SAME RNG state. This avoids
enumerating every integer location or every low-state lift separately.

Approximation error is bounded by3*2^-18 blocks; exact Java-float outputs
are checked after enumeration. All benchmark inputs are synthetic controls.
"""
import json,math,time
from pathlib import Path
from fpylll import IntegerMatrix,LLL,GSO,Enumeration,EnumerationError,EvaluatorStrategy,FPLLL
from benchmark_spawn_recovery import A,C,M,affine,advance,coordinate

BASE=Path(__file__).resolve().parent
SCALE=1<<40
GUARD=3*2**-18+2**-40

def circular_error(a,b):
    return abs((a-b+.5)%1-.5)

def recover(absolute,absolute_errors,fractional,fractional_errors,indices=None,cap=1000000):
    count=len(absolute)
    if indices is None:indices=list(range(count))
    if not count or any(len(v)!=count for v in [absolute_errors,fractional,fractional_errors,indices]):
        raise ValueError('equal nonempty coordinate and uncertainty arrays required')
    if any(not isinstance(i,int) or i<0 for i in indices) or indices!=sorted(indices):
        raise ValueError('nondecreasing nonnegative integer coordinate indices required')
    if any(not math.isfinite(v) for vs in [absolute,absolute_errors,fractional,fractional_errors] for v in vs):
        raise ValueError('all values must be finite')
    if any(e<0 for e in absolute_errors) or any(e<0 or e>=.5 for e in fractional_errors):
        raise ValueError('nonnegative uncertainties required; fractional error must be below.5')
    if not isinstance(cap,int) or cap<1:raise ValueError('positive cap required')
    columns=[]
    for index,obs,err,frac,ferr in zip(indices,absolute,absolute_errors,fractional,fractional_errors):
        mul,off=affine(2*(index-indices[0]))
        for value,error,modulus in [(obs,err,M),(frac%1,ferr,SCALE)]:
            lo=math.floor((value-error-GUARD)*SCALE)
            hi=math.ceil((value+error+GUARD)*SCALE)
            if modulus==M:lo=max(0,lo);hi=min(M-1,hi)
            if lo>hi:
                return {'states':[],'hit_output_cap':False,'reason':'outside world'}
            columns.append({'mul':mul%modulus,'off':off%modulus,'modulus':modulus,'lo':lo,'hi':hi})
    n=len(columns);widest=max(c['hi']-c['lo']+1 for c in columns)
    weights=[1<<max(0,(widest//(c['hi']-c['lo']+1)).bit_length()-1) for c in columns]
    B=IntegerMatrix(n,n);target=[];radius=0
    for i,(c,w) in enumerate(zip(columns,weights)):
        center=(c['lo']+c['hi'])//2
        target.append((center-c['off'])*w)
        radius+=((max(center-c['lo'],c['hi']-center)+2)*w)**2
        B[0,i]=c['mul']*w
        if i:B[i,i]=c['modulus']*w
    LLL.reduction(B);FPLLL.set_precision(160)
    gs=GSO.Mat(B,float_type='mpfr');gs.update_gso()
    enum=Enumeration(gs,nr_solutions=cap,strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS)
    started=time.monotonic()
    try:
        solutions=enum.enumerate(0,n,float(radius)*(1+1e-9),0,target=gs.from_canonical(tuple(target)))
    except EnumerationError as exc:
        if str(exc)!='No solution found.':raise
        solutions=[]
    mul,off=affine(2*indices[0]+1);inverse=pow(mul,-1,M)
    states=set()
    for _,coeff in solutions:
        vector=B.multiply_left(tuple(int(round(c)) for c in coeff))
        if not all(c['lo']*w <= v+c['off']*w <= c['hi']*w
                   for v,c,w in zip(vector,columns,weights)):continue
        assert vector[0]%weights[0]==0
        state=((vector[0]//weights[0]-off)*inverse)%M
        predictions=[coordinate(advance(state,2*i+1)) for i in indices]
        if all(abs(v-o)<=e+1e-12 and circular_error(v,f)<=fe+1e-12
               for v,o,e,f,fe in zip(predictions,absolute,absolute_errors,fractional,fractional_errors)):
            states.add(state)
    return {'states':sorted(states),'candidate_count':len(states),'sphere_points':len(solutions),
            'hit_output_cap':len(solutions)==cap,'cap':cap,'nodes':enum.get_nodes(),
            'column_weights':weights,'seconds':time.monotonic()-started}

def main():
    fixtures=[json.loads(s) for s in (BASE/'spawn-fixtures.jsonl').read_text().splitlines()]
    rows=[]
    for ferr in [.005,.01,.03]:
        for sample,fixture in enumerate(fixtures[:4]):
            errors=[2.,25.,1.,10.]
            true=fixture['coordinates'][:4]
            absolute=[round(v/(2*e))*(2*e) for v,e in zip(true,errors)]
            fractional=[(round((v%1)/(2*ferr))*(2*ferr))%1 for v in true]
            result=recover(absolute,errors,fractional,[ferr]*4)
            states=result.pop('states')
            row={'fixture':sample,'fractional_error':ferr,'absolute_errors':errors,
                 'absolute_observations':absolute,'fractional_observations':fractional,
                 'true_state_retained':((fixture['seed']^A)%M) in states,**result}
            rows.append(row);print(json.dumps(row),flush=True)
    (BASE/'mixed-coordinate-results.json').write_text(json.dumps({
        'label':'Two-spawn synthetic controls only; no historical coordinates supplied',
        'limits':['Known exact random-call spacing and coordinate axes',
                  'Camera-to-raw-reset physics correction not simulated',
                  'Coarse and fractional errors independently bounded here, not established in film',
                  'Floating enumeration, not a formal completeness certificate',
                  'Original generator and its RNG relationship remain unknown'],
        'tests':rows},indent=2)+'\n')

if __name__=='__main__':main()
