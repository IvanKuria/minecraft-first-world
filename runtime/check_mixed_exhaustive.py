"""Run the production mixed-column solver against every state of a toy LCG.

The actual solver function is reused; only the PRNG/output bit widths and
rigorous output-approximation guard are substituted. This checks the mixed
moduli, weighted columns, wrapped intervals and skipped coordinate indices.
It does not certify the floating enumeration numerics at48bits.
"""
import json,random,sys
from pathlib import Path
import recover_mixed_coordinates as solver

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'enumeration-review'))
import toy_exhaustive as toy

def main():
    solver.M=toy.M;solver.SCALE=toy.M//256
    solver.GUARD=2**(8-toy.K)+2**-17+1/solver.SCALE
    solver.affine=toy.affine;solver.advance=toy.advance;solver.coordinate=toy.coordinate
    rng=random.Random(130509)
    cache={};rows=[]
    for trial in range(20):
        indices=([0,1,2,3],[0,1,6,7],[3,4,11,12],[0,0,4,4])[trial%4]
        truth=rng.randrange(toy.M)
        values=[toy.coordinate(toy.advance(truth,2*i+1)) for i in indices]
        errors=([2,25,1,10],[8,8,8,8],[128,128,128,128],[1,25,10,20])[trial%4]
        ferr=[.01,.03,.1,.2][trial%4]
        absolute=[v+rng.uniform(-e,e) for v,e in zip(values,errors)]
        fractional=[(v+rng.uniform(-ferr,ferr))%1 for v in values]
        expected=set(range(toy.M))
        for i,o,e,f in zip(indices,absolute,errors,fractional):
            if i not in cache:
                cache[i]=[toy.coordinate(toy.advance(s,2*i+1)) for s in range(toy.M)]
            vals=cache[i]
            expected={s for s in expected if abs(vals[s]-o)<=e+1e-12 and abs((vals[s]-f+.5)%1-.5)<=ferr+1e-12}
        result=solver.recover(absolute,errors,fractional,[ferr]*4,indices=indices,cap=250000)
        found=set(result.pop('states'))
        row={'trial':trial,'indices':indices,'brute_force_count':len(expected),
             'exact_match':found==expected,'missing':sorted(expected-found),
             'spurious':sorted(found-expected),'true_state_retained':truth in found,**result}
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['missing','spurious']}),flush=True)
    (BASE/'mixed-exhaustive-results.json').write_text(json.dumps({
        'bits':16,'states_checked_per_trial':65536,
        'label':'Exhaustive toy test of production solver structure, not48bit numeric certification',
        'tests':rows},indent=2)+'\n')

if __name__=='__main__':main()
