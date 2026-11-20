"""Partition fractional intervals to keep mixed-state enumeration bounded.

Volume estimates only schedule work; every accepted leaf is enumerated and
filtered. A capped leaf is subdivided again. No completeness claim is made if
the explicit call limit is reached, nor as a formal floating-point certificate.
"""
import json,math,time
from pathlib import Path
from recover_mixed_coordinates import recover

BASE=Path(__file__).resolve().parent

def tiled_recover(absolute,absolute_errors,fractional,fractional_errors,indices=None,
                  cap=250000,max_calls=256,target_estimate=1200):
    pending=[(list(fractional),list(fractional_errors),0)]
    states=set();leaves=[];calls=0;started=time.monotonic();unsearched=[]
    while pending:
        frac,errs,depth=pending.pop()
        estimate=2**48*math.prod(min(256,2*e)/256*2*f for e,f in zip(absolute_errors,errs))
        result=None
        if estimate<=target_estimate:
            if calls>=max_calls:
                unsearched.append({'fractional':frac,'errors':errs,'depth':depth});continue
            result=recover(absolute,absolute_errors,frac,errs,indices,cap)
            calls+=1
            if not result['hit_output_cap']:
                states.update(result.pop('states'))
                leaves.append({'fractional':frac,'errors':errs,'depth':depth,**result})
                if calls%8==0:print(json.dumps({'completed_leaves':len(leaves),'calls':calls,
                    'unique_states_so_far':len(states),'pending_boxes':len(pending)}),flush=True)
                continue
        # The heuristic split is exact interval coverage, including shared edge.
        # A cap-hit partial result is discarded, never used as an exclusion.
        axis=max(range(len(errs)),key=lambda i:errs[i])
        half=errs[axis]/2
        if half<1e-12:
            unsearched.append({'fractional':frac,'errors':errs,'depth':depth,'reason':'minimum split width'});continue
        for sign in [-1,1]:
            child=list(frac);child[axis]=(child[axis]+sign*half)%1
            widths=list(errs);widths[axis]=half
            pending.append((child,widths,depth+1))
    return {'states':sorted(states),'candidate_count':len(states),'leaves':leaves,
            'enumeration_calls':calls,'unsearched_boxes':unsearched,
            'covered_all_boxes_without_output_cap':not unsearched,
            'seconds':time.monotonic()-started,
            'numerical_completeness':'160bitGSO with outwardradiusguard, not a formal certificate'}

def main():
    # This is an explicit conditional historical SCENARIO, not measured raw RNG
    # outputs or a recovered seed. See the mandatory assumptions saved with it.
    scenario={
        'label':'Conditional stationary two-reset candidate family; NOT recovered seed',
        'absolute_intervals':[[195,201],[188,250],[222,225],[113,135]],
        'fractional_intervals':[[.42,.50],[.82,.92],[.78,.85],[.79,.86]],
        'coordinate_indices':[0,1,2,3],
        'assumptions':[
            'Filmed reset uses the later preserved Math.random float-before256 code',
            'Texture camera-to-player convention matches the later .3block setback',
            'Effective image FOV70 and squarepixelcenter model are appropriate',
            'Player horizontal coordinates stayed fixed from RAW reset through measured cleargrassviews',
            'Coarse skyline registration assumes topY64 and measuredpixelerror envelope',
            'Worldcorner(256,0) axes from crossframe texture/skyline registration',
            'Only these two observed resets consumed random numbers: no heldR extra resets or randomchunkstress calls',
            'Selected intervals are outward working scenario ranges, not calibrated confidence intervals'],
        'interpretation':'Many states can satisfy these assumptions. A retained state is not authenticated; missing generator and independent holdout remain unresolved.'}
    ab=[(a+b)/2 for a,b in scenario['absolute_intervals']]
    ae=[(b-a)/2 for a,b in scenario['absolute_intervals']]
    fr=[(a+b)/2 for a,b in scenario['fractional_intervals']]
    fe=[(b-a)/2 for a,b in scenario['fractional_intervals']]
    result=tiled_recover(ab,ae,fr,fe,scenario['coordinate_indices'])
    out=BASE/'conditional-two-reset';out.mkdir(exist_ok=True)
    states=result.pop('states')
    (out/'candidate-internal-states.json').write_text(json.dumps(states)+'\n')
    (out/'scenario-and-results.json').write_text(json.dumps({**scenario,**result},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['leaves','unsearched_boxes']}),flush=True)

if __name__=='__main__':main()
