"""Bounded alternative whole-reset gaps, using the frozen opening predicate.

This only tests0..4unseen whole resets per interval. It does not cover random
chunk updates, arbitrary other calls, or a different original reset routine.
"""
import json,time
from pathlib import Path
from tile_mixed_coordinates import tiled_recover
from benchmark_spawn_recovery import advance

BASE=Path(__file__).resolve().parent;FOLDER=BASE/'conditional-two-reset'
absolute_intervals=[[0,256],[0,256],[195,201],[188,250],[222,225],[113,135]]
fractional_intervals=[[.075,.130],[.785,.835],[.42,.50],[.82,.92],[.78,.85],[.79,.86]]
absolute=[(a+b)/2 for a,b in absolute_intervals];ae=[(b-a)/2 for a,b in absolute_intervals]
fractional=[(a+b)/2 for a,b in fractional_intervals];fe=[(b-a)/2 for a,b in fractional_intervals]
existing={r['state_before_first_observed_reset'] for r in json.loads((FOLDER/'opening-holdout-results.json').read_text())['survivors']}
rows=[];started=time.monotonic()
for gap1 in range(5):
    for gap2 in range(5):
        indices=[0,1,2+2*gap1,3+2*gap1,4+2*(gap1+gap2),5+2*(gap1+gap2)]
        result=tiled_recover(absolute,ae,fractional,fe,indices,cap=250000,max_calls=64,target_estimate=300)
        states=result.pop('states')
        row={'hidden_resets_before_first_observed':gap1,'hidden_resets_between_observed':gap2,
             'coordinate_indices':indices,'initial_internal_state_candidates':states,**result}
        if gap1==gap2==0:
            converted={advance(s,4) for s in states}
            assert converted==existing and result['covered_all_boxes_without_output_cap'], 'joint6coordinate solver must reproduce frozenholdout result'
            row['matches_independent_two_stage_baseline']=True
        rows.append(row)
        summary={k:v for k,v in row.items() if k not in ['initial_internal_state_candidates','leaves','unsearched_boxes']}
        print(json.dumps(summary),flush=True)
        (FOLDER/'hidden-reset-gap-results.json').write_text(json.dumps({
            'status':'Bounded conditional alternative gap families; not seed authentication',
            'absolute_intervals':absolute_intervals,'fractional_intervals':fractional_intervals,
            'limits':['0..4extra whole resets perinterval only','No arbitrary interleaved RNG calls',
                      'Allpreviouscamera/staticrawspawn/code-equivalence assumptions stillapply'],
            'seconds_so_far':time.monotonic()-started,'tests':rows},indent=2)+'\n')
