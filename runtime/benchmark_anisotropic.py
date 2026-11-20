"""Validate unequal coordinate uncertainties on actual Java-generated fixtures.

The narrow-box comparisons have a separately enumerated broad-box reference.
All coordinates are synthetic controls; no claimed historical bounds here.
"""
import json
from pathlib import Path
from enumerate_spawn_states import enumerate_states
from benchmark_spawn_recovery import A,M,coordinate,advance

BASE=Path(__file__).resolve().parent

def main():
    fixtures=[json.loads(s) for s in (BASE/'spawn-fixtures.jsonl').read_text().splitlines()]
    previous=json.loads((BASE/'spawn-state-enumeration.json').read_text())['tests']
    rows=[]
    # A pre-existing uniform enumeration provides an independent reference set
    # for each subset. Bounds are deliberately offset, not centered on truth.
    for i,fixture in enumerate(fixtures[:8]):
        broad=next(r for r in previous if r['fixture']==i and r['error_blocks']==2.)
        assert not broad['hit_output_cap']
        errors=[2.,.2,2.,.2,2.,.2]
        obs=list(broad['observations'])
        for j in [1,3,5]:
            obs[j]=round(fixture['coordinates'][j]/.4)*.4
            # Keep narrow interval entirely within independently enumerated box.
            obs[j]=min(max(obs[j],broad['observations'][j]-1.8),broad['observations'][j]+1.8)
        expected={s for s in broad['states'] if all(
            abs(coordinate(advance(s,2*j+1))-o)<=e+1e-12
            for j,(o,e) in enumerate(zip(obs,errors)))}
        result=enumerate_states(obs,errors)
        got=set(result.pop('states'))
        row={'kind':'subset-reference','fixture':i,'expected_count':len(expected),
             'candidate_count':len(got),'exact_set_match':got==expected,
             'true_state_retained':((fixture['seed']^A)%M) in got,**result}
        rows.append(row);print(json.dumps(row),flush=True)
    for errors in ([.25,8.]*3,[1.,20.]*3):
        for i,fixture in enumerate(fixtures[:4]):
            obs=[round(v/(2*e))*(2*e) for v,e in zip(fixture['coordinates'],errors)]
            result=enumerate_states(obs,errors)
            candidates=result.pop('states')
            row={'kind':'elongated-box','fixture':i,'observations':obs,
                 'candidate_count':len(candidates),
                 'true_state_retained':((fixture['seed']^A)%M) in candidates,**result}
            rows.append(row);print(json.dumps(row),flush=True)
    (BASE/'anisotropic-enumeration-results.json').write_text(json.dumps({
        'label':'Synthetic unequal-uncertainty checks, no historical inputs',
        'limits':['Floating enumeration is not a formal completeness certificate',
                  'Hitting sphere output cap makes candidate set incomplete',
                  'Real original coordinates and original RNG equivalence are unresolved'],
        'tests':rows},indent=2)+'\n')

if __name__=='__main__':main()
