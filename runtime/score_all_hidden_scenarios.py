"""Extend unchanged frozen image scorers to every enumerated hidden-reset family.

Resume from append-only JSONL; baseline results reused by exact positions.
No acceptance cutoff or seed authentication is implied by these rankings.
"""
import json,sys,hashlib,time,concurrent.futures,struct
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;FOLDER=BASE/'conditional-two-reset'
sys.path[:0]=[str(ROOT/'geometry/edge-registration'),str(ROOT/'geometry/opening-edge'),str(ROOT/'generator/first-reset-boundary')]
import frozen_photometric_score as photo
import frozen_geometry_score as edge
import score_positions as opening
import edge_constraint as first
A,C,M=0x5DEECE66D,11,1<<48

def advance(s,n):
    for _ in range(n):s=(s*A+C)%M
    return s

def coord(s):
    t=advance(s,1)
    return struct.unpack('f',struct.pack('f',((s>>22)*2**27+(t>>21))/2**53))[0]*256

def key(row):return tuple(row['opening']+row['first_reset']+row['second_reset'])

def verify():
    for file,base,field in [
      (ROOT/'geometry/edge-registration/frozen-photometric-manifest.json',ROOT/'geometry','files_sha256'),
      (ROOT/'geometry/edge-registration/frozen-score-manifest.json',ROOT/'geometry/edge-registration','files_sha256'),
      (ROOT/'generator/first-reset-boundary/frozen-edge-constraint.json',ROOT/'generator','inputs_sha256')]:
        for name,digest in json.loads(file.read_text())[field].items():
            assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
    for r in json.loads((ROOT/'geometry/opening-edge/freeze-manifest.json').read_text())['files']:
        assert hashlib.sha256((ROOT.parents[1]/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']

def work(row):
    return {**row,'second_reset_photometric':photo.score(*row['second_reset']),
      'second_reset_edge':edge.score(*row['second_reset']),
      'first_reset_edge':first.score(row['first_reset'][0]),
      'opening_edge':opening.score(row['opening'])}

def main():
    verify();started=time.monotonic()
    source=FOLDER/'hidden-reset-gap-results.json';data=json.loads(source.read_text());rows=[]
    for test in data['tests']:
        assert test['covered_all_boxes_without_output_cap']
        g1=test['hidden_resets_before_first_observed'];g2=test['hidden_resets_between_observed'];ii=test['coordinate_indices']
        for s in test['initial_internal_state_candidates']:
            v=[coord(advance(s,2*i+1)) for i in ii]
            assert all(a<=x<=b for x,(a,b) in zip(v,data['absolute_intervals']))
            assert all(a<=x%1<=b for x,(a,b) in zip(v,data['fractional_intervals']))
            rows.append(dict(id=f'{g1}:{g2}:{s}',hidden_before=g1,hidden_between=g2,
              state_before_opening=s,state_before_first_observed_reset=advance(s,2*ii[2]),
              opening=v[:2],first_reset=v[2:4],second_reset=v[4:]))
    out=FOLDER/'all-hidden-scenario-positions.json'
    out.write_text(json.dumps(dict(status='Conditional positions, not original seeds',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidates=rows),indent=2)+'\n')
    cached={key(r):r for r in json.loads((FOLDER/'additional-image-rankings.json').read_text())['candidates']}
    cached_edge={key(r):r['edge_fit'] for r in json.loads((FOLDER/'second-edge-ranking.json').read_text())['candidates']}
    baseline=[r for r in rows if r['hidden_before']==r['hidden_between']==0]
    assert {key(r) for r in baseline}==set(cached),'Independent materialization disagrees with baseline'
    log=FOLDER/'all-hidden-image-scores.jsonl';done={}
    if log.exists():
        for line in log.read_text().splitlines():
            r=json.loads(line);done[r['id']]=r
    with log.open('a') as f:
        for row in rows:
            if row['id'] in done:continue
            k=key(row)
            if k in cached:
                result={**row,**{name:cached[k][name] for name in ['second_reset_photometric','first_reset_edge','opening_edge']},'second_reset_edge':cached_edge[k],'cache_origin':'baseline frozen scores'}
                f.write(json.dumps(result)+'\n');done[row['id']]=result
        f.flush()
        pending=[r for r in rows if r['id'] not in done]
        print(json.dumps(dict(total=len(rows),cached=len(done),pending=len(pending),unique_positions=len({key(r) for r in rows}))),flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
            for r in pool.map(work,pending,chunksize=1):
                f.write(json.dumps(r)+'\n');f.flush();done[r['id']]=r
                if len(done)%100==0:print(json.dumps(dict(scored=len(done),total=len(rows),seconds=time.monotonic()-started)),flush=True)
    results=sorted(done.values(),key=lambda r:r['second_reset_photometric']['training_cost'])
    report=dict(status='Conditional rankings across25fixed reset-gap families, NOT recovered seed',
      positions_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),all_scored=len(done)==len(rows),seconds=time.monotonic()-started,
      ranking='Second-reset photometric TRAINING cost only; edge and heldout metrics remain separate',
      limits=['Raw-spawn stationarity and later-build code transfer remain unproved','Only0..4hidden whole resets per gap; not arbitrary random calls','Local optimization; no calibrated acceptance threshold','Missing original generator prevents terrain/seed authentication'],candidates=results)
    (FOLDER/'all-hidden-image-rankings.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,scored=len(done),seconds=report['seconds'])),flush=True)
if __name__=='__main__':main()
