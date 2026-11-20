"""Apply a frozen image-derived scorer to conditional RNG candidates."""
import json,sys,hashlib,time
from pathlib import Path

BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
sys.path.insert(0,str(ROOT/'geometry/edge-registration'))
import frozen_geometry_score as scorer

folder=BASE/'conditional-two-reset'
source=folder/'candidate-positions.json'
manifest_path=ROOT/'geometry/edge-registration/frozen-score-manifest.json'
manifest=json.loads(manifest_path.read_text())
for name,digest in manifest['files_sha256'].items():
    assert hashlib.sha256((manifest_path.parent/name).read_bytes()).hexdigest()==digest,name
rows=[];started=time.monotonic()
for i,candidate in enumerate(json.loads(source.read_text())['candidates']):
    scored=scorer.score(*candidate['second_reset'])
    rows.append({**candidate,'edge_fit':scored})
    if (i+1)%25==0:print(json.dumps({'scored':i+1,'seconds':time.monotonic()-started}),flush=True)
rows.sort(key=lambda r:r['edge_fit']['best']['training_rms_px'])
report={'status':'Geometry ranking of a conditional candidate family, NOT recovered seed',
        'score_frozen_before_candidate_inspection':True,
        'scorer_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'candidate_positions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'ranking':'Only first4edgegroups; lastframe reported asheldout, not included inranking',
        'limits':['The175states belong tooneunverified random-call-spacing/static-reset scenario',
                  'Eachcandidate profilesdiscreteintegerheight/ROIvariant usingtrainonly',
                  'No acceptance threshold or probability was calibrated',
                  'Scorer assumesworldedges,sharedgrassplane,70FOV,.3cameraoffset; see frozenmanifest'],
        'seconds':time.monotonic()-started,'candidates':rows}
(folder/'second-edge-ranking.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'scored':len(rows),'seconds':report['seconds'],
    'top5':[{'train':r['edge_fit']['best']['training_rms_px'],
             'heldout':r['edge_fit']['best']['heldout_lastframe_rms_px'],
             'second_reset':r['second_reset']} for r in rows[:5]]}),flush=True)
