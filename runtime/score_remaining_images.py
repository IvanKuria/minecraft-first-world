"""Run predeclared image scorers on the baseline175conditional candidates.

Each result preserves separate training/heldout scores. No acceptance cutoff
or combined probability is invented. Other hidden-reset families are not
scored by this bounded baseline experiment.
"""
import json,sys,hashlib,time,concurrent.futures
from pathlib import Path

BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;FOLDER=BASE/'conditional-two-reset'
sys.path[:0]=[str(ROOT/'geometry/edge-registration'),str(ROOT/'geometry/opening-edge'),
              str(ROOT/'generator/first-reset-boundary')]
import frozen_photometric_score as photo
import score_positions as opening
import edge_constraint as first

def work(candidate):
    return {**candidate,'second_reset_photometric':photo.score(*candidate['second_reset']),
            'first_reset_edge':first.score(candidate['first_reset'][0]),
            'opening_edge':opening.score(candidate['opening'])}

def main():
    manifests=[(ROOT/'geometry/edge-registration/frozen-photometric-manifest.json',ROOT/'geometry','files_sha256'),
               (ROOT/'generator/first-reset-boundary/frozen-edge-constraint.json',ROOT/'generator','inputs_sha256')]
    for path,base,key in manifests:
        for name,digest in json.loads(path.read_text())[key].items():
            assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
    opening_manifest=ROOT/'geometry/opening-edge/freeze-manifest.json'
    for row in json.loads(opening_manifest.read_text())['files']:
        assert hashlib.sha256((ROOT.parents[1]/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    source=FOLDER/'candidate-positions.json';candidates=json.loads(source.read_text())['candidates']
    rows=[];started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(work,candidates):
            rows.append(row)
            if len(rows)%25==0:print(json.dumps({'scored':len(rows),'total':len(candidates),'seconds':time.monotonic()-started}),flush=True)
    rows.sort(key=lambda r:r['second_reset_photometric']['training_cost'])
    report={'status':'Conditional baseline image rankings only; NOT recovered seed',
        'all_baseline_candidates_scored':len(rows)==len(candidates),
        'candidate_positions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'ranking':'second-reset photometric TRAINING cost only; all edge and heldout scores reported separately',
        'limits':['Only the zero-hidden-reset175candidate baseline is evaluated',
                  'Camera/terrain/staticrawreset/callspacing assumptions remain',
                  'Local pose optimization is not a global confidence or completeness calculation',
                  'No acceptance cutoff, combined probability, uniqueness or seed claim'],
        'seconds':time.monotonic()-started,'candidates':rows}
    (FOLDER/'additional-image-rankings.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'scored':len(rows),'seconds':report['seconds'],
       'top5':[{'train_photo':r['second_reset_photometric']['mean_training_correlation'],
                'heldout_photo':r['second_reset_photometric']['mean_heldout_correlation'],
                'opening_edge_train':r['opening_edge']['best_training_nuisance']['train_rmse_px'],
                'opening_edge_heldout':r['opening_edge']['best_training_nuisance']['heldout_rmse_px'],
                'first_edge_train':r['first_reset_edge']['best']['train_rms']} for r in rows[:5]]}),flush=True)

if __name__=='__main__':main()
