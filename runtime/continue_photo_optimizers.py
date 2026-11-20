"""Convergence diagnostic for baseline runs that exhausted700evaluations.

No ROI, loss, bounds or candidate selection changes. Continue from saved
parameters with an additional2000evaluation budget; preserve original results.
"""
import sys,json,time,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;FOLDER=BASE/'conditional-two-reset'
sys.path.insert(0,str(ROOT/'geometry/edge-registration'))
import frozen_photometric_score as photo

def work(row):
    prior=row['second_reset_photometric'];phase=np.array(prior['player_phase'])
    def residual(v):
        return np.concatenate([(photo.normalize(photo.prediction(v[3*i:3*i+3],phase,i))-photo.OBS[i])/np.sqrt(len(photo.OBS[i])) for i in range(3)])
    sol=least_squares(residual,np.array(prior['parameters']),bounds=(photo.LOW,photo.HIGH),
        x_scale=np.tile([.01,.01,.1],3),max_nfev=2000,ftol=1e-10,xtol=1e-10)
    held=[]
    for i,d in enumerate(photo.DATA):
        held.append(float(np.corrcoef(photo.prediction(sol.x[3*i:3*i+3],phase,i,True),d['im'][d['YT'],d['XT']].ravel())[0,1]))
    return {'state_before_first_observed_reset':row['state_before_first_observed_reset'],
        'initial_cost':prior['training_cost'],'continued_cost':float(sol.cost),'additional_nfev':int(sol.nfev),
        'success':bool(sol.success),'message':str(sol.message),'active_bounds':sol.active_mask.tolist(),
        'heldout_correlations':held,'parameters':sol.x.tolist()}

def main():
    rows=json.loads((FOLDER/'additional-image-rankings.json').read_text())['candidates']
    failed=[r for r in rows if not r['second_reset_photometric']['optimizer_success']]
    started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(work,failed))
    report={'status':'Local optimizer convergence diagnostic, not seed authentication',
            'original_failures':len(failed),'continued_successes':sum(r['success'] for r in results),
            'change':'Additional2000evaluations from savedparameters; alldata/objective/bounds unchanged',
            'seconds':time.monotonic()-started,'results':results}
    (FOLDER/'photometric-convergence-continuation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)

if __name__=='__main__':main()
