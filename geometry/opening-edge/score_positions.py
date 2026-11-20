"""Frozen opening-edge ranking; consumes positions only after calibration freeze.

Input JSON: [{"id": ..., "player_xz": [x,z]}, ...]. Does not derive candidates.
No pass/fail threshold; nuisance parameters are selected on frames 39 and 41.
Frame 44 is held out from nuisance selection and reported separately.
"""
from pathlib import Path
import argparse,json
import numpy as np
import fit_opening as m

P=Path(__file__).parent

def score(position):
    observed=np.array(json.loads((P/'fit-results.json').read_text())['observed_edge_rows'])
    alternatives=[]
    for variant,pose in enumerate(m.poses):
        m.F=180/np.tan(np.deg2rad(pose['fov']/2))
        for delta in range(64):
            with np.errstate(all='ignore'):
                raw=np.array([m.boundary(position,fr,delta) for fr in pose['frames']])-observed
            if not np.isfinite(raw).all():
                continue
            # Exact least-squares optimum of one shared additive edge-row bias.
            bias=float(np.clip(-raw[:2].mean(),-1,1))
            residual=raw+bias
            alternatives.append({'pose_variant':variant,'fov':pose['fov'],'far_plane_below_foreground_blocks':delta,'common_pixel_bias':bias,'train_rmse_px':float(np.sqrt(np.mean(residual[:2]**2))),'heldout_rmse_px':float(np.sqrt(np.mean(residual[2]**2))),'frame_mean_residual_px':residual.mean(axis=1).tolist()})
    alternatives.sort(key=lambda a:a['train_rmse_px'])
    return {'best_training_nuisance':alternatives[0]if alternatives else None,'five_best_training_alternatives':alternatives[:5]}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('positions');a.add_argument('output');a=a.parse_args()
    rows=json.loads(Path(a.positions).read_text());out=[]
    for row in rows:
        out.append({'id':row.get('id'), 'player_xz':row['player_xz'],**score(row['player_xz'])})
    out.sort(key=lambda r:r['best_training_nuisance']['train_rmse_px'] if r['best_training_nuisance'] else float('inf'))
    Path(a.output).write_text(json.dumps({'status':'conditional geometric ranking only; no acceptance cutoff','assumption':'skyline is the boundary of the complete [0,256]x[0,256] world on one common-height plane','train_frames':[39,41],'heldout_frame':44,'ranked':out},indent=2)+'\n')
