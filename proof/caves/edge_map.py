"""RMSE map of the frozen opening world-edge model (same-height plane = plain top) over the whole world, fixed70 pose
variants only + the scorer's bias rule; frac parts fixed at the opening texture phase."""
import sys, json
from pathlib import Path
import numpy as np
EDGE = Path(__file__).resolve().parents[2] / 'geometry/opening-edge'
sys.path.insert(0, str(EDGE))
import fit_opening as m
obs = np.array(json.loads((EDGE / 'fit-results.json').read_text())['observed_edge_rows'])
P = Path(__file__).parent
xs = np.arange(0, 256) + .096; zs = np.arange(0, 256) + .804
step = int(sys.argv[1]) if len(sys.argv) > 1 else 2
tr = np.full((256, 256), np.nan); ho = np.full((256, 256), np.nan)
poses = [(v, p) for v, p in enumerate(m.poses)]
for i in range(0, 256, step):
    for j in range(0, 256, step):
        best = None
        for v, pose in poses:
            m.F = 180 / np.tan(np.deg2rad(pose['fov'] / 2))
            with np.errstate(all='ignore'):
                raw = np.array([m.boundary([xs[i], zs[j]], fr, 0) for fr in pose['frames']]) - obs
            if not np.isfinite(raw).all(): continue
            bias = float(np.clip(-raw[:2].mean(), -1, 1)); res = raw + bias
            t = float(np.sqrt(np.mean(res[:2] ** 2)))
            if best is None or t < best[0]: best = (t, float(np.sqrt(np.mean(res[2] ** 2))))
        if best: tr[j, i], ho[j, i] = best
np.save(P / 'edge-map-train.npy', tr); np.save(P / 'edge-map-held.npy', ho)
ok = np.isfinite(tr)
print('min', np.nanmin(tr), 'at', np.unravel_index(np.nanargmin(tr), tr.shape))
for thr in [.3, .4, .5, .75, 1.0]:
    sel = ok & (tr < thr)
    zz, xx = np.nonzero(sel)
    print('thr', thr, 'cells', int(sel.sum()), 'of', int(ok.sum()), 'X range', (xx.min(), xx.max()) if xx.size else None, 'Z range', (zz.min(), zz.max()) if zz.size else None)
