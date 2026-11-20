"""Robustness of the same-height (plain = top of world) opening world-edge registration: pose groups, column halves,
alternative edge extraction, single frames.  Reports the best-fit region and the named rows' RMSE per variant."""
import sys, json
from pathlib import Path
import numpy as np
EDGE = Path(__file__).resolve().parents[2] / 'geometry/opening-edge'
sys.path.insert(0, str(EDGE)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'link'))
import fit_opening as m
from lcg import reset_xz
obs_l = np.array(json.loads((EDGE / 'fit-results.json').read_text())['observed_edge_rows'])
obs_c = np.array(json.loads((EDGE / 'sensitivity.json').read_text())['alternative_color_midpoint_rows'])
NAMED = [99100327271809, 266841333641652, 233513965841111, 14593523350806, 170375616623189, 128528575296967]
named_pos = {s: reset_xz(s)[:2] for s in NAMED}
xs_cols = m.xs

def rmse(pos, obs, variants, cols, frames_fit):
    best = None
    for v in variants:
        pose = m.poses[v]; m.F = 180 / np.tan(np.deg2rad(pose['fov'] / 2))
        with np.errstate(all='ignore'):
            raw = np.array([m.boundary(pos, fr, 0) for fr in pose['frames']]) - obs
        raw = raw[:, cols]
        if not np.isfinite(raw[frames_fit]).all(): continue
        bias = float(np.clip(-raw[frames_fit].mean(), -1, 1)); res = raw[frames_fit] + bias
        t = float(np.sqrt(np.mean(res ** 2)))
        if best is None or t < best: best = t
    return best if best is not None else np.inf

fixed = [i for i, p in enumerate(m.poses) if p['mode'] == 'fixed70']; free = [i for i, p in enumerate(m.poses) if p['mode'] != 'fixed70']
allc = np.arange(len(xs_cols)); left = allc[xs_cols < 320]; right = allc[xs_cols >= 320]
VARIANTS = {
    'frozen(all poses, luma, frames39+41)': (obs_l, fixed + free, allc, [0, 1]),
    'fixed70 only': (obs_l, fixed, allc, [0, 1]),
    'free-FOV only': (obs_l, free, allc, [0, 1]),
    'left half columns': (obs_l, fixed + free, left, [0, 1]),
    'right half columns': (obs_l, fixed + free, right, [0, 1]),
    'colour-midpoint edges': (obs_c, fixed + free, allc, [0, 1]),
    'frame 39 only': (obs_l, fixed + free, allc, [0]),
    'frame 44 only': (obs_l, fixed + free, allc, [2]),
}
step = int(sys.argv[1]) if len(sys.argv) > 1 else 4
grid = [(x + .096, z + .804) for x in range(0, 256, step) for z in range(0, 256, step)]
out = {}
for name, (obs, var, cols, ff) in VARIANTS.items():
    vals = np.array([rmse(list(p), obs, var, cols, ff) for p in grid])
    b = np.nanmin(vals); i = int(np.nanargmin(vals))
    near = [grid[k] for k in np.nonzero(vals < b + .1)[0]]
    xr = (min(p[0] for p in near), max(p[0] for p in near)); zr = (min(p[1] for p in near), max(p[1] for p in near))
    q = {s: rmse(list(named_pos[s]), obs, var, cols, ff) for s in NAMED}
    frac_better = {s: float(np.mean(vals < q[s])) for s in NAMED}
    out[name] = {'best_rmse': float(b), 'best_xz': grid[i], 'region_best_plus_0.1px': {'x': xr, 'z': zr, 'n': len(near)},
                 'named_rmse': {str(s): q[s] for s in NAMED}, 'named_fraction_of_world_grid_fitting_better': {str(s): frac_better[s] for s in NAMED}}
    print(name, '| best', round(b, 3), [round(v, 1) for v in grid[i]], '| region(+0.1px) x', [round(v) for v in xr], 'z', [round(v) for v in zr],
          '| named', {str(s)[:6]: (round(q[s], 2), round(frac_better[s], 2)) for s in NAMED}, flush=True)
(Path(__file__).parent / 'edge-robustness.json').write_text(json.dumps(out, indent=1))
