"""Sensitivity of the same-height opening world-edge fit to the eye height above the far plane (fits give 0.60-0.64)."""
import sys, json, copy
from pathlib import Path
import numpy as np
import edge_rows as E
m = E.m
base = copy.deepcopy(m.poses)
NAMED = [99100327271809, 266841333641652, 88325364808660]
res = {}
for dh in (-.06, -.03, 0, .03, .06):
    for v, pose in enumerate(m.poses):
        for k, fr in enumerate(pose['frames']):
            fr['camera_parameters'][2] = base[v]['frames'][k]['camera_parameters'][2] + dh
    grid = [(x + .096, z + .804) for x in range(0, 256, 4) for z in range(0, 256, 4)]
    vals = np.array([(E.score(list(p), deltas=(0,))[0] or {'train_rmse_px': np.inf})['train_rmse_px'] for p in grid])
    i = int(np.argmin(vals)); b = vals[i]; near = [grid[k] for k in np.nonzero(vals < b + .1)[0]]
    nm = {}
    for s in NAMED:
        x0, z0, _ = E.reset_xz(s); nm[str(s)] = E.score([x0, z0], deltas=(0,))[0]['train_rmse_px']
    res[dh] = {'best': float(b), 'best_xz': grid[i], 'x': [min(p[0] for p in near), max(p[0] for p in near)], 'z': [min(p[1] for p in near), max(p[1] for p in near)], 'named': nm}
    print('dh %+.2f' % dh, 'best %.3f at' % b, [round(v, 1) for v in grid[i]], 'region x', [round(v) for v in res[dh]['x']], 'z', [round(v) for v in res[dh]['z']], {k[:6]: round(v, 2) for k, v in nm.items()}, flush=True)
(Path(__file__).parent / 'edge-height-sensitivity.json').write_text(json.dumps(res, indent=1))
