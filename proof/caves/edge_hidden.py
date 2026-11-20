"""Same-height (plain = top) frozen opening world-edge RMSE for every hidden-reset scenario candidate
(runtime/conditional-two-reset/all-hidden-scenario-positions.json, 4,385 state/scenario pairs)."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_rows import score
P = Path(__file__).parent
src = Path(__file__).resolve().parents[2] / 'runtime/conditional-two-reset/all-hidden-scenario-positions.json'
C = json.loads(src.read_text())['candidates']
out = []
for c in C:
    s = score(c['opening'], deltas=(0,))[0]
    out.append({'id': c['id'], 'hidden_before': c['hidden_before'], 'hidden_between': c['hidden_between'],
                'state_before_opening': c['state_before_opening'], 'opening': c['opening'],
                'rmse_same_height': None if s is None else s['train_rmse_px'], 'heldout': None if s is None else s['heldout_rmse_px']})
out.sort(key=lambda r: 1e9 if r['rmse_same_height'] is None else r['rmse_same_height'])
(P / 'edge-hidden-scenarios.json').write_text(json.dumps(out, indent=0))
v = np.array([r['rmse_same_height'] for r in out if r['rmse_same_height'] is not None])
print('n', len(out), 'valid', len(v), 'quantiles', np.percentile(v, [0, 1, 5, 25, 50]).round(3))
for thr in [.3, .35, .4, .5]:
    sel = [r for r in out if r['rmse_same_height'] is not None and r['rmse_same_height'] < thr]
    by = {}
    for r in sel: by.setdefault((r['hidden_before'], r['hidden_between']), 0); by[(r['hidden_before'], r['hidden_between'])] += 1
    print('thr', thr, 'pairs', len(sel), 'by scenario', dict(sorted(by.items())))
for r in out[:15]:
    print(r['id'], r['hidden_before'], r['hidden_between'], [round(x, 2) for x in r['opening']], round(r['rmse_same_height'], 3))
