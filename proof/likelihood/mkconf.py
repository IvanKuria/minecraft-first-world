"""Confusion tables + scorer params for a fixture key (same code path as fx_rank.py; calibration fixtures only)."""
import json, sys
from pathlib import Path
import numpy as np, lobsio as O
HERE = Path(__file__).resolve().parent
key, fam, T = sys.argv[1], sys.argv[2], float(sys.argv[3])
FX = HERE / 'fixtures' / key
cnt = {w: np.zeros((O.NCLS, O.LD + 1)) for w in ('r1', 'r2')}
fs = sorted((FX / 'calib').glob('fx*.json'))
for f in fs:
    rec = json.loads(f.read_text())
    for w, m in rec['maps'].items():
        for (x, z), c in O.fixture_classes(m).items(): cnt[w][c, m['truth'].get(f'{x},{z}', 8)] += 1
conf, lnull, p0s = {}, {}, {}
for w in cnt:
    c = cnt[w] + 0.5; pcd = c / c.sum(0, keepdims=True); p0 = cnt[w].sum(0) / cnt[w].sum()
    conf[w] = pcd.tolist(); p0s[w] = p0.tolist(); lnull[w] = np.log(pcd @ p0).tolist()
cp = FX / 'conf.json'
if not cp.exists(): cp.write_text(json.dumps({'conf': conf, 'lnull': lnull, 'p0': p0s, 'counts': {w: cnt[w].tolist() for w in cnt}, 'n_fixtures': len(fs)}))
cal = json.loads((HERE / f'calib2-{fam}.json').read_text())['params']
K = int(key.split('-K')[1].split('-')[0]); nt = int(key.split('-nt')[1].split('-')[0])
work = HERE / 'work' / f'fx-{key}-test'; work.mkdir(parents=True, exist_ok=True)
(work / 'params.txt').write_text(''.join(f'{k} {cal[k]}\n' for k in ('a', 'b', 'p', 'su', 'sv', 'lam', 'kap')) + f'T {T}\nytop 63\nmargin 0\nK {K}\nnt {nt}\ntf 0\nrmax1 12\n')
print(key, 'calibration fixtures', len(fs))
