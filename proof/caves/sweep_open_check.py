"""Cross-view consistency of the opening sweep: labels from frames 39/41/44 (yaw ~316) vs frames 48/49/51 (yaw 358-67),
near field only (column centre within 10 blocks of the player)."""
import sys, json, math
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'terrace'))
import sweep as S
from popin_obs import frame
d = json.loads((HERE / 'sweep-opening.json').read_text())
fx, fz = d['phase']
per = {}
for n, yaw, pitch, h in d['frames']:
    per[n] = S.frame_scores(Image.fromarray(frame(n)[:, :, ::-1].copy()), yaw, pitch, h, (fx, fz))
A = S.assign(S.combine([per[n] for n in (39, 41, 44)])); B = S.assign(S.combine([per[n] for n in (48, 49, 51)]))
near = lambda c: math.hypot(c[0] + .5 - fx, c[1] + .5 - fz) < 10
a = {c: v[0] for c, v in A.items() if near(c)}; b = {c: v[0] for c, v in B.items() if near(c)}
common = sorted(set(a) & set(b))
agree = sum(a[c] == b[c] for c in common)
print('near-field labels: A(316)', len(a), 'B(358-67)', len(b), 'common', len(common), 'agree', agree)
for c in common:
    if a[c] != b[c]: print('  disagree', c, a[c], b[c])
allc = {**a, **b}
for c in common: allc[c] = a[c] if a[c] == b[c] else None
xs = [c[0] for c in allc]; zs = [c[1] for c in allc]
for z in range(min(zs), max(zs) + 1):
    print('z=%3d ' % z + ''.join('%3s' % ('?' if (x, z) in allc and allc[(x, z)] is None else (allc[(x, z)] if (x, z) in allc else '.')) for x in range(min(xs), max(xs) + 1)))
print('x from', min(xs), 'to', max(xs))
json.dump({'A_yaw316': [[c[0], c[1], k] for c, k in sorted(a.items())], 'B_yaw358_67': [[c[0], c[1], k] for c, k in sorted(b.items())],
           'common': len(common), 'agree': agree}, open(HERE / 'sweep-opening-crossview.json', 'w'))
