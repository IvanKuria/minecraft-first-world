"""Score a candidate world against the CAVES constraint files.
Usage: check_world.py <world.blocks (uint8, layout (y*256+z)*256+x, nonzero=solid)> <opening_x> <opening_z> <r1_x> <r1_z>
Reports, per file, the confidence-weighted fraction of satisfied items and the list of violated high-confidence items.
Positions are the candidate's own (x0,z0) and (x1,z1); constraints are relative to their floors.  Freeze any acceptance
rule BEFORE looking at candidate scores; a random-position null (e.g. 200 random positions in the same world) is needed
to interpret a value."""
import sys, json, math
from pathlib import Path
import numpy as np
P = Path(__file__).parent
W = H = 256; D = 64

def load(path):
    b = np.fromfile(path, dtype=np.uint8); assert b.size == W * H * D
    return (b.reshape(D, H, W) != 0)          # [y, z, x]

def solid(w, x, y, z):
    if x < 0 or z < 0 or x >= W or z >= H or y < 0: return None
    if y >= D: return False
    return bool(w[y, z, x])

def top(w, x, z):
    if x < 0 or z < 0 or x >= W or z >= H: return None
    ys = np.nonzero(w[:, z, x])[0]
    return int(ys.max()) if ys.size else -1

def score(w, cons, X, Z):
    bx, bz = math.floor(X), math.floor(Z)
    tot = ok = 0.0; viol = []
    for c in cons.get('cells', []):
        s = solid(w, bx + c['dx'], c['y'], bz + c['dz'])
        if s is None: continue
        good = (s == (c['state'] == 'solid')); tot += c['confidence']; ok += c['confidence'] * good
        if not good and c['confidence'] >= .7: viol.append(('cell', c['dx'], c['y'], c['dz'], c['state']))
    for c in cons.get('column_constraints', []):
        tops = [top(w, bx + dx, bz + dz) for dx, dz in c['columns_dxdz']]
        if any(t is None for t in tops): continue
        rule = c['rule']
        if c['type'] == 'max_top_equals':
            good = max(tops) == c['max_top_solid_y']
        else:
            good = all(t == c['top_solid_y'] for t in tops)
        tot += c['confidence']; ok += c['confidence'] * good
        if not good and c['confidence'] >= .7: viol.append(('col', c['columns_dxdz'], rule, tops))
    return (ok / tot if tot else float('nan')), tot, viol

if __name__ == '__main__':
    w = load(sys.argv[1]); x0, z0, x1, z1 = map(float, sys.argv[2:6])
    for name, (X, Z) in (('opening', (x0, z0)), ('r1', (x1, z1))):
        cons = json.loads((P / f'constraints-{name}.json').read_text())
        s, t, v = score(w, cons, X, Z)
        print(f'{name}: weighted satisfied fraction {s:.3f} (total weight {t:.1f}); high-confidence violations {len(v)}')
        for item in v[:12]: print('   ', item)
