"""Surface statistics of a world file: column-top histogram, plain level (mode), +k/-k fractions, holes, step-edge
density, slits (1-wide), over the interior (excluding 8-block border)."""
import sys, json
import numpy as np
W = H = 256; D = 64

def load(p):
    return np.fromfile(p, np.uint8).reshape(D, H, W)

def tops(b):
    s = b != 0
    # highest solid y per column (lightDepth convention: y>0 scan), -1 if none
    idx = np.where(s.any(0), D - 1 - np.argmax(s[::-1], axis=0), -1)
    return idx  # [z, x]

def stats(b, border=8, deep=4):
    t = tops(b)[border:-border, border:-border]
    vals, cnt = np.unique(t, return_counts=True)
    mode = int(vals[np.argmax(cnt)])
    d = t - mode
    n = d.size
    out = {'mode': mode, 'f0': float((d == 0).mean()), 'f+1': float((d == 1).mean()), 'f+2': float((d == 2).mean()),
           'f+3up': float((d >= 3).mean()), 'f-1': float((d == -1).mean()), 'f-2': float((d == -2).mean()),
           'f-3': float((d == -3).mean()), 'f_deep(<=-%d)' % deep: float((d <= -deep).mean())}
    # step edges: adjacent column pairs (4-neighbour) whose tops differ by exactly 1 (shallow), excluding deep holes
    dx = np.abs(np.diff(t, axis=1)); dz = np.abs(np.diff(t, axis=0))
    out['edge1_density'] = float(((dx == 1).sum() + (dz == 1).sum()) / (dx.size + dz.size))
    out['edge2_density'] = float(((dx == 2).sum() + (dz == 2).sum()) / (dx.size + dz.size))
    out['edge_any_density'] = float(((dx > 0).sum() + (dz > 0).sum()) / (dx.size + dz.size))
    # 1-wide slits: a column lower than both neighbours along x or along z by >=1 (and the neighbours equal level)
    c = t[1:-1, 1:-1]
    sx = (t[1:-1, :-2] > c) & (t[1:-1, 2:] > c)
    sz = (t[:-2, 1:-1] > c) & (t[2:, 1:-1] > c)
    out['slit_density'] = float((sx | sz).mean())
    # 1-wide bumps: column higher than both neighbours along x or z
    bx = (t[1:-1, :-2] < c) & (t[1:-1, 2:] < c)
    bz = (t[:-2, 1:-1] < c) & (t[2:, 1:-1] < c)
    out['bump_density'] = float((bx | bz).mean())
    return out, t

if __name__ == '__main__':
    for p in sys.argv[1:]:
        o, _ = stats(load(p))
        print(p.split('/')[-1], json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}))
