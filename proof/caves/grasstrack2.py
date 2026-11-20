"""Tracker v2: continuity unwrap against the previous position only, candidate planes h, h+-1, adaptive rotation search,
max plausible motion per video frame 0.35 block (walk 0.18, fall/jump drift similar)."""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import grasstrack as g
from popin_obs import frame
P = Path(__file__).parent

def candidates(im, pred, rot_half, rot_step, hs, x, y, obs, sub):
    out = []
    for h in hs:
        for dyw in np.arange(-rot_half, rot_half + 1e-9, rot_step):
            for dpt in np.arange(-rot_half, rot_half + 1e-9, rot_step):
                yw = pred[0] + math.radians(dyw); pt = pred[1] + math.radians(dpt)
                s, a, b = g.phase_search(yw, pt, h, x[sub], y[sub], obs[sub])
                out.append((s, yw, pt, h, a, b))
    out.sort(key=lambda c: -c[0])
    return out

def refine(c, prev, x, y, obs):
    s, yw, pt, h, a, b = c
    cx = prev[3] + ((a - prev[3]) + .5) % 1 - .5; cz = prev[4] + ((b - prev[4]) + .5) % 1 - .5
    p0 = np.array([yw, pt, h, cx, cz]); simp = np.vstack([p0, p0 + np.diag([.006, .006, .02, .008, .008])])
    r = minimize(lambda p: -g.corr_model(p, x, y, obs), p0, method='Nelder-Mead',
                 options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
    return r

def step(n, prev):
    im = frame(n)
    x, y, obs = g.roi(im)
    if len(x) < 300: return None
    sub = np.arange(len(x)) % 2 == 0
    hs = sorted({round(prev[2], 3), round(prev[2] + 1, 3)} | ({round(prev[2] - 1, 3)} if prev[2] > 1.3 else set()))
    cs = candidates(im, prev, 6, 1.5, hs, x, y, obs, sub)
    if cs[0][0] < .35:
        cs = candidates(im, prev, 18, 3, hs, x, y, obs, sub) + cs
        cs.sort(key=lambda c: -c[0])
    best = None
    for c in cs[:4]:
        r = refine(c, prev, x, y, obs)
        mv = math.hypot(r.x[3] - prev[3], r.x[4] - prev[4])
        key = -r.fun - (0.5 if mv > .35 else 0)
        if best is None or key > best[0]: best = (key, r, mv)
    return best[1], best[2], len(x)

if __name__ == '__main__':
    import os
    g.ROI_V0 = int(os.environ.get('ROI_V0', '200'))
    n0, n1 = int(sys.argv[1]), int(sys.argv[2]); prev = [float(v) for v in sys.argv[3].split(',')]
    out = P / sys.argv[4]; res = []
    for n in range(n0, n1):
        st = step(n, prev)
        if st is None:
            res.append({'n': n, 'ok': False}); print(n, 'no grass roi', flush=True); out.write_text(json.dumps(res)); continue
        r, mv, npx = st; p = r.x.tolist(); sc = -r.fun
        s_, c_, sp, cp = math.sin(p[0]), math.cos(p[0]), math.sin(p[1]), math.cos(p[1])
        player = [p[3] + .3 * s_ * cp, p[4] - .3 * c_ * cp]
        rec = {'n': n, 'ok': True, 'score': sc, 'move': mv, 'npx': npx, 'yaw_deg': math.degrees(p[0]) % 360, 'pitch_deg': math.degrees(p[1]), 'h': p[2], 'cam': p[3:5], 'player': player}
        res.append(rec); out.write_text(json.dumps(res))
        print(n, round(sc, 3), 'mv', round(mv, 3), 'yaw', round(rec['yaw_deg'], 1), 'pit', round(rec['pitch_deg'], 1), 'h', round(p[2], 2), 'player', [round(v, 3) for v in player], flush=True)
        if sc > .3 and mv < .35: prev = p
