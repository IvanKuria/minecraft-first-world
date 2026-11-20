"""Per-frame grass fit with a FREE pitch grid (no horizon assumption; trench walls break the horizon rule) and plane
heights 0.62/1.62/2.62 below the camera; full yaw circle.  Output player position mod 1 for continuity unwrapping."""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import grasstrack as g
from popin_obs import frame
P = Path(__file__).parent
def fit(n, hs=(.62, 1.62, 2.62)):
    im = frame(n); x, y, obs = g.roi(im, v0=60)
    if len(x) < 400: return None
    sub = np.arange(len(x)) % 3 == 0
    cands = []
    for h in hs:
        for yw in np.radians(np.arange(0, 360, 4.0)):
            for pt in np.radians(np.arange(4, 66, 4.0)):
                s, a, b = g.phase_search(yw, pt, h, x[sub], y[sub], obs[sub])
                cands.append((s, yw, pt, h, a, b))
    cands.sort(key=lambda c: -c[0])
    best = None; alts = []
    for s, yw, pt, h, a, b in cands[:3]:
        q0 = np.array([yw, pt, h, a, b]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .02, .008, .008])])
        r = minimize(lambda q: -g.corr_model(q, x, y, obs), q0, method='Nelder-Mead', options={'maxiter': 450, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
        alts.append(-r.fun)
        if best is None or r.fun < best.fun: best = r
    q = best.x; s_, c_, sp, cp = math.sin(q[0]), math.cos(q[0]), math.sin(q[1]), math.cos(q[1])
    return {'n': n, 'score': -best.fun, 'yaw_deg': math.degrees(q[0]) % 360, 'pitch_deg': math.degrees(q[1]), 'h': q[2],
            'player_frac': [(q[3] + .3 * s_ * cp) % 1, (q[4] - .3 * c_ * cp) % 1], 'npx': int(len(x))}
if __name__ == '__main__':
    n0, n1 = int(sys.argv[1]), int(sys.argv[2]); out = P / sys.argv[3]; res = []
    for n in range(n0, n1):
        r = fit(n)
        if r is None: res.append({'n': n, 'score': None}); print(n, 'no roi', flush=True)
        else:
            res.append(r); print(n, round(r['score'], 3), round(r['yaw_deg'], 1), round(r['pitch_deg'], 1), round(r['h'], 2), [round(v, 3) for v in r['player_frac']], flush=True)
        out.write_text(json.dumps(res))
