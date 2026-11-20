"""Is there a reset at ~7.3 s (frame ~142)?  Fit camera height above the grass plane for frames after the view change,
searching heights up to 12 blocks (a fresh resetPos puts the eye ~10.7 above the plain)."""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import grasstrack as g
from popin_obs import frame
P = Path(__file__).parent
def fit(n, hs):
    im = frame(n); x, y, obs = g.roi(im, v0=20)
    sub = np.arange(len(x)) % 3 == 0
    cands = []
    for h in hs:
        for yw in np.radians(np.arange(0, 360, 3.0)):
            for pt in np.radians(np.arange(10, 80, 4.0)):
                s, a, b = g.phase_search(yw, pt, h, x[sub], y[sub], obs[sub])
                cands.append((s, yw, pt, h, a, b))
    cands.sort(key=lambda c: -c[0])
    best = None
    for s, yw, pt, h, a, b in cands[:4]:
        q0 = np.array([yw, pt, h, a, b]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .05, .008, .008])])
        r = minimize(lambda q: -g.corr_model(q, x, y, obs), q0, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
        if best is None or r.fun < best.fun: best = r
    q = best.x; s_, c_, sp, cp = math.sin(q[0]), math.cos(q[0]), math.sin(q[1]), math.cos(q[1])
    return {'n': n, 'score': -best.fun, 'yaw_deg': math.degrees(q[0]) % 360, 'pitch_deg': math.degrees(q[1]), 'h': q[2],
            'player_frac': [(q[3] + .3 * s_ * cp) % 1, (q[4] - .3 * c_ * cp) % 1]}
if __name__ == '__main__':
    out = P / sys.argv[2]; res = []
    for n in map(int, sys.argv[1].split(',')):
        r = fit(n, [2.5, 4.5, 6.5, 8.5, 10.5])
        res.append(r); out.write_text(json.dumps(res, indent=1))
        print(n, round(r['score'], 3), 'yaw', round(r['yaw_deg'], 1), 'pitch', round(r['pitch_deg'], 1), 'h', round(r['h'], 2), [round(v, 3) for v in r['player_frac']], flush=True)
