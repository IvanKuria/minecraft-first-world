"""Per-frame wide grass fits for the opening walk (independent per frame => robust to fast mouse turns); continuity of the
texture phase between consecutive frames then fixes the integer path."""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import grasstrack as g
from widefit import horizon_pitch
from popin_obs import frame
P = Path(__file__).parent
def fit(n, hs=(.62, 1.62, 2.62)):
    im = frame(n)
    p0, vh = horizon_pitch(im)
    x, y, obs = g.roi(im, v0=int(min(300, max(20, vh + 12))))
    if len(x) < 400: return None
    sub = np.arange(len(x)) % 3 == 0
    cands = []
    for h in hs:
        for yw in np.radians(np.arange(0, 360, 3.0)):
            for dp in np.radians([-1.0, 1.0]):
                s, a, b = g.phase_search(yw, p0 + dp, h, x[sub], y[sub], obs[sub])
                cands.append((s, yw, p0 + dp, h, a, b))
    cands.sort(key=lambda c: -c[0])
    best = None
    for s, yw, pt, h, a, b in cands[:3]:
        q0 = np.array([yw, pt, h, a, b]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .02, .008, .008])])
        r = minimize(lambda q: -g.corr_model(q, x, y, obs), q0, method='Nelder-Mead', options={'maxiter': 450, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
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
