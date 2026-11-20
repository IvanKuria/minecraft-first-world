"""MOTION lane: brute-force single-frame grass fit for HIGH camera frames (falling after a reset), caves/grasstrack
model unchanged: yaw full circle, pitch and camera height grids, FFT phase, Nelder-Mead refine of the top candidates.
usage: widefit_high.py n1,n2 out.json [hmin hmax]"""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
P = Path(__file__).parent
sys.path.insert(0, str(P.parent / 'caves'))
import popin_obs
popin_obs.FR = P / 'frames'; popin_obs.FR.mkdir(exist_ok=True)
import grasstrack as g
hmin, hmax = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (5.0, 11.0)
res = []
for n in map(int, sys.argv[1].split(',')):
    im = popin_obs.frame(n); x, y, obs = g.roi(im, v0=150)
    sub = np.arange(len(x)) % 3 == 0
    c = []
    for h in np.arange(hmin, hmax + 1e-9, .5):
        for yw in np.radians(np.arange(0, 360, 3)):
            for pt in np.radians(np.arange(10, 72, 3)):
                s, a, b = g.phase_search(yw, pt, h, x[sub], y[sub], obs[sub])
                c.append((s, yw, pt, h, a, b))
    c.sort(key=lambda t: -t[0])
    outs = []
    for s, yw, pt, h, a, b in c[:6]:
        q0 = np.array([yw, pt, h, a, b]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .02, .008, .008])])
        r = minimize(lambda q: -g.corr_model(q, x, y, obs), q0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
        outs.append((-r.fun, r.x.tolist()))
    outs.sort(key=lambda t: -t[0])
    sc, q = outs[0]
    s_, c_, cp = math.sin(q[0]), math.cos(q[0]), math.cos(q[1])
    rec = {'n': n, 'score': sc, 'params': q, 'yaw_deg': math.degrees(q[0]) % 360, 'pitch_deg': math.degrees(q[1]), 'h': q[2],
           'player_frac': [(q[3] + .3 * s_ * cp) % 1, (q[4] - .3 * c_ * cp) % 1],
           'alternatives': [[o[0], math.degrees(o[1][0]) % 360, o[1][2]] for o in outs[1:4]]}
    res.append(rec); (P / sys.argv[2]).write_text(json.dumps(res, indent=1))
    print(n, '%.3f yaw %.1f pitch %.1f h %.2f player (%.4f,%.4f) | alt %s' % (sc, rec['yaw_deg'], rec['pitch_deg'], q[2], *rec['player_frac'], [[round(v, 3) for v in a] for a in rec['alternatives']]), flush=True)
