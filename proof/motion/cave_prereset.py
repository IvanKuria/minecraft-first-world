"""MOTION lane: unchanged caves/cavefloor.py cobble-floor phase fit applied to the cave frames just BEFORE a reset,
to measure the pre-reset walking velocity (retained by resetPos).  usage: cave_prereset.py n1,n2,... out.json"""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
P = Path(__file__).parent
sys.path.insert(0, str(P.parent / 'caves'))
import popin_obs
popin_obs.FR = P / 'frames'; popin_obs.FR.mkdir(exist_ok=True)
import cavefloor as cf
res = []
for n in map(int, sys.argv[1].split(',')):
    im = popin_obs.frame(n); x, y, lum = cf.prep(im)
    best = []
    for h in (1.62, 1.75):
        for yw in np.radians(np.arange(0, 360, 3)):
            for pt in np.radians(np.arange(-8, 60, 4)):
                s, a, b = cf.phase(yw, pt, h, x[::2], y[::2], lum[::2])
                best.append((s, yw, pt, h, a, b))
    best.sort(key=lambda c: -c[0])
    out = []
    for s, yw, pt, h, a, b in best[:4]:
        q0 = np.array([yw, pt, h, a, b]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .02, .008, .008])])
        r = minimize(lambda q: -cf.corr(q, x, y, lum), q0, method='Nelder-Mead', options={'maxiter': 500, 'initial_simplex': simp})
        out.append((-r.fun, r.x.tolist()))
    out.sort(key=lambda t: -t[0])
    sc, q = out[0]
    s_, c_, cp = math.sin(q[0]), math.cos(q[0]), math.cos(q[1])
    rec = {'n': n, 'score': sc, 'yaw_deg': math.degrees(q[0]) % 360, 'pitch_deg': math.degrees(q[1]), 'h': q[2],
           'cam_frac': [q[3] % 1, q[4] % 1], 'player_frac': [(q[3] + .3 * s_ * cp) % 1, (q[4] - .3 * c_ * cp) % 1],
           'runner_up': [[o[0], (math.degrees(o[1][0]) % 360)] for o in out[1:3]]}
    res.append(rec); (P / sys.argv[2]).write_text(json.dumps(res, indent=1))
    print(n, '%.3f yaw %.1f pitch %.1f h %.2f player (%.3f,%.3f) runner %.3f' % (sc, rec['yaw_deg'], rec['pitch_deg'], q[2], *rec['player_frac'], out[1][0]), flush=True)
