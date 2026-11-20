"""R1 trench walk: multi-plane grass tracker with the eye height FIXED by the R1 landing analysis (feet on the -2 trench
floor => eye 63.62 => plain plane 0.38 ABOVE the eye; visible top planes at 0.62, 1.62, 2.62, 3.62 below the eye).
Positions are in the R1 tracker frame (R1 player at 1668 = (0.468, 0.899))."""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import grasstrack as g
import mptrack as M
from popin_obs import frame
P = Path(__file__).parent
E0 = -0.38

def phase_multi(yaw, pitch, x, y, obs):
    return M.multi_phase(yaw, pitch, E0, x, y, obs)

def fit(n, prev, yaw_half=30, pitch_rng=(-6, 24), step=3, refine=3):
    im = frame(n); fr = M.Frame(im, 0, step=3)
    if fr.np < 8: return None
    sub = np.arange(len(fr.x)) % 2 == 0
    cands = []
    for dyw in np.arange(-yaw_half, yaw_half + 1e-9, step):
        for pt in np.arange(pitch_rng[0], pitch_rng[1] + 1e-9, step):
            yw = prev[0] + math.radians(dyw)
            s, a, b = M.multi_phase(yw, math.radians(pt), E0, fr.x[sub], fr.y[sub], fr.obs[sub])
            cands.append((s, yw, math.radians(pt), a, b))
    cands.sort(key=lambda c: -c[0])
    best = None
    for s, yw, pt, a, b in cands[:refine]:
        cx = prev[1] + ((a - prev[1]) + .5) % 1 - .5; cz = prev[2] + ((b - prev[2]) + .5) % 1 - .5
        q0 = np.array([yw, pt, cx, cz]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .008, .008])])
        f = lambda q: -M.multi_corr([q[0], q[1], E0, q[2], q[3]], fr)
        r = minimize(f, q0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
        mv = math.hypot(r.x[2] - prev[1], r.x[3] - prev[2])
        key = -r.fun - (.2 if mv > .45 else 0)
        if best is None or key > best[0]: best = (key, r, mv)
    return best

if __name__ == '__main__':
    n0, n1 = int(sys.argv[1]), int(sys.argv[2])
    # prev = [yaw, cam_x, cam_z]: start from the fall track's last good frame extrapolated to the landing
    prev = [float(v) for v in sys.argv[3].split(',')]
    out = P / sys.argv[4]; res = []
    for n in range(n0, n1):
        b = fit(n, prev, yaw_half=float(sys.argv[5]) if len(sys.argv) > 5 else 30)
        if b is None: print(n, 'no roi', flush=True); continue
        key, r, mv = b; q = r.x
        s_, c_, sp, cp = math.sin(q[0]), math.cos(q[0]), math.sin(q[1]), math.cos(q[1])
        player = [q[2] + .3 * s_ * cp, q[3] - .3 * c_ * cp]
        rec = {'n': n, 'score': -r.fun, 'move': mv, 'yaw_deg': math.degrees(q[0]) % 360, 'pitch_deg': math.degrees(q[1]), 'cam': [q[2], q[3]], 'player': player}
        res.append(rec); out.write_text(json.dumps(res))
        print(n, round(-r.fun, 3), 'mv', round(mv, 3), 'yaw', round(rec['yaw_deg'], 1), 'pitch', round(rec['pitch_deg'], 1), 'player', [round(v, 3) for v in player], flush=True)
        if -r.fun > .3 and mv < .45: prev = [q[0], q[2], q[3]]
