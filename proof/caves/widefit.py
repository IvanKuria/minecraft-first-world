"""Wide single-frame grass fit: pitch from the measured horizon (+-2 deg), yaw full circle, h grid; FFT phase; refine."""
import sys, json, math
import numpy as np
from scipy.optimize import minimize
import grasstrack as g
from popin_obs import frame, skymask
def horizon_pitch(im):
    s = skymask(im); t = ~s
    rows = [np.argmax(t[:, u]) for u in range(40, 600, 8) if t[:, u].any()]
    v = np.median(rows) - 1.2          # far world edge sits ~1-2 px below the horizon (eye 0.62 above plain)
    return math.atan((179.5 - v) / g.F), v
def wide(n, v0=None, hs=(.62, 1.62, 2.62), yaw_step=2.0, top=6):
    im = frame(n); x, y, obs = g.roi(im, v0=v0 if v0 is not None else 150)
    p0, vh = horizon_pitch(im)
    sub = np.arange(len(x)) % 2 == 0
    cands = []
    for h in hs:
        for yw in np.radians(np.arange(0, 360, yaw_step)):
            for dp in np.radians([-1.5, -.5, .5, 1.5]):
                s, a, b = g.phase_search(yw, p0 + dp, h, x[sub], y[sub], obs[sub])
                cands.append((s, yw, p0 + dp, h, a, b))
    cands.sort(key=lambda c: -c[0])
    out = []
    for s, yw, pt, h, a, b in cands[:top]:
        q0 = np.array([yw, pt, h, a, b]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .02, .008, .008])])
        r = minimize(lambda q: -g.corr_model(q, x, y, obs), q0, method='Nelder-Mead', options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
        out.append((-r.fun, r.x.tolist(), s))
    out.sort(key=lambda o: -o[0])
    return out, vh, len(x)
if __name__ == '__main__':
    for n in map(int, sys.argv[1].split(',')):
        out, vh, npx = wide(n)
        print('frame', n, 'horizon row', round(vh, 1), 'roi px', npx)
        for sc, q, s0 in out[:4]:
            s_, c_, sp, cp = math.sin(q[0]), math.cos(q[0]), math.sin(q[1]), math.cos(q[1])
            print('  score %.3f yaw %.1f pitch %.2f h %.3f cam (%.3f,%.3f) player (%.3f,%.3f) fft %.3f' % (sc, math.degrees(q[0]) % 360, math.degrees(q[1]), q[2], q[3] % 1, q[4] % 1, (q[3] + .3 * s_ * cp) % 1, (q[4] - .3 * c_ * cp) % 1, s0), flush=True)
