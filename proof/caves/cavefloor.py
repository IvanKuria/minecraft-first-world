"""Feasibility: cobble (rock tile) phase fit on a cave floor right below the player (plane 1.62 below the eye,
unlit + GL_EXP fog: brightness = f*0.8*T + (1-f)*fog, f = exp(-0.2*dist)).  Fog is divided out with the model."""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import grasstrack as g
from popin_obs import frame
ROCK = g.TEX[:16, 16:32, :].mean(axis=2).astype(float)
R64 = g.lookup((np.arange(64) + .5)[None, :] / 64, (np.arange(64) + .5)[:, None] / 64, tile=ROCK); R64 -= R64.mean()
FOGC = (14 + 11 + 10) / 3

def prep(im, v0=200):
    lum = im.astype(float).mean(axis=2)
    ys, xs = np.mgrid[v0:356:2, 30:610:2]
    return xs.ravel().astype(float), ys.ravel().astype(float), lum[ys, xs].ravel()

def obs_defog(yaw, pitch, h, x, y, lum):
    dx, dy, dz = g.dirs(yaw, pitch, x, y)
    ok = dy < -1e-3
    t = np.where(ok, -h / np.where(ok, dy, -1), np.nan)
    dist = t * np.sqrt(dx * dx + dy * dy + dz * dz)
    f = np.exp(-.2 * dist)
    T = (lum - (1 - f) * FOGC) / np.maximum(f * .8, 1e-3)
    ok &= (dist < 6) & np.isfinite(T)
    return ok, t * dx, t * dz, T

def phase(yaw, pitch, h, x, y, lum):
    ok, ox, oz, T = obs_defog(yaw, pitch, h, x, y, lum)
    if ok.sum() < 300: return -1, 0, 0
    o = T[ok] - T[ok].mean()
    ix = np.floor((ox[ok] % 1) * 64).astype(int) % 64; iz = np.floor((oz[ok] % 1) * 64).astype(int) % 64
    G = np.zeros((64, 64)); np.add.at(G, (iz, ix), o)
    C = np.real(np.fft.ifft2(np.conj(np.fft.fft2(G)) * np.fft.fft2(R64)))
    k = np.argmax(C); b, a = divmod(k, 64)
    return float(C.flat[k] / (np.linalg.norm(o) * math.sqrt((R64 ** 2).mean() * ok.sum()) + 1e-9)), a / 64, b / 64

def corr(p, x, y, lum):
    yaw, pitch, h, cx, cz = p
    ok, ox, oz, T = obs_defog(yaw, pitch, h, x, y, lum)
    if ok.sum() < 300: return -1
    pr = g.lookup(cx + ox[ok], cz + oz[ok], tile=ROCK); o = T[ok]
    pr = pr - pr.mean(); o = o - o.mean()
    return float(pr @ o / (np.linalg.norm(pr) * np.linalg.norm(o) + 1e-9))

if __name__ == '__main__':
    for n in map(int, sys.argv[1].split(',')):
        im = frame(n); x, y, lum = prep(im)
        best = []
        for h in (1.62, 1.75, 1.9):
            for yw in np.radians(np.arange(0, 360, 3)):
                for pt in np.radians(np.arange(0, 60, 4)):
                    s, a, b = phase(yw, pt, h, x[::2], y[::2], lum[::2])
                    best.append((s, yw, pt, h, a, b))
        best.sort(key=lambda c: -c[0])
        res = []
        for s, yw, pt, h, a, b in best[:4]:
            q0 = np.array([yw, pt, h, a, b]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .02, .008, .008])])
            r = minimize(lambda q: -corr(q, x, y, lum), q0, method='Nelder-Mead', options={'maxiter': 500, 'initial_simplex': simp})
            res.append((-r.fun, r.x))
        res.sort(key=lambda t: -t[0])
        sc, q = res[0]
        print(n, 'best corr %.3f yaw %.1f pitch %.1f h %.2f frac (%.3f,%.3f); runner-up %.3f' % (sc, math.degrees(q[0]) % 360, math.degrees(q[1]), q[2], q[3] % 1, q[4] % 1, res[1][0]), flush=True)
