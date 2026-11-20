"""Lit cobble (rock tile, no fog) phase fits inside the R1 pit, free plane height; tests whether the pit floor supports
continuity tracking from the registered pit edge (frame 1744)."""
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import grasstrack as g
from popin_obs import frame
ROCK = g.TEX[:16, 16:32, :].mean(axis=2).astype(float)
R64 = g.lookup((np.arange(64) + .5)[None, :] / 64, (np.arange(64) + .5)[:, None] / 64, tile=ROCK); R64 -= R64.mean()
P = Path(__file__).parent
def pix(im, lmin=110):
    a = im.astype(float); b, gg, r = a[..., 0], a[..., 1], a[..., 2]; lum = (b + gg + r) / 3
    m = (np.abs(r - gg) < 18) & (np.abs(gg - b) < 18) & (lum > lmin)
    ys, xs = np.nonzero(m[::2, ::2]); ys *= 2; xs *= 2
    return xs.astype(float), ys.astype(float), lum[ys, xs]
def phase(yaw, pitch, h, x, y, o):
    ox, oz = g.ground(yaw, pitch, h, x, y); ok = np.isfinite(ox) & (np.hypot(ox, oz) < 20)
    if ok.sum() < 300: return -1, 0, 0
    oo = o[ok] - o[ok].mean()
    ix = np.floor((ox[ok] % 1) * 64).astype(int) % 64; iz = np.floor((oz[ok] % 1) * 64).astype(int) % 64
    G = np.zeros((64, 64)); np.add.at(G, (iz, ix), oo)
    C = np.real(np.fft.ifft2(np.conj(np.fft.fft2(G)) * np.fft.fft2(R64)))
    k = np.argmax(C); b, a = divmod(k, 64)
    return float(C.flat[k] / (np.linalg.norm(oo) * math.sqrt((R64 ** 2).mean() * ok.sum()) + 1e-9)), a / 64, b / 64
def corr(p, x, y, o):
    yaw, pitch, h, cx, cz = p
    if not (.3 < h < 30): return -1
    ox, oz = g.ground(yaw, pitch, h, x, y); ok = np.isfinite(ox) & (np.hypot(ox, oz) < 20)
    if ok.sum() < 300: return -1
    pr = g.lookup(cx + ox[ok], cz + oz[ok], tile=ROCK); oo = o[ok]; pr = pr - pr.mean(); oo = oo - oo.mean()
    return float(pr @ oo / (np.linalg.norm(pr) * np.linalg.norm(oo) + 1e-9))
if __name__ == '__main__':
    out = []
    for n in map(int, sys.argv[1].split(',')):
        x, y, o = pix(frame(n))
        cands = []
        for h in (1.62, 2.62, 3.62, 4.62, 5.62, 6.62):
            for yw in np.radians(np.arange(0, 360, 4)):
                for pt in np.radians(np.arange(10, 80, 5)):
                    s, a, b = phase(yw, pt, h, x[::2], y[::2], o[::2]); cands.append((s, yw, pt, h, a, b))
        cands.sort(key=lambda c: -c[0]); best = None
        for s, yw, pt, h, a, b in cands[:4]:
            q0 = np.array([yw, pt, h, a, b]); simp = np.vstack([q0, q0 + np.diag([.006, .006, .03, .008, .008])])
            r = minimize(lambda q: -corr(q, x, y, o), q0, method='Nelder-Mead', options={'maxiter': 500, 'initial_simplex': simp})
            if best is None or r.fun < best.fun: best = r
        q = best.x; s_, c_, sp, cp = math.sin(q[0]), math.cos(q[0]), math.sin(q[1]), math.cos(q[1])
        rec = {'n': n, 'score': -best.fun, 'yaw_deg': math.degrees(q[0]) % 360, 'pitch_deg': math.degrees(q[1]), 'h_rock': q[2],
               'player_frac': [(q[3] + .3 * s_ * cp) % 1, (q[4] - .3 * c_ * cp) % 1], 'npx': int(len(x))}
        out.append(rec); print(n, 'rock fit %.3f yaw %.1f pitch %.1f h_rock %.2f frac (%.3f,%.3f) px %d' % (rec['score'], rec['yaw_deg'], rec['pitch_deg'], q[2], *rec['player_frac'], len(x)), flush=True)
        (P / sys.argv[2]).write_text(json.dumps(out, indent=1))
