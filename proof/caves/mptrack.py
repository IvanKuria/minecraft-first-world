"""Multi-plane grass-texture tracker.  Pose = (yaw, pitch, E, cx, cz): E = camera height above the PLATEAU top plane
(plane level 0 = the plain); planes j = 0..4 blocks below it are candidate top-face levels.  All planes share the
integer block grid, so their FFT phase-correlation surfaces are summed; refinement maximises the mean over image patches
of the best-plane correlation.  Camera conventions = generator/opening-texture/model.py (player = camera + 0.3*fwd)."""
import sys, json, math, os
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import minimize
import grasstrack as g
from popin_obs import frame
P = Path(__file__).parent
JS = range(0, 5)
PATCH = 16

def visible_planes(E):
    return [j for j in JS if E + j > .08]

def multi_phase(yaw, pitch, E, x, y, obs):
    o = obs - obs.mean()
    Ctot = np.zeros((g.NB, g.NB)); used = 0
    for j in visible_planes(E):
        ox, oz = g.ground(yaw, pitch, E + j, x, y)
        ok = np.isfinite(ox) & (np.hypot(ox, oz) < 20)
        if ok.sum() < 100: continue
        oo = o[ok] - o[ok].mean()
        ix = np.floor((ox[ok] % 1) * g.NB).astype(int) % g.NB; iz = np.floor((oz[ok] % 1) * g.NB).astype(int) % g.NB
        G = np.zeros((g.NB, g.NB)); np.add.at(G, (iz, ix), oo)
        C = np.real(np.fft.ifft2(np.conj(np.fft.fft2(G)) * np.fft.fft2(g.T64)))
        Ctot += C / (np.linalg.norm(oo) * math.sqrt((g.T64 ** 2).mean() * ok.sum()) + 1e-9); used += 1
    if not used: return -1, 0, 0
    k = np.argmax(Ctot); b, a = divmod(k, g.NB)
    return float(Ctot.flat[k] / used), a / g.NB, b / g.NB

class Frame:
    def __init__(self, im, v0, v1=356, step=2):
        x, y, obs = g.roi(im, v0=v0, v1=v1, step=step)
        self.x, self.y, self.obs = x, y, obs
        pid = (y // PATCH).astype(int) * 100 + (x // PATCH).astype(int)
        u, inv = np.unique(pid, return_inverse=True)
        cnt = np.bincount(inv)
        keep = cnt[inv] >= 12
        self.x, self.y, self.obs, inv = x[keep], y[keep], obs[keep], inv[keep]
        u2, self.inv = np.unique(inv, return_inverse=True)
        self.np = len(u2)

def multi_corr(p, fr, detail=False):
    yaw, pitch, E, cx, cz = p
    js = visible_planes(E)
    if not js: return (-1.0, None) if detail else -1.0
    x, y, obs, inv, n = fr.x, fr.y, fr.obs, fr.inv, fr.np
    cnt = np.bincount(inv, minlength=n).astype(float)
    mo = np.bincount(inv, obs, n) / np.maximum(cnt, 1)
    od = obs - mo[inv]
    so = np.sqrt(np.bincount(inv, od * od, n))
    best = np.full(n, -1.0); bj = np.full(n, -1)
    for j in js:
        ox, oz = g.ground(yaw, pitch, E + j, x, y)
        ok = np.isfinite(ox)
        pr = np.where(ok, g.lookup(cx + np.where(ok, ox, 0), cz + np.where(ok, oz, 0)), np.nan)
        okc = np.bincount(inv, ok.astype(float), n)
        prz = np.where(ok, pr, 0)
        mp = np.bincount(inv, prz, n) / np.maximum(okc, 1)
        pd = np.where(ok, pr - mp[inv], 0)
        num = np.bincount(inv, od * pd, n); sp = np.sqrt(np.bincount(inv, pd * pd, n))
        c = num / (so * sp + 1e-9)
        c = np.where(okc >= cnt * .9, c, -1)
        upd = c > best; best = np.where(upd, c, best); bj = np.where(upd, j, bj)
    w = cnt
    score = float((np.clip(best, 0, None) * w).sum() / w.sum())
    return (score, bj) if detail else score

def track(n0, n1, p, out, v0=60, notes=None):
    res = []; prev = list(p)
    for n in range(n0, n1):
        im = frame(n); fr = Frame(im, v0, step=3)
        if fr.np < 8:
            res.append({'n': n, 'ok': False}); print(n, 'no roi', flush=True); continue
        Es = sorted({round(prev[2], 3), round(prev[2] - 1, 3), round(prev[2] + 1, 3)})
        cands = []
        sub = np.arange(len(fr.x)) % 2 == 0
        for E in Es:
            for dy in np.arange(-4.5, 4.51, 1.5):
                for dp in np.arange(-4.5, 4.51, 1.5):
                    yw, pt = prev[0] + math.radians(dy), prev[1] + math.radians(dp)
                    s, a, b = multi_phase(yw, pt, E, fr.x[sub], fr.y[sub], fr.obs[sub])
                    cands.append((s, yw, pt, E, a, b))
        cands.sort(key=lambda c: -c[0])
        best = None
        for s, yw, pt, E, a, b in cands[:2]:
            cx = prev[3] + ((a - prev[3]) + .5) % 1 - .5; cz = prev[4] + ((b - prev[4]) + .5) % 1 - .5
            p0 = np.array([yw, pt, E, cx, cz]); simp = np.vstack([p0, p0 + np.diag([.005, .005, .02, .006, .006])])
            r = minimize(lambda q: -multi_corr(q, fr), p0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
            mv = math.hypot(r.x[3] - prev[3], r.x[4] - prev[4])
            key = -r.fun - (.3 if mv > .4 else 0)
            if best is None or key > best[0]: best = (key, r, mv)
        _, r, mv = best; q = r.x.tolist(); sc = -r.fun
        sc_, bj = multi_corr(q, fr, detail=True)
        s_, c_, sp, cp = math.sin(q[0]), math.cos(q[0]), math.sin(q[1]), math.cos(q[1])
        player = [q[3] + .3 * s_ * cp, q[4] - .3 * c_ * cp]
        hist = np.bincount(bj[bj >= 0], minlength=5).tolist()
        rec = {'n': n, 'ok': True, 'score': sc, 'move': mv, 'yaw_deg': math.degrees(q[0]) % 360, 'pitch_deg': math.degrees(q[1]),
               'E': q[2], 'cam': q[3:5], 'player': player, 'plane_patch_hist': hist}
        res.append(rec); out.write_text(json.dumps(res))
        print(n, round(sc, 3), 'mv', round(mv, 3), 'yaw', round(rec['yaw_deg'], 1), 'pit', round(rec['pitch_deg'], 1), 'E', round(q[2], 2), 'pl', [round(v, 3) for v in player], hist, flush=True)
        if sc > .25 and mv < .4: prev = q
    return res

if __name__ == '__main__':
    n0, n1 = int(sys.argv[1]), int(sys.argv[2]); p = [float(v) for v in sys.argv[3].split(',')]
    track(n0, n1, p, P / sys.argv[4], v0=int(os.environ.get('ROI_V0', '60')))
