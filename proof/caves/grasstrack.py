"""Grass-texture visual odometry on the plain (authentic rd-132211 terrain.png tile 0, model.py camera conventions).
State per frame: yaw, pitch (rad), h (camera height above the fitted grass plane), cx, cz (CAMERA position, continuous;
the texture fixes them mod 1, continuity from frame to frame fixes the integer part).  Player = camera + 0.3*forward.
Phase search: rectify ROI pixels onto the ground plane at 1/64-block resolution and circularly cross-correlate with the
tile (FFT), then Nelder-Mead refine on the exact bilinear model."""
import sys, json, math, time
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import minimize
P = Path(__file__).parent
TEX = cv2.imread(str(P.parent / 'terrace/tex/terrain.png'))
TILE = TEX[:16, :16, 1].astype(float)          # grass top, green channel (BGR index 1 = G)
F = 180 / math.tan(math.radians(35))
SPAN = 15.984
NB = 64

def lookup(wx, wz, tile=TILE):
    u = ((wx % 1) * SPAN - .5) % 16; v = ((wz % 1) * SPAN - .5) % 16
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int); a = u - u0; b = v - v0
    return (tile[v0, u0] * (1 - a) * (1 - b) + tile[v0, (u0 + 1) % 16] * a * (1 - b) +
            tile[(v0 + 1) % 16, u0] * (1 - a) * b + tile[(v0 + 1) % 16, (u0 + 1) % 16] * a * b)

fr = (np.arange(NB) + .5) / NB
T64 = lookup(fr[None, :], fr[:, None])          # [z-row, x-col]
T64 = T64 - T64.mean()
FT = np.conj(np.fft.fft2(T64))

def dirs(yaw, pitch, x, y):
    rx = (x + .5 - 320) / F; ry = (y + .5 - 180) / F
    s, c, sp, cp = np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)
    return c * rx - s * sp * ry + s * cp, -cp * ry - sp, s * rx + c * sp * ry - c * cp

def ground(yaw, pitch, h, x, y):
    dx, dy, dz = dirs(yaw, pitch, x, y)
    with np.errstate(divide='ignore', invalid='ignore'):
        ok = dy < -1e-3
        ox = np.where(ok, -h * dx / dy, np.nan); oz = np.where(ok, -h * dz / dy, np.nan)
    return ox, oz

def corr_model(p, x, y, obs):
    yaw, pitch, h, cx, cz = p
    if not (0.05 < h < 40): return -1.0
    ox, oz = ground(yaw, pitch, h, x, y)
    ok = np.isfinite(ox) & (np.hypot(ox, oz) < 25)
    if ok.sum() < 200: return -1.0
    pr = lookup(cx + ox[ok], cz + oz[ok]); o = obs[ok]
    pr = pr - pr.mean(); o = o - o.mean()
    return float(pr @ o / (np.linalg.norm(pr) * np.linalg.norm(o) + 1e-9))

def phase_search(yaw, pitch, h, x, y, obs):
    """best (cx mod 1, cz mod 1) and score via FFT cross-correlation of rectified observations"""
    ox, oz = ground(yaw, pitch, h, x, y)
    ok = np.isfinite(ox) & (np.hypot(ox, oz) < 25)
    if ok.sum() < 200: return -1, 0, 0
    o = obs[ok] - obs[ok].mean()
    ix = np.floor((ox[ok] % 1) * NB).astype(int) % NB; iz = np.floor((oz[ok] % 1) * NB).astype(int) % NB
    G = np.zeros((NB, NB)); np.add.at(G, (iz, ix), o)
    # C(b,a) = sum G[k,l] T[k+b, l+a]  -> ifft(conj(FFT(G)) * FFT(T))
    C = np.real(np.fft.ifft2(np.conj(np.fft.fft2(G)) * np.fft.fft2(T64)))
    k = np.argmax(C); b, a = divmod(k, NB)
    norm = np.linalg.norm(o) * math.sqrt((T64 ** 2).mean() * ok.sum())
    return float(C.flat[k] / (norm + 1e-9)), a / NB, b / NB

ROI_V0 = 200
def roi(im, v0=None, v1=356, u0=40, u1=600, step=2):
    v0 = ROI_V0 if v0 is None else v0
    ys, xs = np.mgrid[v0:v1:step, u0:u1:step]
    b, g, r = [im[ys, xs, i].astype(float) for i in range(3)]
    sel = (g > r + 25) & (g > b + 25) & (g > 90)        # grass-coloured
    if sel.sum() == 0: return xs[:0], ys[:0], g[:0]
    gthr = np.percentile(g[sel], 35) * .9               # keep the brighter (lit top-face) grass
    sel &= g > gthr
    return xs[sel].astype(float), ys[sel].astype(float), g[sel]

def fit_frame(im, prev, wide=False, hs=None):
    x, y, obs = roi(im)
    if len(x) < 300: return None
    yaw0, pitch0, h0 = prev[:3]
    if wide:
        yaws = yaw0 + np.radians(np.arange(-20, 20.1, 2)); pitches = pitch0 + np.radians(np.arange(-16, 16.1, 2))
    else:
        yaws = yaw0 + np.radians(np.arange(-6, 6.1, 1.5)); pitches = pitch0 + np.radians(np.arange(-6, 6.1, 1.5))
    hs = hs if hs is not None else [h0]
    cands = []
    sub = np.arange(len(x)) % 2 == 0
    for h in hs:
        for yw in yaws:
            for pt in pitches:
                s, a, b = phase_search(yw, pt, h, x[sub], y[sub], obs[sub])
                cands.append((s, yw, pt, h, a, b))
    cands.sort(key=lambda c: -c[0])
    best = None
    for s, yw, pt, h, a, b in cands[:4]:
        # unwrap phase near prediction
        cx = prev[3] + ((a - prev[3]) + .5) % 1 - .5; cz = prev[4] + ((b - prev[4]) + .5) % 1 - .5
        p0 = [yw, pt, h, cx, cz]
        p0 = np.array(p0); simp = np.vstack([p0, p0 + np.diag([.006, .006, .02, .008, .008])])
        r = minimize(lambda p: -corr_model(p, x, y, obs), p0, method='Nelder-Mead',
                     options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-6, 'initial_simplex': simp})
        if best is None or r.fun < best.fun: best = r
    return best, len(x)

def track(n0, n1, p_init, out, wide_frames=()):
    res = []; prev = list(p_init); prev2 = None
    for n in range(n0, n1):
        from popin_obs import frame
        im = frame(n)
        pred = list(prev)
        if prev2 is not None:                      # constant-velocity prediction for position only
            pred[3] = prev[3] + .7 * (prev[3] - prev2[3]); pred[4] = prev[4] + .7 * (prev[4] - prev2[4])
        hs = [prev[2]]
        f = fit_frame(im, pred, wide=(n in wide_frames), hs=hs)
        if f is None:
            res.append({'n': n, 'ok': False}); print(n, 'no roi', flush=True); continue
        r, npx = f
        sc = -r.fun
        if sc < .3:                               # try other planes / wider rotation
            f2 = fit_frame(im, pred, wide=True, hs=[max(.3, prev[2] - 1), prev[2], prev[2] + 1])
            if f2 is not None and -f2[0].fun > sc: r, npx = f2; sc = -r.fun
        p = r.x.tolist()
        s, c, sp, cp = math.sin(p[0]), math.cos(p[0]), math.sin(p[1]), math.cos(p[1])
        player = [p[3] + .3 * s * cp, p[4] - .3 * c * cp]
        rec = {'n': n, 'ok': True, 'score': sc, 'npx': int(npx), 'yaw_deg': math.degrees(p[0]) % 360, 'pitch_deg': math.degrees(p[1]),
               'h': p[2], 'cam': [p[3], p[4]], 'player': player}
        res.append(rec)
        print(n, round(sc, 3), round(rec['yaw_deg'], 2), round(rec['pitch_deg'], 2), round(p[2], 3), [round(v, 3) for v in player], flush=True)
        if sc > .25:
            prev2 = prev; prev = p
        out.write_text(json.dumps(res, indent=0))
    return res

if __name__ == '__main__':
    n0, n1 = int(sys.argv[1]), int(sys.argv[2])
    p = [float(v) for v in sys.argv[3].split(',')]
    track(n0, n1, p, P / sys.argv[4])
