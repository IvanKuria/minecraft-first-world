"""Plane sweep with the ROCK texture (terrain.png tile 1) over non-green, non-sky pixels: are there lit stone TOP faces?
Under H-grass (column-top solid is grass) every lit top face is grass; a lit stone top at some level refutes it."""
import json, math, sys
import numpy as np
from PIL import Image
import sweep as S

_rt = None
def rtile():
    global _rt
    if _rt is None:
        a = np.asarray(Image.open(S.HERE / 'tex/terrain.png').convert('RGB')).astype(float)
        _rt = a[:16, 16:32].mean(2)
    return _rt

def rlookup(wx, wz):
    t = rtile()
    u = ((wx % 1) * 15.984 - .5) % 16; v = ((wz % 1) * 15.984 - .5) % 16
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int); a = u - u0; b = v - v0
    return (t[v0, u0] * (1 - a) * (1 - b) + t[v0, (u0 + 1) % 16] * a * (1 - b) + t[(v0 + 1) % 16, u0] * (1 - a) * b + t[(v0 + 1) % 16, (u0 + 1) % 16] * a * b)

def frame_scores(img, yaw, pitch, h, phase, ks=range(-6, 3)):
    a = np.asarray(img).astype(float)
    Hh, Ww = a.shape[:2]
    sky, green = S.classes(a)
    obs = a.mean(2)
    bright = obs > 60  # lit stone is bright grey; unlit/fogged stone is dark
    dx, dy, dz = S.rays(yaw, pitch, Ww, Hh)
    s, c, cp = math.sin(yaw), math.cos(yaw), math.cos(pitch)
    camx = phase[0] - .3 * s * cp; camz = phase[1] + .3 * c * cp
    sel0 = (~sky) & (~green) & bright & (dy < -1e-6)
    out = {}
    for k in ks:
        hk = h - k
        if hk <= .05: continue
        t = -hk / np.where(sel0, dy, -1)
        Xp = camx + t * dx; Zp = camz + t * dz
        ok = sel0 & (np.hypot(Xp - camx, Zp - camz) <= S.DMAX)
        xs, zs, o = Xp[ok], Zp[ok], obs[ok]
        p = rlookup(xs, zs)
        key = (np.floor(xs).astype(np.int64) + 4096) * 8192 + (np.floor(zs).astype(np.int64) + 4096)
        uk, inv = np.unique(key, return_inverse=True)
        n = np.bincount(inv).astype(float)
        so, sp_ = np.bincount(inv, o), np.bincount(inv, p)
        cov = np.bincount(inv, o * p) - so * sp_ / n
        vo = np.bincount(inv, o * o) - so * so / n; vp = np.bincount(inv, p * p) - sp_ * sp_ / n
        r = cov / np.sqrt(np.maximum(vo * vp, 1e-9))
        out[k] = (np.stack([uk // 8192 - 4096, uk % 8192 - 4096], 1), n, r)
    return out

if __name__ == '__main__':
    for which in ('r1', 'r2'):
        phase, frames = S.r1_poses() if which == 'r1' else S.r2_poses()
        per = [frame_scores(Image.open(p).convert('RGB'), y, pi, h, phase) for _, p, y, pi, h in frames]
        res = S.combine(per, nmin=20)
        for rmin, mg in ((.45, .15), (.35, .12)):
            lab = S.assign(res, rmin, mg)
            hist = {k: sum(1 for v in lab.values() if v[0] == k) for k in range(-6, 3)}
            print(which, 'stone-top sweep rmin', rmin, 'labelled', len(lab), hist)
        json.dump({'labels': [[x, z, v[0], round(v[1], 3), int(v[3])] for (x, z), v in sorted(S.assign(res, .35, .12).items())]},
                  open(S.HERE / f'stonesweep-{which}.json', 'w'))
