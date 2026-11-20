"""TERRACE: texture plane-sweep height map of the visible grass surface (footage or synthetic renders).

For a frame with a known pose (yaw, pitch, camera height h above the reference grass plane, player phase), every pixel
ray is intersected with the horizontal planes k = -3..+3 blocks relative to the reference plane.  At each crossing the
authentic grass tile (rd-132211 terrain.png tile 0, green channel, span 15.984 texels, bilinear as in the
first-reset-texture fits) predicts the pixel value.  For each column (cx,cz) (relative to floor(player)) and each k the
Pearson correlation between observed and predicted values over the pixels whose plane-k crossing falls in that column
measures 'the top face of this column is at height k and visible'.  Only top faces carry the right phase; side faces,
stone and sky do not correlate.  Pose conventions are exactly obs_footage.project (FOV 70 => f=180/tan35, pixel
centres, 0.3 setback, zero roll).
"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
ROOT = PROOF.parent
F = 180 / math.tan(math.radians(35))
KS = list(range(-8, 4))
DMAX = 32.0

_tile = None
def tile():
    global _tile
    if _tile is None:
        im = Image.open(HERE / 'tex/terrain.png').convert('RGB')
        _tile = np.asarray(im).astype(float)[:16, :16, 1]
    return _tile


def lookup(wx, wz):
    t = tile()
    u = ((wx % 1) * 15.984 - .5) % 16; v = ((wz % 1) * 15.984 - .5) % 16
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int); a = u - u0; b = v - v0
    return (t[v0, u0] * (1 - a) * (1 - b) + t[v0, (u0 + 1) % 16] * a * (1 - b) + t[(v0 + 1) % 16, u0] * (1 - a) * b
            + t[(v0 + 1) % 16, (u0 + 1) % 16] * a * b)


def rays(yaw, pitch, W=640, H=360):
    X, Y = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
    f = F * H / 360.0
    rx = (X - W / 2) / f; ry = (Y - H / 2) / f
    s, c, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    dx = c * rx - s * sp * ry + s * cp
    dy = -cp * ry - sp
    dz = s * rx + c * sp * ry - c * cp
    return dx, dy, dz


def classes(a):
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    sky = (B > 200) & (B - R > 60)
    green = (~sky) & (G > R + 15) & (G > B + 15)
    return sky, green


def frame_scores(img, yaw, pitch, h, phase, ks=KS, mask=None):
    """returns dict k -> (keys[n,2], n, r) per column, using green pixels only."""
    a = np.asarray(img).astype(float)
    Hh, Ww = a.shape[:2]
    sky, green = classes(a)
    obs = a[..., 1]
    dx, dy, dz = rays(yaw, pitch, Ww, Hh)
    s, c, cp = math.sin(yaw), math.cos(yaw), math.cos(pitch)
    camx = phase[0] - .3 * s * cp; camz = phase[1] + .3 * c * cp
    sel0 = green & (dy < -1e-6)
    if mask is not None: sel0 &= mask
    out = {}
    for k in ks:
        hk = h - k
        if hk <= 0.05: continue
        t = -hk / np.where(sel0, dy, -1)
        Xp = camx + t * dx; Zp = camz + t * dz
        ok = sel0 & (np.hypot(Xp - camx, Zp - camz) <= DMAX)
        xs = Xp[ok]; zs = Zp[ok]; o = obs[ok]
        p = lookup(xs, zs)
        cx = np.floor(xs).astype(np.int64); cz = np.floor(zs).astype(np.int64)
        key = (cx + 4096) * 8192 + (cz + 4096)
        uk, inv = np.unique(key, return_inverse=True)
        n = np.bincount(inv).astype(float)
        so = np.bincount(inv, o); sp_ = np.bincount(inv, p)
        soo = np.bincount(inv, o * o); spp = np.bincount(inv, p * p); sop = np.bincount(inv, o * p)
        cov = sop - so * sp_ / n; vo = soo - so * so / n; vp = spp - sp_ * sp_ / n
        r = cov / np.sqrt(np.maximum(vo * vp, 1e-9))
        keys = np.stack([uk // 8192 - 4096, uk % 8192 - 4096], 1)
        out[k] = (keys, n, r)
    return out


def combine(per_frame, nmin=20):
    """per column: n-weighted mean correlation per k over frames (only frame entries with n>=nmin)."""
    acc = {}
    for fs in per_frame:
        for k, (keys, n, r) in fs.items():
            for (x, z), nn, rr in zip(keys.tolist(), n.tolist(), r.tolist()):
                if nn < nmin: continue
                d = acc.setdefault((x, z), {})
                e = d.setdefault(k, [0.0, 0.0])
                e[0] += nn * rr; e[1] += nn
    res = {}
    for col, d in acc.items():
        res[col] = {k: (e[0] / e[1], e[1]) for k, e in d.items()}
    return res


def assign(res, rmin=0.45, margin=0.15):
    """best k per column if its correlation >= rmin and beats the runner-up by margin."""
    lab = {}
    for col, d in res.items():
        items = sorted(d.items(), key=lambda kv: -kv[1][0])
        if not items: continue
        kb, (rb, nb) = items[0]
        r2 = items[1][1][0] if len(items) > 1 else -1
        if rb >= rmin and rb - r2 >= margin:
            lab[col] = (kb, rb, rb - r2, nb)
    return lab


def r1_poses(mode=0):
    TEX = ROOT / 'generator/first-reset-texture'
    sh = json.loads((TEX / 'shared-fixed70.json').read_text())[mode]
    bk = json.loads((TEX / 'backward-holdout.json').read_text())[mode]
    phase = tuple(sh['shared_player_phase'])
    out = [('n1662', *bk['pose'][:3])]
    for name, key in (('n1663', '86.00'), ('n1666', '86.15'), ('n1668', '86.30')):
        p = next(fr['camera_parameters'] for fr in sh['frames'] if fr['frame'] == key)
        out.append((name, p[0], p[1], p[2]))
    return phase, [(n, PROOF / 'r1map/frames' / f'{n}.png', y, p, h) for n, y, p, h in out]


def r2_poses():
    P = json.loads((ROOT / 'geometry/edge-registration/four-texture-poses.json').read_text())
    names = ['r2-153.054', 'r2-153.158', 'r2-153.313', 'r2-153.520']
    phase = (.8118, .8185)
    return phase, [(n, PROOF / 'link/frames' / f'{n}.png', *fr['yaw_pitch_height']) for n, fr in zip(names, P['frames'])]


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'r1'
    phase, frames = r1_poses() if which == 'r1' else r2_poses()
    per = []
    for name, path, yaw, pitch, h in frames:
        fs = frame_scores(Image.open(path).convert('RGB'), yaw, pitch, h, phase)
        per.append(fs)
        # per-frame histogram of best k
        rr = combine([fs]); lab = assign(rr)
        hist = {k: sum(1 for v in lab.values() if v[0] == k) for k in KS}
        print(name, 'h=%.2f' % h, 'labelled', len(lab), hist, flush=True)
    res = combine(per)
    lab = assign(res)
    hist = {k: sum(1 for v in lab.values() if v[0] == k) for k in KS}
    print('pooled labelled', len(lab), hist)
    out = {'which': which, 'phase': phase, 'frames': [[n, str(p), y, pi, h] for n, p, y, pi, h in frames],
           'labels': [[x, z, v[0], round(v[1], 3), round(v[2], 3), int(v[3])] for (x, z), v in sorted(lab.items())],
           'scores': [[x, z, {str(k): [round(r, 3), int(n)] for k, (r, n) in d.items()}] for (x, z), d in sorted(res.items())]}
    (HERE / f'sweep-{which}.json').write_text(json.dumps(out) + '\n')
