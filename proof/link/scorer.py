"""LINK frozen scorer: compare a candidate 256x256x64 world + candidate spawn positions with calibrated film frames.

Evidence used (all measured before any candidate world existed):
  * 7 frames of the 2011 original-channel 640x360 copy (sha256 0fcefdaf...), decoded indices 39/41/44 (opening,
    2.017-2.276 s) and 2959/2961/2964/2968 (second reset, 153.054-153.520 s).
  * camera yaw/pitch/height-above-nearby-grass from authentic grass-texture fits:
      opening   ../../geometry/opening-edge/pose-snapshot.json (best fixed-70 variant, the one fit_opening.py uses)
      reset 2   ../../geometry/edge-registration/four-texture-poses.json
  * camera model of generator/opening-texture/model.py (pixel centres, f=180/tan(35 deg), 0.3-block setback).
  * Notch's lighting rule (IRC 20:01 / rd-132211 Level+Tile): per-column depth map = highest solid y; a face is LIT iff
    the cell it faces is at or above that depth (top face: own cell). Lit faces drawn without fog; unlit faces drawn
    with brightness 0.8 and GL_EXP fog (density 0.2, colour 0x0E0B0A). Face shades 1.0 (y), 0.8 (z), 0.6 (x).

Candidate inputs: blocks (uint8, layout (y*256+z)*256+x, nonzero=solid) and the candidate's opening and second-reset
player x,z. The player is assumed horizontally stationary since its reset (same assumption that produced the RNG
candidates). Its nearby grass plane is the landing surface under its 0.6x0.6 bounding box (max top over the
overlapped columns), and camera y = plane + measured height.

Classes per 4x4-pooled pixel (160x90): SKY / DARK / LIT. Observed: SKY if B>200 and B-R>60 (pooled majority),
DARK if pooled luminance < 45, else LIT. Predicted: same luminance threshold on the rendered luminance
(albedo 0.49 = lit grass top luminance 125/255).
Statistics: per-frame IoU of each class; T_dark = pooled DARK IoU over all 7 frames (primary: holes and unlit faces),
mIoU = mean of pooled class IoUs (secondary). Thresholds live in FROZEN.json (set from null calibration only).
"""
import ctypes, json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

P = Path(__file__).parent
ROOT = P.parents[1]
W = H = 256
D = 64
OUT_W, OUT_H, STRIDE = 160, 90, 4
FOCAL = 180 / math.tan(math.radians(35))
SKY_Y = 0.299 * 117 + 0.587 * 200 + 0.114 * 245          # measured sky colour
DARK_T = 45.0
ALBEDO = 0.49
FOG_Y = 0.299 * 14 + 0.587 * 11 + 0.114 * 10

_lib = None


def lib():
    global _lib
    if _lib is None:
        so = P / 'libraycast.dylib'
        if not so.exists() or so.stat().st_mtime < (P / 'raycast.c').stat().st_mtime:
            subprocess.run(['cc', '-O2', '-shared', '-fPIC', '-o', str(so), str(P / 'raycast.c')], check=True)
        _lib = ctypes.CDLL(str(so))
        d = ctypes.POINTER(ctypes.c_double)
        u8 = ctypes.POINTER(ctypes.c_uint8)
        _lib.cast_rays.argtypes = [u8, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, d, d, d, ctypes.c_int, ctypes.c_double, u8, d, u8,
                                   ctypes.POINTER(ctypes.c_int32)]
    return _lib


def load_frames():
    opening = json.loads((ROOT / 'geometry/opening-edge/pose-snapshot.json').read_text())
    op = max((r for r in opening if r['mode'] == 'fixed70'), key=lambda r: r['mean_train'])
    r2 = json.loads((ROOT / 'geometry/edge-registration/four-texture-poses.json').read_text())
    frames = []
    for name, fr in zip(['o-2.017', 'o-2.121', 'o-2.276'], op['frames']):
        ya, pi, h = fr['camera_parameters'][:3]
        frames.append({'name': name, 'which': 0, 'yaw': ya, 'pitch': pi, 'h': h})
    for name, fr in zip(['r2-153.054', 'r2-153.158', 'r2-153.313', 'r2-153.520'], r2['frames']):
        ya, pi, h = fr['yaw_pitch_height']
        frames.append({'name': name, 'which': 1, 'yaw': ya, 'pitch': pi, 'h': h})
    for f in frames:
        f['obs'] = observed_classes(P / 'frames' / f'{f["name"]}.png')
    return frames


def observed_classes(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    Y = .299 * R + .587 * G + .114 * B
    sky = ((B > 200) & (B - R > 60)).astype(float)
    pool = lambda m: m.reshape(OUT_H, STRIDE, OUT_W, STRIDE).mean(axis=(1, 3))
    skyp, Yp = pool(sky), pool(Y)
    cls = np.where(skyp > .5, 0, np.where(Yp < DARK_T, 1, 2)).astype(np.uint8)
    return cls


def ray_dirs(yaw, pitch):
    xs = np.arange(OUT_W) * STRIDE + 1.5
    ys = np.arange(OUT_H) * STRIDE + 1.5
    X, Y = np.meshgrid(xs, ys)
    rx = (X + .5 - 320) / FOCAL
    ry = (Y + .5 - 180) / FOCAL
    s, c, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    dx = c * rx - s * sp * ry + s * cp
    dy = -cp * ry - sp
    dz = s * rx + c * sp * ry - c * cp
    return [np.ascontiguousarray(v.ravel(), dtype=np.float64) for v in (dx, dy, dz)]


class World:
    def __init__(self, blocks):
        b = np.ascontiguousarray(np.asarray(blocks, dtype=np.uint8).reshape(D, H, W))
        self.b = (b != 0).astype(np.uint8)
        solid = self.b.astype(bool)                        # [y,z,x]
        ys = np.arange(D)[:, None, None]
        top = np.where(solid, ys, -1).max(axis=0)          # [z,x]
        self.top = top                                     # highest solid y, -1 if none
        self.light = np.where(top > 0, top, 0)             # calcLightDepths: stops at y>0 -> 0 if none

    def bright(self, x, y, z):
        """getBrightness: 1 lit, 0 dark (x,y,z int arrays)."""
        out = np.ones(x.shape, dtype=bool)
        inside = (x >= 0) & (y >= 0) & (z >= 0) & (x < W) & (y < D) & (z < H)
        xi, zi = np.clip(x, 0, W - 1), np.clip(z, 0, H - 1)
        dark = inside & (y < self.light[zi, xi])
        out[dark] = False
        return out

    def landing_plane(self, px, pz):
        xs = range(int(math.floor(px - .3)), int(math.floor(px + .3)) + 1)
        zs = range(int(math.floor(pz - .3)), int(math.floor(pz + .3)) + 1)
        tops = [self.top[z, x] for x in xs for z in zs if 0 <= x < W and 0 <= z < H]
        t = max(tops) if tops else -1
        return int(t) + 1 if t >= 0 else None

    def render(self, px, pz, f, plane=None):
        if plane is None:
            plane = self.landing_plane(px, pz)
        if plane is None:
            return None
        s, c, cp = math.sin(f['yaw']), math.cos(f['yaw']), math.cos(f['pitch'])
        cx, cz, cy = px - .3 * s * cp, pz + .3 * c * cp, plane + f['h']
        dx, dy, dz = ray_dirs(f['yaw'], f['pitch'])
        n = dx.size
        hit = np.zeros(n, np.uint8); t = np.zeros(n); face = np.zeros(n, np.uint8); cell = np.zeros(3 * n, np.int32)
        L = lib()
        cd = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        L.cast_rays(self.b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), W, H, D, cx, cy, cz, cd(dx), cd(dy), cd(dz),
                    n, 2000.0, hit.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), cd(t),
                    face.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), cell.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        cell = cell.reshape(n, 3)
        lum = np.full(n, SKY_Y)
        h = hit.astype(bool)
        x, y, z, fc = cell[h, 0], cell[h, 1], cell[h, 2], face[h]
        off = np.array([[0, -1, 0], [0, 0, 0], [0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0]])
        shade = np.array([1.0, 1.0, .8, .8, .6, .6])[fc]
        nb = off[fc]
        lit = self.bright(x + nb[:, 0], y + nb[:, 1], z + nb[:, 2])
        fog = np.exp(-.2 * t[h])
        val = np.where(lit, ALBEDO * shade * 255, fog * ALBEDO * .8 * shade * 255 + (1 - fog) * FOG_Y)
        lum[h] = val
        cls = np.where(~h, 0, np.where(lum < DARK_T, 1, 2)).astype(np.uint8)
        return {'cls': cls.reshape(OUT_H, OUT_W), 'lum': lum.reshape(OUT_H, OUT_W), 'plane': plane}


def iou(pred, obs, k):
    a, b = pred == k, obs == k
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0, int((a & b).sum()), int(u)


def score(world, positions, frames, plane_override=None):
    """positions: [[x0,z0],[x2,z2]] for opening and second reset. Returns dict of statistics."""
    inter = np.zeros(3); union = np.zeros(3); per = []
    sub = {0: [np.zeros(3), np.zeros(3)], 1: [np.zeros(3), np.zeros(3)]}
    for f in frames:
        px, pz = positions[f['which']]
        r = world.render(px, pz, f, None if plane_override is None else plane_override[f['which']])
        if r is None:
            pred = np.zeros_like(f['obs'])            # fell out of the world: all sky
        else:
            pred = r['cls']
        row = {'frame': f['name'], 'plane': None if r is None else r['plane']}
        for k, nm in enumerate(['sky', 'dark', 'lit']):
            v, i, u = iou(pred, f['obs'], k)
            inter[k] += i; union[k] += u; row[nm] = v
            sub[f['which']][0][k] += i; sub[f['which']][1][k] += u
        per.append(row)
    pooled = inter / np.maximum(union, 1)
    so = sub[0][0] / np.maximum(sub[0][1], 1); sr = sub[1][0] / np.maximum(sub[1][1], 1)
    return {'T_dark': float(pooled[1]), 'mIoU': float(pooled.mean()), 'iou_sky': float(pooled[0]),
            'iou_lit': float(pooled[2]), 'T_dark_open': float(so[1]), 'sky_open': float(so[0]),
            'T_dark_r2': float(sr[1]), 'sky_r2': float(sr[0]), 'per_frame': per}


def load_blocks(path):
    b = np.fromfile(path, dtype=np.uint8)
    assert b.size == W * H * D, f'{path}: expected {W*H*D} bytes, got {b.size}'
    return b
