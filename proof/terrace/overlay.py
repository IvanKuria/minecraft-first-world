"""Overlay the plane-sweep labels on the frames: a pixel is coloured by k if its plane-k crossing column is labelled k."""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
import sweep as S

COL = {-3: (120, 0, 160), -2: (0, 0, 255), -1: (0, 200, 255), 0: (255, 255, 255), 1: (255, 200, 0), 2: (255, 80, 0), 3: (200, 0, 0)}

def overlay(img, yaw, pitch, h, phase, lab):
    a = np.asarray(img).astype(float)
    Hh, Ww = a.shape[:2]
    dx, dy, dz = S.rays(yaw, pitch, Ww, Hh)
    s, c, cp = math.sin(yaw), math.cos(yaw), math.cos(pitch)
    camx = phase[0] - .3 * s * cp; camz = phase[1] + .3 * c * cp
    out = (a * 0.35).copy()
    painted = np.zeros(a.shape[:2], bool)
    for k in sorted(S.KS, key=lambda k: -k):  # nearer (higher) planes first
        hk = h - k
        if hk <= .05: continue
        t = -hk / np.where(dy < -1e-6, dy, -1)
        Xp = camx + t * dx; Zp = camz + t * dz
        cx = np.floor(Xp).astype(int); cz = np.floor(Zp).astype(int)
        m = (dy < -1e-6) & ~painted
        idx = np.argwhere(m)
        for (j, i) in idx:
            v = lab.get((cx[j, i], cz[j, i]))
            if v is not None and v == k:
                out[j, i] = 0.5 * a[j, i] + 0.5 * np.array(COL[k]); painted[j, i] = True
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

if __name__ == '__main__':
    which = sys.argv[1]
    d = json.loads((S.HERE / f'sweep-{which}.json').read_text())
    lab = {(x, z): k for x, z, k, *_ in d['labels']}
    phase, frames = S.r1_poses() if which == 'r1' else S.r2_poses()
    tiles = []
    for name, path, yaw, pitch, h in frames:
        img = Image.open(path).convert('RGB')
        ov = overlay(img, yaw, pitch, h, phase, lab)
        both = Image.new('RGB', (640, 720)); both.paste(img, (0, 0)); both.paste(ov, (0, 360))
        tiles.append(both)
    sheet = Image.new('RGB', (640 * len(tiles), 720))
    for i, t in enumerate(tiles): sheet.paste(t, (640 * i, 0))
    sheet.save(S.HERE / f'img/sweep-overlay-{which}.png')
