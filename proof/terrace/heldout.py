"""Held-out control for the footage depression labels: labels from 3 frames of a reset, world = plain + labelled
levels (+ obs-map holes), rendered at the 4th frame's pose; compare the 4th frame's near field (lower half) with
(a) holes-only world, (b) holes + held-out labels.  Score = G-channel correlation over pixels whose plane-0 crossing
lies within 10 blocks of the player."""
import json, math, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import sweep as S, checkrender as C
W, H, D = 256, 256, 64
def world(lab, opn, YREF=42):
    top = np.full((H, W), YREF, int)
    for (x, z), k in lab.items(): top[128 + z, 128 + x] = YREF + k
    for (x, z) in opn:
        if (x, z) not in lab: top[128 + z, 128 + x] = YREF - 10
    b = (np.arange(D)[:, None, None] <= top[None]).astype(np.uint8); b[0] = 1
    return b
for which in ('r1', 'r2'):
    phase, frames = S.r1_poses() if which == 'r1' else S.r2_poses()
    obs = json.loads((S.PROOF / ('r1map/footage-obs-r1.json' if which == 'r1' else 'c5/footage-obs.json')).read_text())
    opn = {(x, z) for x, z, v in obs['labels'] if v == 1}
    per = [S.frame_scores(Image.open(p).convert('RGB'), y, pi, h, phase) for _, p, y, pi, h in frames]
    for i in range(4):
        lab = {c: v[0] for c, v in S.assign(S.combine([per[j] for j in range(4) if j != i]), .35, .12).items()
               if math.hypot(c[0] + .5 - phase[0], c[1] + .5 - phase[1]) < 10}
        name, path, yaw, pitch, h = frames[i]
        foot = np.asarray(Image.open(path).convert('RGB')).astype(float)[..., 1]
        dx, dy, dz = S.rays(yaw, pitch)
        s, c, cp = math.sin(yaw), math.cos(yaw), math.cos(pitch)
        camx = phase[0] - .3 * s * cp; camz = phase[1] + .3 * c * cp
        t = -h / np.where(dy < -1e-6, dy, -1)
        near = (dy < -1e-6) & (np.hypot(t * dx + camx - phase[0], t * dz + camz - phase[1]) < 10)
        res = {}
        for tag, lb in (('holes-only', {}), ('holes+labels', lab), ('holes+labels-shuffled', {(x, z): 0 for (x, z) in lab})):
            out = S.HERE / 'tmp_h.ppm'
            C.render(world(lb, opn), frames[i], phase, out)
            r = gaussian_filter(np.asarray(Image.open(out).convert('RGB')).astype(float)[..., 1], .9); out.unlink()
            res[tag] = float(np.corrcoef(foot[near], r[near])[0, 1])
        nneg = sum(1 for k in lab.values() if k < 0)
        print(which, name, 'held-out labels %d (%d below plane)' % (len(lab), nneg), {k: round(v, 3) for k, v in res.items()})
