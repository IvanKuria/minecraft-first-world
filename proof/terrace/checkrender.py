"""Control: build a local world from the sweep labels (+ C5/R1MAP open columns as holes) and render it with the
authentic textures from the footage pose, side by side with the footage frame."""
import json, math, sys, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import sweep as S
HERE = S.HERE
W, H, D = 256, 256, 64
YREF = 42

def build(which, rmin=.35, margin=.12, flat=False):
    d = json.loads((HERE / f'sweep-{which}.json').read_text())
    sc = {(x, z): {int(k): v for k, v in s.items()} for x, z, s in d['scores']}
    lab = {}
    for col, s in sc.items():
        it = sorted(s.items(), key=lambda kv: -kv[1][0])
        if it and it[0][1][0] >= rmin and (len(it) < 2 or it[0][1][0] - it[1][1][0] >= margin): lab[col] = it[0][0]
    obs = json.loads((S.PROOF / ('r1map/footage-obs-r1.json' if which == 'r1' else 'c5/footage-obs.json')).read_text())
    opn = {(x, z) for x, z, v in obs['labels'] if v == 1}
    top = np.full((H, W), YREF, int)
    ox, oz = 128, 128
    if not flat:
        for (x, z), k in lab.items(): top[oz + z, ox + x] = YREF + k
        for (x, z) in opn:
            if (x, z) not in lab: top[oz + z, ox + x] = YREF - 10
    blocks = np.zeros((D, H, W), np.uint8)
    yy = np.arange(D)[:, None, None]
    blocks[:] = (yy <= top[None]).astype(np.uint8)
    blocks[0] = 1
    return blocks, lab

def render(blocks, pose, phase, out, rule='top', gy=YREF):
    name, path, yaw, pitch, h = pose
    p = HERE / 'tmp_world.bin'; blocks.tofile(p)
    px = 128 + phase[0]; pz = 128 + phase[1]
    py = YREF + 1 + h - .3 * math.sin(pitch)
    subprocess.run([str(HERE / 'trender'), str(p), str(W), str(H), str(D), '%.6f' % px, '%.6f' % py, '%.6f' % pz,
                    '%.8f' % yaw, '%.8f' % pitch, '640', '360', rule, str(gy), str(HERE / 'tex/terrain.ppm'), str(out)], check=True)
    p.unlink()

if __name__ == '__main__':
    which = sys.argv[1]; fi = int(sys.argv[2]); flat = len(sys.argv) > 3 and sys.argv[3] == 'flat'
    phase, frames = S.r1_poses() if which == 'r1' else S.r2_poses()
    blocks, lab = build(which, flat=flat)
    out = HERE / 'tmp_render.ppm'
    render(blocks, frames[fi], phase, out)
    a = Image.open(frames[fi][1]).convert('RGB'); b = Image.open(out).convert('RGB')
    sheet = Image.new('RGB', (640, 720)); sheet.paste(a, (0, 0)); sheet.paste(b, (0, 360))
    tag = 'flat' if flat else 'labels'
    sheet.save(HERE / f'img/control-{which}-{frames[fi][0]}-{tag}.png'); out.unlink()
    # numeric: correlation footage vs render green channel over green pixels
    A = np.asarray(a).astype(float)[..., 1]; Bb = np.asarray(b).astype(float)[..., 1]
    print(frames[fi][0], tag, 'corr(G) all', round(float(np.corrcoef(A.ravel(), Bb.ravel())[0, 1]), 3))
