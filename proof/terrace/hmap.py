"""Top-down picture of the sweep labels (k per column) next to the C5/R1MAP open/closed maps."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
COL = {-3: (120, 0, 160), -2: (0, 0, 255), -1: (0, 200, 255), 0: (240, 240, 240), 1: (255, 200, 0), 2: (255, 80, 0), 3: (200, 0, 0)}

def load_obs(which):
    p = PROOF / ('r1map/footage-obs-r1.json' if which == 'r1' else 'c5/footage-obs.json')
    d = json.loads(p.read_text())
    return {(x, z): v for x, z, v in d['labels']}

def main(which, rmin, margin):
    d = json.loads((HERE / f'sweep-{which}.json').read_text())
    sc = {(x, z): {int(k): v for k, v in s.items()} for x, z, s in d['scores']}
    lab = {}
    for col, s in sc.items():
        it = sorted(s.items(), key=lambda kv: -kv[1][0])
        if it and it[0][1][0] >= rmin and (len(it) < 2 or it[0][1][0] - it[1][1][0] >= margin): lab[col] = it[0][0]
    obs = load_obs(which)
    xs = [k[0] for k in list(lab) + list(obs)]; zs = [k[1] for k in list(lab) + list(obs)]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    S = 10
    im = Image.new('RGB', ((x1 - x0 + 1) * S * 2 + S, (z1 - z0 + 1) * S), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    for (x, z), v in obs.items():
        c = (60, 60, 60) if v else (110, 190, 110)
        dr.rectangle([(x - x0) * S, (z - z0) * S, (x - x0) * S + S - 1, (z - z0) * S + S - 1], fill=c)
    off = (x1 - x0 + 1) * S + S
    for (x, z), v in obs.items():
        dr.rectangle([off + (x - x0) * S, (z - z0) * S, off + (x - x0) * S + S - 1, (z - z0) * S + S - 1], fill=(60, 60, 60) if v else (200, 200, 200))
    for (x, z), k in lab.items():
        dr.rectangle([off + (x - x0) * S, (z - z0) * S, off + (x - x0) * S + S - 1, (z - z0) * S + S - 1], fill=COL[k])
    for o in (0, off):
        dr.rectangle([o + (0 - x0) * S + 3, (0 - z0) * S + 3, o + (0 - x0) * S + S - 4, (0 - z0) * S + S - 4], fill=(255, 0, 0))
    im.save(HERE / f'img/hmap-{which}.png')
    hist = {k: sum(1 for v in lab.values() if v == k) for k in range(-3, 4)}
    print(which, 'rmin', rmin, 'margin', margin, 'labelled', len(lab), hist)
    return lab

if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
