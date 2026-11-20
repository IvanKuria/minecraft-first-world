"""Comparison images: footage frame | synthetic render from the same pose at a chosen spawn, for several worlds.
Spawn choice per world (frozen rule): the spawn whose r1+r2 near-field shallow-depression label fraction is closest to
the footage mean (0.29); this is the MOST footage-like view each world offers, so GEN worlds are shown at their best."""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import sweep as S, synth as SY, compare as CP, surf
HERE = S.HERE

def pick_spawn(js):
    d, agg, tru, per = CP.synth(js, '0.35')
    best = None
    for i, sp in enumerate(d['spawns']):
        v = []
        for w in ('r1', 'r2'):
            b = CP.bins(sp[f'{w}_0.35']['hist'])
            v.append((b['-1'] + b['-2'] + b['-3']) if b else 0)
        s = abs(np.mean(v) - .29)
        if best is None or s < best[0]: best = (s, i, sp['cx'], sp['cz'], v)
    return d, best

def main(worlds, out):
    rows = []
    sets = [('r1', 3), ('r2', 3)]
    for which, fi in sets:
        phase, frames = S.r1_poses() if which == 'r1' else S.r2_poses()
        row = [Image.open(frames[fi][1]).convert('RGB')]
        labels = ['FOOTAGE ' + frames[fi][0]]
        for wtag in worlds:
            js = HERE / f'synth/{wtag}.json'
            d, best = pick_spawn(js)
            _, i, cx, cz, v = best
            mode = d['mode']
            name, path, yaw, pitch, h = frames[fi]
            tmp = HERE / 'tmp_sbs.ppm'
            SY.render(d['world'], cx + phase[0], mode + 1 + h - .3 * math.sin(pitch), cz + phase[1], yaw, pitch, 'top', mode, tmp)
            row.append(Image.open(tmp).convert('RGB')); tmp.unlink(missing_ok=True)
            labels.append(f'{wtag} spawn {i} ({cx},{cz}) shallow {v[0]:.2f}/{v[1]:.2f}')
        rows.append((row, labels))
    W, H = 640, 360
    ncol = len(worlds) + 1
    sheet = Image.new('RGB', (W * ncol, (H + 18) * len(rows)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for r, (row, labels) in enumerate(rows):
        for c, (im, lab) in enumerate(zip(row, labels)):
            sheet.paste(im, (c * W, r * (H + 18) + 18))
            dr.text((c * W + 4, r * (H + 18) + 3), lab, fill=(0, 0, 0))
    sheet.save(HERE / f'img/{out}.png')

if __name__ == '__main__':
    main(sys.argv[2:], sys.argv[1])
