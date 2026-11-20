"""Grass-rule control: local world from the r1/r2 sweep labels (+obs-map holes) rendered with three grass rules,
cropped to the near-field depression region, next to the footage crop."""
import sys, math
import numpy as np
from PIL import Image
import sweep as S, checkrender as C
HERE = S.HERE
def run(which, fi, box, tag):
    phase, frames = S.r1_poses() if which == 'r1' else S.r2_poses()
    blocks, lab = C.build(which)
    crops = [Image.open(frames[fi][1]).convert('RGB').crop(box)]
    for rule, gy in (('top', 42), ('eq', 42), ('ge', 34)):
        out = HERE / 'tmp_gr.ppm'
        C.render(blocks, frames[fi], phase, out, rule=rule, gy=gy)
        crops.append(Image.open(out).convert('RGB').crop(box)); out.unlink()
    w, h = crops[0].size
    z = 2
    sheet = Image.new('RGB', (w * z * 2 + 8, h * z * 2 + 8), (255, 255, 255))
    for i, c in enumerate(crops):
        sheet.paste(c.resize((w * z, h * z), Image.LANCZOS), ((i % 2) * (w * z + 8), (i // 2) * (h * z + 8)))
    sheet.save(HERE / f'img/grassrule-{tag}.png')
run('r1', 3, (0, 150, 420, 360), 'r1-n1668')
run('r2', 3, (240, 150, 640, 360), 'r2-153.520')
