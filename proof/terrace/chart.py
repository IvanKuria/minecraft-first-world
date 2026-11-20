"""Dot-strip chart: per synthetic view, near-field shallow-depression label fraction (-1..-3), per world, vs footage."""
import glob, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import compare as CP
F = CP.footage('0.35')
fv = {w: F[w]['bins']['-1'] + F[w]['bins']['-2'] + F[w]['bins']['-3'] for w in ('r1', 'r2')}
worlds = sys.argv[2:]
rows = []
for t in worlds:
    d, agg, tru, per = CP.synth(str(CP.HERE / f'synth/{t}.json'), '0.35')
    v = [((b['-1'] + b['-2'] + b['-3']) if b else 0.0) for w in ('r1', 'r2') for b in per[w]]
    rows.append((t, v))
W, rh, L, R, T = 1100, 34, 300, 40, 50
H = T + rh * len(rows) + 60
im = Image.new('RGB', (W, H), (255, 255, 255)); dr = ImageDraw.Draw(im)
try: font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 15)
except Exception: font = None
X = lambda f: L + f * (W - L - R)
for g in (0, .2, .4, .6, .8, 1.0):
    dr.line([X(g), T - 10, X(g), H - 50], fill=(230, 230, 230)); dr.text((X(g) - 10, H - 44), f'{g:.1f}', fill=(90, 90, 90), font=font)
dr.text((L, H - 24), 'near-field shallow-depression fraction (labels at -1..-3), one dot per synthetic view (24 per world)', fill=(60, 60, 60), font=font)
for w, c in (('r1', (20, 20, 20)), ('r2', (20, 20, 20))):
    dr.line([X(fv[w]), T - 30, X(fv[w]), H - 50], fill=c, width=2)
    dr.text((X(fv[w]) + 4, T - 34), f'footage {w} {fv[w]:.2f}', fill=c, font=font)
for i, (t, v) in enumerate(rows):
    y = T + rh * i + rh // 2
    dr.text((10, y - 8), t, fill=(40, 40, 40), font=font)
    for f in v:
        dr.ellipse([X(f) - 4, y - 4, X(f) + 4, y + 4], fill=(70, 110, 180), outline=(255, 255, 255))
im.save(CP.HERE / f'img/{sys.argv[1]}.png')
