"""Exploratory (NOT frozen): grass-pixel fraction of non-sky pixels in steep top-down views (footage 86.5 s analogue).
usage: greenfrac.py world.blocks -> mean over 4 cameras"""
import sys, subprocess, numpy as np
def frac(p):
    a = np.asarray(open_ppm(p)).astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    sky = (b > 200) & (r < 150); green = (g > r + 25) & (g > b + 25) & ~sky
    return green.sum() / max((~sky).sum(), 1)
def open_ppm(p):
    from PIL import Image; return Image.open(p).convert("RGB")
vals = []
for i, (x, z) in enumerate([(64, 64), (64, 192), (192, 64), (192, 192)]):
    o = f"tmp/gf{i}.ppm"
    subprocess.check_call(["./render", sys.argv[1], "256", "256", "64", str(x + .5), "74", str(z + .5), str(20 + 90 * i), "58", "426", "240", "y42", o])
    vals.append(frac(o))
print("%.3f" % np.mean(vals), " ".join("%.2f" % v for v in vals))
