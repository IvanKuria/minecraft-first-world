"""Render footage-analogous views of a world and tile them next to the footage frames.
usage: views.py world.blocks label out.png [grassrule]
Views (approximate analogues, NOT registered poses):
  A  high after respawn, steep look down      ~ footage 86.5 s
  B  standing on the plain, looking along it  ~ footage 9 s / 162 s
  C  inside a large cave                      ~ footage 92 s / 20-80 s
  D  outside the world, below the rim         ~ footage 167-172 s
"""
import sys, subprocess, os
import numpy as np
from scipy import ndimage
W, H, D = 256, 256, 64
f, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
gr = sys.argv[4] if len(sys.argv) > 4 else "y42"
s = np.fromfile(f, np.uint8).reshape(D, H, W) != 0
has = s.any(0)
top = np.where(has, D - 1 - np.argmax(s[::-1], 0), -1)
vals, cnt = np.unique(top, return_counts=True)
mode = int(vals[np.argmax(cnt)])
# B: a plain column near the centre whose 9x9 neighbourhood is mostly at mode level
best = None
for zc in range(100, 160, 4):
    for xc in range(100, 160, 4):
        sc = (top[zc - 4:zc + 5, xc - 4:xc + 5] == mode).mean()
        if best is None or sc > best[0]: best = (sc, xc, zc)
_, bx, bz = best
# C: cave camera = interior air cell with the largest distance to rock, below the surface
air = ~s
dist = ndimage.distance_transform_edt(air[2:mode - 3, 8:-8, 8:-8]) if mode > 8 else None
if dist is not None and dist.max() > 1.5:
    y, z, x = np.unravel_index(np.argmax(dist), dist.shape); cy, cz, cx = y + 2, z + 8, x + 8
    # stand on the floor below that point
    while cy > 1 and not s[cy - 1, cz, cx]: cy -= 1
    cave = (cx + 0.5, cy + 1.62, cz + 0.5)
else:
    cave = (128.5, mode / 2, 128.5)
views = [("A", (128.5, 74.0, 128.5, 20, 58)),
         ("B", (bx + 0.5, mode + 1 + 1.62, bz + 0.5, 30, 18)),
         ("C", (cave[0], cave[1], cave[2], 60, 15)),
         ("D", (-14.0, mode - 18.0, -14.0, 135, -28))]
os.makedirs("tmp/v", exist_ok=True)
files = []
for name, (px, py, pz, yaw, pitch) in views:
    o = f"tmp/v/{label}-{name}.ppm"
    subprocess.check_call(["./render", f, "256", "256", "64", str(px), str(py), str(pz), str(yaw), str(pitch), "426", "240", gr, o])
    files.append(o)
foot = ["frames/f86.5.png", "frames/f9.png", "frames/f92.png", "frames/f168.png"]
inp = []
for p in foot + files: inp += ["-i", p]
fc = "".join(f"[{i}]scale=426:240[s{i}];" for i in range(8))
fc += "".join(f"[s{i}]" for i in range(8)) + "xstack=inputs=8:layout=0_0|426_0|852_0|1278_0|0_240|426_240|852_240|1278_240"
subprocess.check_call(["ffmpeg", "-v", "error", "-y"] + inp + ["-filter_complex", fc, out])
print(out, "mode", mode, "plain-cam", bx, bz, "cave-cam", cave)
