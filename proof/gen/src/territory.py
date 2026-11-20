"""Seed -> territory relation for multi-type growth into UNSET cells (for route C5 in proof/IDEA-MAP.md).
usage: territory.py world.blocks owner.u16 seeds.txt [capy]
seeds.txt lines: k x y z type   (from GEN_SEEDLOG; type 0 air, 1 rock)
Reports, for each surviving seed: territory size, centroid offset from the seed coordinate, and how often a
visible (surface-connected) cave contains the seed cell."""
import sys, json
import numpy as np
from scipy import ndimage
W, H, D = 256, 256, 64
blk = np.fromfile(sys.argv[1], np.uint8).reshape(D, H, W)
own = np.fromfile(sys.argv[2], np.uint16).reshape(D, H, W)
seeds = np.loadtxt(sys.argv[3], dtype=np.int64, ndmin=2)
capy = int(sys.argv[4]) if len(sys.argv) > 4 else 42
K = len(seeds)
ids = own.ravel().astype(np.int64)
cnt = np.bincount(ids, minlength=K + 1)
yy, zz, xx = np.indices((D, H, W))
cx = np.bincount(ids, xx.ravel(), K + 1); cy = np.bincount(ids, yy.ravel(), K + 1); cz = np.bincount(ids, zz.ravel(), K + 1)
rows = []
air = blk == 0
# surface-connected air below cap: components of air (y<=capy) touching the layer y==capy
lab, n = ndimage.label(air[:capy + 1])
surf = set(np.unique(lab[capy])) - {0}
for k in range(1, K + 1):
    if cnt[k] == 0:
        continue  # overwritten seed (another seed placed later at same cell) or never placed
    _, x, y, z, t = seeds[k - 1]
    c = np.array([cx[k], cy[k], cz[k]]) / cnt[k]
    off = c - np.array([x + 0.5 - 0.5, y, z])
    vis = bool(t == 0 and y <= capy and lab[y, z, x] in surf)
    rows.append({"k": int(k), "type": int(t), "x": int(x), "y": int(y), "z": int(z), "size": int(cnt[k]),
                 "centroid_offset": [round(float(v), 2) for v in off], "offset_norm": round(float(np.linalg.norm(off)), 2), "air_seed_in_surface_cave": vis})
airrows = [r for r in rows if r["type"] == 0]
summ = {
    "seeds_placed": int(K), "seeds_with_territory": len(rows), "air_seeds_with_territory": len(airrows),
    "median_territory_size": float(np.median([r["size"] for r in rows])) if rows else 0,
    "median_centroid_offset_blocks": float(np.median([r["offset_norm"] for r in rows])) if rows else 0,
    "p90_centroid_offset_blocks": float(np.percentile([r["offset_norm"] for r in rows], 90)) if rows else 0,
    "median_territory_radius_equiv": float(np.median([(3 * r["size"] / (4 * np.pi)) ** (1 / 3) for r in rows])) if rows else 0,
    "air_seeds_in_surface_connected_caves": sum(r["air_seed_in_surface_cave"] for r in airrows),
    "surface_connected_air_components": len(surf),
}
print(json.dumps({"summary": summ, "seeds": rows[:2000]}, indent=0))
