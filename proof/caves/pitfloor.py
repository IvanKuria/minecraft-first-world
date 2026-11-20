"""Depth of the lit cobble floor of the large hole the player approaches after R1 (frames ~1730-1745).
1) independent wide grass fit -> yaw, pitch, camera height above the grass plane h_g, camera position (tile grid);
2) with that pose and grid, sweep a horizontal plane at depth h_s below the camera and correlate the authentic ROCK tile
   (terrain.png tile 1) with the lit grey pixels.  All block faces share the integer grid, so no new phase is fitted.
   Peak h_s - h_g = floor depth below the grass plane (whole blocks expected)."""
import sys, json, math
from pathlib import Path
import numpy as np, cv2
import grasstrack as g
import widefit
from popin_obs import frame
P = Path(__file__).parent
ROCK = g.TEX[:16, 16:32, :].mean(axis=2).astype(float)

def rock_corr(yaw, pitch, cx, cz, hs, x, y, obs):
    out = []
    for h in hs:
        ox, oz = g.ground(yaw, pitch, h, x, y)
        ok = np.isfinite(ox)
        if ok.sum() < 200: out.append(np.nan); continue
        pr = g.lookup(cx + ox[ok], cz + oz[ok], tile=ROCK); o = obs[ok]
        pr = pr - pr.mean(); o = o - o.mean()
        out.append(float(pr @ o / (np.linalg.norm(pr) * np.linalg.norm(o) + 1e-9)))
    return np.array(out)

if __name__ == '__main__':
    res = []
    for n in map(int, sys.argv[1].split(',')):
        o, vh, npx = widefit.wide(n, v0=150)
        sc, q, _ = o[0]
        yaw, pitch, hg, cx, cz = q
        im = frame(n).astype(float)
        b, gg, r = im[..., 0], im[..., 1], im[..., 2]
        lum = (b + gg + r) / 3
        grey = (np.abs(r - gg) < 18) & (np.abs(gg - b) < 18) & (lum > float(__import__("os").environ.get("LUMMIN", "110")))
        ys, xs = np.nonzero(grey)
        sel = ys > vh + 3
        xs, ys = xs[sel].astype(float), ys[sel].astype(float); obs = lum[ys.astype(int), xs.astype(int)]
        hs = np.arange(hg + .5, hg + 25, .05)
        c = rock_corr(yaw, pitch, cx, cz, hs, xs, ys, obs)
        k = int(np.nanargmax(c)); top3 = np.argsort(-np.nan_to_num(c, nan=-1))[:40]
        # local maxima list
        loc = [i for i in range(1, len(c) - 1) if c[i] >= c[i - 1] and c[i] >= c[i + 1] and c[i] > .1]
        loc = sorted(loc, key=lambda i: -c[i])[:5]
        rec = {'n': n, 'grass_fit': {'score': sc, 'yaw_deg': math.degrees(yaw) % 360, 'pitch_deg': math.degrees(pitch), 'h_grass': hg},
               'n_grey_px': int(len(xs)), 'peak_h_stone': float(hs[k]), 'peak_corr': float(c[k]),
               'depth_below_grass_plane': float(hs[k] - hg), 'local_maxima': [[round(float(hs[i] - hg), 2), round(float(c[i]), 3)] for i in loc]}
        res.append(rec)
        print(n, 'grass fit %.3f yaw %.1f pitch %.1f h_g %.2f | grey px %d | stone peak at depth %.2f below grass plane (corr %.3f); maxima %s' % (
            sc, rec['grass_fit']['yaw_deg'], rec['grass_fit']['pitch_deg'], hg, len(xs), rec['depth_below_grass_plane'], c[k], rec['local_maxima']), flush=True)
        (P / 'pitfloor.json').write_text(json.dumps(res, indent=1))
