"""Calibrate the C5 reach radius r for the TERRACE shell model.
Prediction for a surface column (x, z): OPEN iff its nearest kept seed (3D distance to (x+.5, 42.5, z+.5)) is AIR, or that
distance exceeds r (an unfilled region the growth front never reached by the fixed T). Truth: cell (x, 42, z) is air in the
tgen shell world. For each (rule, K, T) window: N worlds from random states, IoU(pred open, true open) over interior
columns as a function of r; report the best r and the IoU of the plain Voronoi rule (r = inf) for comparison.
Output: reach-calibration.json"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regrow
TGEN = HERE / 'build' / 'tgen.exe'
RS = np.arange(4.0, 30.01, 0.5)
DMIN = (1, 2, 3, 4, 6)   # truth: column top at least dmin below the plain (footage open fraction R2 .16, R1 .24)


def one(spec, state, tmp):
    log = tmp / 'seeds.txt'; blk = tmp / 'w.blocks'
    subprocess.run([str(TGEN), 'run', spec, str(state), str(blk)], capture_output=True, check=True, env={'GEN_SEEDLOG': str(log), 'SystemRoot': 'C:\\Windows'})
    a = np.fromfile(blk, np.uint8).reshape(64, 256, 256)
    solid = a != 0
    top = np.where(solid.any(axis=0), 63 - np.argmax(solid[::-1], axis=0), -1)   # [z, x] highest solid y
    depth = 42 - top                                             # 0 = plain, k = pit k deep
    s = np.loadtxt(log, dtype=np.int64, ndmin=2)                 # k x y z t
    s = s[s[:, 2] <= 42]
    sx, sy, sz, st = s[:, 1] + .5, s[:, 2] + .5, s[:, 3] + .5, s[:, 4] == 0
    zz, xx = np.mgrid[1:255, 1:255]
    px, pz = xx.ravel() + .5, zz.ravel() + .5
    best = np.full(px.size, np.inf); bair = np.zeros(px.size, bool)
    for i0 in range(0, sx.size, 256):
        d = (px[:, None] - sx[None, i0:i0+256]) ** 2 + (42.5 - sy[None, i0:i0+256]) ** 2 + (pz[:, None] - sz[None, i0:i0+256]) ** 2
        j = d.argmin(axis=1); dm = d[np.arange(px.size), j]
        upd = dm < best; best[upd] = dm[upd]; bair[upd] = st[i0:i0+256][j[upd]]
    out, frac = {}, {}
    for dmin in DMIN:
        t = (depth[1:255, 1:255] >= dmin).ravel(); frac[dmin] = float(t.mean())
        for r in list(RS) + [np.inf]:
            pred = bair | (best > r * r)
            u = (pred | t).sum()
            out[f'{dmin}|{float(r)}'] = float((pred & t).sum() / u) if u else 1.0
    return out, frac


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tmp = Path(tempfile.mkdtemp(prefix='reach-'))
    res = {}
    for v in ['U6s:K=1024,ntypes=8', 'U6s:K=4096,ntypes=8']:
        for rule, T, spec in regrow.shell_specs(v):
            curves = []; opens = []
            for i in range(n):
                c, o = one(spec, 0x9E3779B97F4A + 7919 * i, tmp); curves.append(c); opens.append(o)
            for dmin in DMIN:
                mean = {float(k.split('|')[1]): float(np.mean([c[k] for c in curves])) for k in curves[0] if k.startswith(f'{dmin}|')}
                rb = max((r for r in mean if r != float('inf')), key=lambda r: mean[r])
                cell = f'{v}|{rule}|{T}V|depth>={dmin}'
                res[cell] = {'best_r': rb, 'iou_best_r': mean[rb], 'iou_voronoi_only': mean[float('inf')],
                             'true_open_fraction': float(np.mean([o[dmin] for o in opens])), 'curve': mean}
                print(json.dumps({k: round(res[cell][k], 3) for k in ('best_r', 'iou_best_r', 'iou_voronoi_only', 'true_open_fraction')} | {'cell': cell}), flush=True)
    (HERE / 'reach-calibration.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
