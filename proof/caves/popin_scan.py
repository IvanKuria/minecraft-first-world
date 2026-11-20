"""Brute-force scan of the threshold pop-in model over (floor(X) mod 16, floor(Z) mod 16, yaw, pitch, eye)."""
import sys, json, math, time
from pathlib import Path
import numpy as np
import popin
from popin_fit import us, vs, R0, R1, obs_masks, FR
P = Path(__file__).parent
FX, FZ = 0.0961, 0.8042           # opening texture phase (player position fractions)

def frame_scores(ci, masks):
    """per frame: best IoU over thresholds, using bincounts. ci: chunk index map (-1 sky)."""
    valid = ci >= 0
    idx = ci[valid]
    out = []
    for n in FR:
        ob = masks[n]
        tot = np.bincount(idx, minlength=1024).astype(float)
        ter = np.bincount(idx, weights=ob[valid].astype(float), minlength=1024)
        cum_tot = np.concatenate([[0], np.cumsum(tot)]); cum_ter = np.concatenate([[0], np.cumsum(ter)])
        O = ob.sum()
        union = cum_tot + O - cum_ter
        iou = np.where(union > 0, cum_ter / np.maximum(union, 1), 0)
        j = int(np.argmax(iou))
        out.append((float(iou[j]), j, int(cum_tot[j])))
    return out

def scan(yaws, pitches, eyes, base=(128, 128), masks=None):
    masks = masks or obs_masks()
    res = []
    for eye in eyes:
        for p in pitches:
            for yw in yaws:
                yaw, pitch = math.radians(yw), math.radians(p)
                cx_, cy_, cz_ = popin.cam_pos(0.0, eye, 0.0, yaw, pitch)
                rx, rz, t = popin.plane_hits(yaw, pitch, cy_, us, vs)
                for xo in range(16):
                    for zo in range(16):
                        X = base[0] + xo + FX; Z = base[1] + zo + FZ
                        hx = X + cx_ + rx; hz = Z + cz_ + rz
                        ok = np.isfinite(hx) & (hx >= 0) & (hx < 256) & (hz >= 0) & (hz < 256)
                        ci = np.full(hx.shape, -1)
                        ci[ok] = (np.floor(hx[ok] / 16).astype(int) + 48) * 16 + np.floor(hz[ok] / 16).astype(int)
                        fs = frame_scores(ci, masks)
                        res.append((float(np.mean([f[0] for f in fs])), eye, p, yw, xo, zo, [f[0] for f in fs], [f[1] for f in fs]))
    return res

if __name__ == '__main__':
    t0 = time.time()
    masks = obs_masks()
    yaws = np.arange(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    pitches = [float(v) for v in sys.argv[4].split(',')]
    eyes = [float(v) for v in sys.argv[5].split(',')]
    res = scan(yaws, pitches, eyes, masks=masks)
    res.sort(key=lambda r: -r[0])
    for r in res[:25]:
        print(round(r[0], 4), r[1], r[2], round(r[3], 2), r[4], r[5], [round(v, 3) for v in r[6]], r[7])
    print('n', len(res), 'time', round(time.time() - t0, 1))
    tag = sys.argv[6] if len(sys.argv) > 6 else 'scan'
    (P / f'popin-{tag}.json').write_text(json.dumps(res))
