"""Opening world-edge registration with a SECOND view direction.  Frames 39/41(/44) look toward yaw ~316 (x=0 / z=0 edges);
frames 50-54 (player still at the spawn: same texture phase .10/.80, h .68) look toward yaw ~60-67 (z=0 / x=256 edges).
Same-height boundary model (plain = top of world) = frozen geometry/opening-edge/fit_opening.boundary; edge rows measured
with the same luminance-transition rule; one common row bias per view group, bounded +-1 px."""
import sys, json, math
from pathlib import Path
import numpy as np, cv2
EDGE = Path(__file__).resolve().parents[2] / 'geometry/opening-edge'
sys.path.insert(0, str(EDGE)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'link'))
import fit_opening as m
from lcg import reset_xz
from popin_obs import frame, skymask
P = Path(__file__).parent
xs = m.xs

def extract(n):
    im = frame(n); lum = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    sky = skymask(im); ys = []
    for x in xs.astype(int):
        col = ~sky[:, x - 3:x + 4].any(axis=1) if False else ~np.median(sky[:, x - 3:x + 4], axis=1).astype(bool)
        first = int(np.argmax(col)) if col.any() else -1
        if first < 3: ys.append(np.nan); continue
        line = np.median(lum[:, x - 3:x + 4], axis=1)
        grad = np.maximum(0, line[:-1] - line[1:])
        w0, w1 = max(0, first - 5), min(358, first + 4)
        peak = w0 + int(np.argmax(grad[w0:w1]))
        inds = np.arange(max(w0, peak - 1), min(w1, peak + 2)); wt = grad[inds]
        ys.append(float(np.sum((inds + .5) * wt) / (wt.sum() + 1e-10)))
    return np.array(ys)

def pose_group_frozen():
    pose = m.pose      # best fixed70 (frozen)
    obs = np.array(json.loads((EDGE / 'fit-results.json').read_text())['observed_edge_rows'])
    return [(fr, obs[i]) for i, fr in enumerate(pose['frames'][:2])]

def pose_group_new(wide_json, frames):
    W = {r['n']: r for r in json.loads((P / wide_json).read_text())}
    out = []
    for n in frames:
        r = W[n]
        fr = {'frame_name': str(n), 'camera_parameters': [r['yaw_rad'], r['pitch_rad'], r['h'], r['cam'][0], r['cam'][1]]}
        out.append((fr, extract(n)))
    return out

def group_rmse(pos, group):
    with np.errstate(all='ignore'):
        raw = np.array([m.boundary(pos, fr, 0) - ob for fr, ob in group])
    ok = np.isfinite(raw)
    if ok.sum() < .7 * raw.size: return np.inf, 0
    bias = float(np.clip(-np.nanmean(raw), -1, 1))
    return float(np.sqrt(np.nanmean((raw + bias) ** 2))), int(ok.sum())

if __name__ == '__main__':
    m.F = 180 / np.tan(np.deg2rad(35))
    wide_json = sys.argv[1]; frames = [int(v) for v in sys.argv[2].split(',')]
    gA = pose_group_frozen(); gB = pose_group_new(wide_json, frames)
    for fr, ob in gB: print('frame', fr['frame_name'], 'edge rows (every 8th sample):', np.round(ob[::8], 1))
    step = 2
    grid = [(x + .096, z + .804) for x in range(0, 256, step) for z in range(0, 256, step)]
    A = np.array([group_rmse(list(p), gA)[0] for p in grid]); B = np.array([group_rmse(list(p), gB)[0] for p in grid])
    J = np.sqrt((A ** 2 + B ** 2) / 2)
    res = {}
    for name, v in (('A_yaw316', A), ('B_yaw60', B), ('joint', J)):
        b = np.nanmin(v); i = int(np.nanargmin(v)); near = [grid[k] for k in np.nonzero(v < b + .1)[0]]
        res[name] = {'best': float(b), 'best_xz': grid[i], 'region_x': [min(p[0] for p in near), max(p[0] for p in near)],
                     'region_z': [min(p[1] for p in near), max(p[1] for p in near)], 'n_region_cells_step2': len(near)}
        print(name, res[name])
    NAMED = [99100327271809, 266841333641652, 233513965841111, 14593523350806, 170375616623189, 128528575296967, 88325364808660]
    for s in NAMED:
        x0, z0, _ = reset_xz(s)
        a = group_rmse([x0, z0], gA)[0]; b = group_rmse([x0, z0], gB)[0]
        print('row', s, [round(x0, 2), round(z0, 2)], 'A %.3f B %.3f joint %.3f' % (a, b, math.sqrt((a * a + b * b) / 2)))
        res[str(s)] = {'opening_xz': [x0, z0], 'A': a, 'B': b}
    np.save(P / 'edge2-A.npy', A.reshape(128, 128)); np.save(P / 'edge2-B.npy', B.reshape(128, 128))
    (P / 'edge-two-views.json').write_text(json.dumps(res, indent=1))
