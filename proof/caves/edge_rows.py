"""Apply the FROZEN opening-edge boundary model (geometry/opening-edge, frozen 2026-09-27, candidate-blind) to the opening
positions of the RNG rows, reporting the far-plane offset 0 (plain = top of world, DEEP-REVIEW) separately from the
scorer's free 0..63 offset choice.  No scorer code is modified; the same bias rule (+-1 px, fit on frames 39/41, 44 held out)."""
import sys, json
from pathlib import Path
import numpy as np
EDGE = Path(__file__).resolve().parents[2] / 'geometry/opening-edge'
sys.path.insert(0, str(EDGE)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'link'))
import fit_opening as m
from lcg import reset_xz
obs = np.array(json.loads((EDGE / 'fit-results.json').read_text())['observed_edge_rows'])

def score(pos, deltas=(0, 1)):
    out = {}
    for delta in deltas:
        best = None
        for variant, pose in enumerate(m.poses):
            m.F = 180 / np.tan(np.deg2rad(pose['fov'] / 2))
            with np.errstate(all='ignore'):
                raw = np.array([m.boundary(pos, fr, delta) for fr in pose['frames']]) - obs
            if not np.isfinite(raw).all(): continue
            bias = float(np.clip(-raw[:2].mean(), -1, 1)); res = raw + bias
            tr = float(np.sqrt(np.mean(res[:2] ** 2))); ho = float(np.sqrt(np.mean(res[2] ** 2)))
            if best is None or tr < best['train_rmse_px']:
                best = {'variant': variant, 'bias': bias, 'train_rmse_px': tr, 'heldout_rmse_px': ho}
        out[delta] = best
    return out

if __name__ == '__main__':
    P = Path(__file__).parent
    states = [int(l) for l in (P.parent / 'fingerprint/states-baseline175.txt').read_text().split()]
    named = {99100327271809, 266841333641652, 233513965841111, 14593523350806, 170375616623189, 128528575296967}
    rows = []
    for s in states:
        x0, z0, _ = reset_xz(s)
        sc = score([x0, z0])
        rows.append({'state': s, 'opening_xz': [x0, z0], 'named': s in named,
                     'same_height': sc[0], 'one_lower': sc[1]})
    # reference: best continuous position (same height) from the lane's own fit
    rows.sort(key=lambda r: r['same_height']['train_rmse_px'] if r['same_height'] else 1e9)
    (P / 'edge-rows.json').write_text(json.dumps(rows, indent=1))
    tr = np.array([r['same_height']['train_rmse_px'] for r in rows if r['same_height']])
    print('175 rows, same-height train RMSE quantiles', np.percentile(tr, [0, 5, 25, 50, 75, 100]).round(2))
    for r in rows[:12]:
        print('best', r['state'], [round(v, 2) for v in r['opening_xz']], round(r['same_height']['train_rmse_px'], 3), round(r['same_height']['heldout_rmse_px'], 3))
    for r in rows:
        if r['named']:
            print('NAMED', r['state'], [round(v, 2) for v in r['opening_xz']], 'same-height train/held', round(r['same_height']['train_rmse_px'], 3), round(r['same_height']['heldout_rmse_px'], 3),
                  '| one-lower', round(r['one_lower']['train_rmse_px'], 3), 'rank', rows.index(r) + 1)
