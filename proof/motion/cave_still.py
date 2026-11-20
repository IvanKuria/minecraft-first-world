"""MOTION lane: frame-to-frame image shift (phase correlation) and mean abs difference on the cave frames before a
reset.  A translating camera next to a wall shifts the image by several px per 0.01 block."""
import sys, json
from pathlib import Path
import numpy as np, cv2
P = Path(__file__).parent
sys.path.insert(0, str(P.parent / 'caves'))
import popin_obs
popin_obs.FR = P / 'frames'
out = {}
for rng in sys.argv[1:]:
    a, b = map(int, rng.split('-'))
    rows = []
    for n in range(a, b):
        A = cv2.cvtColor(popin_obs.frame(n), cv2.COLOR_BGR2GRAY).astype(np.float32)
        B = cv2.cvtColor(popin_obs.frame(n + 1), cv2.COLOR_BGR2GRAY).astype(np.float32)
        (sx, sy), resp = cv2.phaseCorrelate(A, B)
        rows.append({'pair': [n, n + 1], 'shift_px': [round(sx, 3), round(sy, 3)], 'response': round(resp, 3),
                     'mean_abs_diff': round(float(np.abs(A - B).mean()), 3)})
        print(n, n + 1, rows[-1])
    out[rng] = rows
(P / 'cave-still.json').write_text(json.dumps(out, indent=1))
