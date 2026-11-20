"""MOTION lane: regional phase correlation (near wall vs far corridor) between consecutive pre-reset cave frames.
Equal shifts = pure mouse rotation; different shifts = translation (parallax)."""
import sys, json
from pathlib import Path
import numpy as np, cv2
P = Path(__file__).parent
sys.path.insert(0, str(P.parent / 'caves'))
import popin_obs
popin_obs.FR = P / 'frames'
REG = {'near_left': (slice(40, 320), slice(0, 190)), 'far_centre': (slice(30, 190), slice(260, 560)),
       'floor_centre': (slice(220, 350), slice(230, 520))}
out = []
for a in map(int, sys.argv[1].split(',')):
    A = cv2.cvtColor(popin_obs.frame(a), cv2.COLOR_BGR2GRAY).astype(np.float32)
    B = cv2.cvtColor(popin_obs.frame(a + 1), cv2.COLOR_BGR2GRAY).astype(np.float32)
    rec = {'pair': [a, a + 1]}
    for k, (ys, xs) in REG.items():
        w = cv2.createHanningWindow((xs.stop - xs.start, ys.stop - ys.start), cv2.CV_32F)
        (sx, sy), r = cv2.phaseCorrelate(A[ys, xs], B[ys, xs], w)
        rec[k] = [round(sx, 3), round(sy, 3), round(r, 3)]
    out.append(rec); print(rec)
(P / 'cave-parallax.json').write_text(json.dumps(out, indent=1))
