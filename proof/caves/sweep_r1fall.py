"""TERRACE plane sweep on the R1 fall frames 1669-1676 (this lane's grass track; camera 5.6 -> 1.7 above the plain),
columns relative to floor(R1 player).  Compared with TERRACE sweep-r1 (frames 1662-1668)."""
import sys, json, math
from pathlib import Path
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'terrace'))
import sweep as S
from popin_obs import frame
TR = [r for r in json.loads((HERE / 'track-r1-a.json').read_text()) if r.get('ok') and 1669 <= r['n'] <= 1676 and r['n'] not in (1672, 1676)]
per = []
for r in TR:
    img = Image.fromarray(frame(r['n'])[:, :, ::-1].copy())
    fs = S.frame_scores(img, math.radians(r['yaw_deg']), math.radians(r['pitch_deg']), r['h'], tuple(r['player']))
    per.append(fs); lab = S.assign(S.combine([fs]))
    print(r['n'], 'h %.2f' % r['h'], 'labelled', len(lab), flush=True)
lab = S.assign(S.combine(per))
T = {(x, z): k for x, z, k, *_ in json.loads((HERE.parent / 'terrace/sweep-r1.json').read_text())['labels']}
near = {c: v for c, v in lab.items() if math.hypot(c[0] + .5 - .47, c[1] + .5 - .86) < 10}
common = [c for c in near if c in T]; agree = [c for c in common if near[c][0] == T[c]]
print('new near-field labels', len(near), 'overlap with TERRACE', len(common), 'agree', len(agree))
for c in common:
    if near[c][0] != T[c]: print(' disagree', c, near[c][0], T[c])
new = {c: v for c, v in near.items() if c not in T}
print('columns not in TERRACE r1 map:', len(new), sorted((c, v[0]) for c, v in new.items()))
json.dump({'frames': [r['n'] for r in TR], 'labels': [[c[0], c[1], v[0], round(v[1], 3), round(v[2], 3), int(v[3])] for c, v in sorted(lab.items())],
           'near_overlap': len(common), 'near_agree': len(agree)}, open(HERE / 'sweep-r1fall.json', 'w'))
