"""TERRACE texture plane sweep (proof/terrace/sweep.py, unchanged functions) applied to the OPENING frames where this lane
has strong independent pose fits with the camera pitched down (frames 48-54, yaw 358-67 deg) plus the frozen frames
39/41/44 poses (yaw ~316).  Player at the spawn in all of them.  Output: per-column top-level labels relative to the plain,
in columns relative to floor(opening player)."""
import sys, json, math
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'terrace'))
import sweep as S
from popin_obs import frame
P = HERE
fx, fz = 0.0961, 0.8042
W = {r['n']: r for r in json.loads((P / 'wide-48-61.json').read_text())}
T = {r['n']: r for r in json.loads((P / 'track-open-a.json').read_text()) if r.get('ok')}
frames = []
for n in (39, 41, 44):
    r = T[n]; frames.append((n, math.radians(r['yaw_deg']), math.radians(r['pitch_deg']), r['h']))
for n in (48, 49, 51):
    r = W[n]; frames.append((n, r['yaw_rad'], r['pitch_rad'], r['h']))
# walking frames 55/56 (player moved by < 0.5 block from the spawn; unwrap by continuity from frame 54)
WW = {r['n']: r for r in json.loads((P / 'wide-walk-55-140.json').read_text()) if r.get('score')}
phase_of = {}
for n in (55, 56):
    r = WW[n]
    if True: continue   # disabled: adding 55/56 lowered pooled agreement (82 -> 71 labels); poses during the walk less certain
    d = [((r['player_frac'][i] - (fx, fz)[i]) + .5) % 1 - .5 for i in range(2)]
    frames.append((n, math.radians(r['yaw_deg']), math.radians(r['pitch_deg']), r['h']))
    phase_of[n] = (fx + d[0], fz + d[1])
per = []
for n, yaw, pitch, h in frames:
    img = Image.fromarray(frame(n)[:, :, ::-1].copy())
    fs = S.frame_scores(img, yaw, pitch, h, phase_of.get(n, (fx, fz)))
    per.append(fs)
    rr = S.combine([fs]); lab = S.assign(rr)
    hist = {k: sum(1 for v in lab.values() if v[0] == k) for k in S.KS}
    print(n, 'yaw %.1f pitch %.1f h %.2f' % (math.degrees(yaw) % 360, math.degrees(pitch), h), 'labelled', len(lab), {k: v for k, v in hist.items() if v}, flush=True)
res = S.combine(per)
out = {}
for rmin, margin in ((.45, .15), (.35, .12), (.55, .2)):
    lab = S.assign(res, rmin, margin)
    out[f'{rmin}/{margin}'] = [[x, z, v[0], round(v[1], 3), round(v[2], 3), int(v[3])] for (x, z), v in sorted(lab.items())]
    hist = {k: sum(1 for v in lab.values() if v[0] == k) for k in S.KS}
    print('pooled', rmin, margin, 'labelled', len(lab), {k: v for k, v in hist.items() if v})
(P / 'sweep-opening.json').write_text(json.dumps({'phase': [fx, fz], 'frames': [[n, yaw, pitch, h] for n, yaw, pitch, h in frames], 'labels': out,
                                                   'scores': [[x, z, {str(k): [round(r, 3), int(nn)] for k, (r, nn) in d.items()}] for (x, z), d in sorted(res.items())]}))
lab = {(x, z): k for x, z, k, *_ in out['0.45/0.15']}
xs = [k[0] for k in lab]; zs = [k[1] for k in lab]
if lab:
    for z in range(min(zs), max(zs) + 1):
        print('z=%4d ' % z + ''.join('%3s' % (lab[(x, z)] if (x, z) in lab else '.') for x in range(min(xs), max(xs) + 1)))
    print('x from', min(xs), 'to', max(xs))
