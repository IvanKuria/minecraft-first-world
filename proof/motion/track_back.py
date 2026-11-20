"""MOTION lane: grass-texture tracking BACKWARD from a known stationary-window pose toward the reset, and forward,
reusing the caves lane's validated tracker unchanged (caves/grasstrack.py + grasstrack2.step).  Frames are cached in
proof/motion/frames (not the caves directory).
usage: track_back.py n_start n_stop yaw,pitch,h,cx,cz out.json      (n_stop exclusive; direction from sign)"""
import sys, json, math
from pathlib import Path
P = Path(__file__).parent
sys.path.insert(0, str(P.parent / 'caves'))
import popin_obs
popin_obs.FR = P / 'frames'; popin_obs.FR.mkdir(exist_ok=True)
import grasstrack2 as g2

n0, n1 = int(sys.argv[1]), int(sys.argv[2])
prev = [float(v) for v in sys.argv[3].split(',')]
out = P / sys.argv[4]
step = -1 if n1 < n0 else 1
res = []
for n in range(n0, n1, step):
    st = g2.step(n, prev)
    if st is None:
        res.append({'n': n, 'ok': False}); out.write_text(json.dumps(res, indent=0)); print(n, 'no roi', flush=True); continue
    r, mv, npx = st
    p = r.x.tolist(); sc = -r.fun
    s_, c_, sp, cp = math.sin(p[0]), math.cos(p[0]), math.sin(p[1]), math.cos(p[1])
    player = [p[3] + .3 * s_ * cp, p[4] - .3 * c_ * cp]
    rec = {'n': n, 'ok': True, 'score': sc, 'move': mv, 'npx': npx, 'yaw_deg': math.degrees(p[0]) % 360,
           'pitch_deg': math.degrees(p[1]), 'h': p[2], 'cam': p[3:5], 'player': player, 'params': p}
    res.append(rec); out.write_text(json.dumps(res, indent=0))
    print(n, round(sc, 3), round(rec['yaw_deg'], 2), round(rec['pitch_deg'], 2), round(p[2], 3), [round(v, 4) for v in player], flush=True)
    if sc > .25: prev = p
