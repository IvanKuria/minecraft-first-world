"""Spawn-height check (DEEP-REVIEW weak link 3: resetPos y = depth+10?) from this lane's R1 fall track.
Eye height above the plain = camera height (texture fit) - 0.3*sin(pitch).  Physics rd-132211: per tick yd-=.005, move,
yd*=.98; resetPos centres the bbox on y=74 so after the first move the eye is at bb.y0+1.62 = 74.72 (minus that tick's
fall).  Video time -> game time with factor 1/0.913 (DEEP-REVIEW) or free.  Fit reset tick time t0 and spawn eye E0."""
import json, math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
P = Path(__file__).parent
pts = [r for r in json.loads((P / 'track-r1-a.json').read_text()) if r.get('ok') and 1668 <= r['n'] <= 1675 and r['n'] != 1672]
pts_t = [float(l) for l in (P.parent / 'link/footage/frame-pts.txt').read_text().split()]
t = np.array([pts_t[r['n']] for r in pts]); e = np.array([r['h'] - .3 * math.sin(math.radians(r['pitch_deg'])) for r in pts])
def fall(n):
    """eye drop after n (real-valued) ticks from rest, linear interpolation inside a tick"""
    v, s, out = 0.0, 0.0, []
    S = [0.0]
    for k in range(400):
        v -= .005; s += v; v *= .98; S.append(-s)
    S = np.array(S); n = np.clip(n, 0, 399); i = np.floor(n).astype(int)
    return S[i] + (S[i + 1] - S[i]) * (n - i)
def model(p, t, speed):
    E0, t0 = p[0], p[1]
    return E0 - fall((t - t0) * 60 * speed)
for speed, lab in ((1 / .913, 'video 0.913x real (DEEP-REVIEW)'), (1.0, 'video = real time')):
    r = least_squares(lambda p: model(p, t, speed) - e, [10.7, 85.5])
    print(lab, 'E0 = %.2f  t0 = %.3f s  rms = %.3f' % (r.x[0], r.x[1], np.sqrt(np.mean(r.fun ** 2))))
r = least_squares(lambda p: model(p[:2], t, p[2]) - e, [10.7, 85.5, 1.09])
print('free speed: E0 = %.2f t0 = %.3f speed = %.3f rms = %.3f' % (r.x[0], r.x[1], r.x[2], np.sqrt(np.mean(r.fun ** 2))))
# fixed E0 = 10.72 (spawn eye 74.72 over plain 64): fit t0 only, compare with the visible reset (last cave frame 85.481, first air 85.532)
for E0 in (10.72, 10.0, 9.72):
    r = least_squares(lambda p: model([E0, p[0]], t, 1 / .913) - e, [85.5])
    print('E0 fixed %.2f: t0 = %.3f s rms %.3f  (visible reset between 85.481 and 85.532)' % (E0, r.x[0], np.sqrt(np.mean(r.fun ** 2))))
print('data (t, eye above plain):', [(round(a, 3), round(b, 2)) for a, b in zip(t, e)])
