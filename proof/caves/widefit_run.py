import sys, json, math
from pathlib import Path
import widefit
P = Path(__file__).parent
out = P / sys.argv[2]; res = []
for n in map(int, sys.argv[1].split(',')):
    o, vh, npx = widefit.wide(n)
    sc, q, s0 = o[0]
    s_, c_, sp, cp = math.sin(q[0]), math.cos(q[0]), math.sin(q[1]), math.cos(q[1])
    rec = {'n': n, 'score': sc, 'yaw_rad': q[0], 'pitch_rad': q[1], 'h': q[2], 'cam': [q[3], q[4]],
           'player_frac': [(q[3] + .3 * s_ * cp) % 1, (q[4] - .3 * c_ * cp) % 1], 'horizon_row': vh,
           'alternatives': [[a[0], a[1]] for a in o[1:4]]}
    res.append(rec); out.write_text(json.dumps(res, indent=1))
    print(n, round(sc, 3), round(math.degrees(q[0]) % 360, 2), round(math.degrees(q[1]), 2), round(q[2], 3), [round(v, 3) for v in rec['player_frac']], flush=True)
