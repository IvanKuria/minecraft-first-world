"""Pool open/closed labels of consecutive well-fitted frames (relative offsets from texture-phase continuity) and register
the pooled map to R1MAP by an integer shift search."""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import mapreg as M
frames = [int(v) for v in sys.argv[1].split(',')]; ref = frames[-1]
L = {}; poses = {}
for n in frames:
    lab, pose = M.frame_labels(n); L[n] = lab; poses[n] = pose
    print(n, 'fit %.3f yaw %.1f pitch %.1f h %.2f frac (%.3f,%.3f) labels %d' % (pose['score'], pose['yaw_deg'], pose['pitch_deg'], pose['h'], *pose['player_frac'], len(lab)), flush=True)
good = [n for n in frames if poses[n]['score'] > .6]
# unwrap player positions backwards from ref by continuity (|step| < .5)
pos = {ref: np.array(poses[ref]['player_frac'])}
order = sorted(good, reverse=True)
for a, b in zip(order[:-1], order[1:]):        # b is earlier than a
    fa, fb = np.array(poses[a]['player_frac']), np.array(poses[b]['player_frac'])
    d = (fb - fa + .5) % 1 - .5
    pos[b] = pos[a] + d
pooled = {}
for n in good:
    off = np.floor(pos[n]).astype(int)          # floor(p_n) - floor(p_ref) in the ref frame (p_ref in [0,1))
    for (x, z), v in L[n].items():
        k = (x + off[0], z + off[1]); pooled.setdefault(k, []).append(v)
lab = {k: int(round(np.mean(v))) for k, v in pooled.items() if abs(np.mean(v) - .5) > .15}
print('pooled frames', good, 'positions rel ref', {n: [round(u, 2) for u in pos[n]] for n in good}, 'labels', len(lab))
m = M.match(lab)
for r in m[:8]: print('   agree %.3f openIoU %.3f n %d shift (%d,%d)' % r)
(HERE / f'mapreg-pool-{ref}.json').write_text(json.dumps({'frames': good, 'positions_rel_ref': {n: pos[n].tolist() for n in good},
    'poses': poses, 'n_labels': len(lab), 'top': [list(r) for r in m[:30]]}, indent=1))
