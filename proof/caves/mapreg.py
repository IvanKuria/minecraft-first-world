"""Register an R1-segment frame to the R1 position by matching its surface open/closed map against R1MAP's map
(footage-obs-r1.json, relative to floor(R1)).  The frame's own pose (yaw, pitch, camera height above the plateau,
player fraction) comes from an independent wide grass fit; only the integer offset floor(player) - floor(R1) is searched."""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'c5'))
import obs_footage as OF
import widefit
from popin_obs import frame, FR
R1MAP = {(x, z): v for x, z, v in json.loads((HERE.parent / 'r1map/footage-obs-r1.json').read_text())['labels']}

def frame_labels(n):
    o, vh, npx = widefit.wide(n, v0=150)
    sc, q, _ = o[0]
    yaw, pitch, h, cx, cz = q
    s_, c_, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    px, pz = (cx + .3 * s_ * cp) % 1, (cz - .3 * c_ * cp) % 1
    X, Z, ok = OF.project(yaw, pitch, h, phase=(px, pz))
    frame(n)
    sky, green, openev = OF.classes(FR / f'n{n:04d}.png')
    d = {}
    for lab, m in (('closed', green), ('open', openev)):
        sel = ok & m
        keys, cnt = np.unique(np.stack([X[sel], Z[sel]], 1), axis=0, return_counts=True)
        for (x, z), k in zip(keys.tolist(), cnt.tolist()):
            d.setdefault((x, z), [0, 0])[0 if lab == 'closed' else 1] += k
    return OF.labels(d), {'score': sc, 'yaw_deg': math.degrees(yaw) % 360, 'pitch_deg': math.degrees(pitch), 'h': h, 'player_frac': [px, pz]}

def match(lab, R=40):
    out = []
    for sx in range(-R, R + 1):
        for sz in range(-R, R + 1):
            common = [(x + sx, z + sz) for (x, z) in lab if (x + sx, z + sz) in R1MAP]
            if len(common) < 40: continue
            agree = sum(lab[(x - sx, z - sz)] == R1MAP[(x, z)] for x, z in common)
            op = [(c) for c in common if R1MAP[c] == 1 or lab[(c[0] - sx, c[1] - sz)] == 1]
            oi = sum(1 for c in op if R1MAP[c] == 1 and lab[(c[0] - sx, c[1] - sz)] == 1)
            out.append((agree / len(common), oi / max(len(op), 1), len(common), sx, sz))
    out.sort(key=lambda r: -(r[0] + r[1]))
    return out

if __name__ == '__main__':
    res = {}
    for n in map(int, sys.argv[1].split(',')):
        lab, pose = frame_labels(n)
        m = match(lab)
        nopen = sum(lab.values())
        print(n, 'pose', {k: (round(v, 3) if isinstance(v, float) else [round(u, 3) for u in v]) for k, v in pose.items()}, 'labels', len(lab), 'open', nopen)
        for r in m[:6]:
            print('   agree %.3f openIoU %.3f n %d shift (%d,%d)' % r)
        res[n] = {'pose': pose, 'n_labels': len(lab), 'n_open': nopen, 'top': [list(r) for r in m[:20]]}
    (HERE / 'mapreg.json').write_text(json.dumps(res, indent=1))
