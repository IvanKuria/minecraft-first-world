"""C5 task 1: surface open/closed column map from the second-reset frames, registered to the player.

Inputs (all pre-existing, measured before this lane):
  frames  proof/link/frames/r2-153.{054,158,313,520}.png (2011 original-channel 640x360 copy; LINK's frozen set)
  poses   geometry/edge-registration/four-texture-poses.json (yaw, pitch, camera height above the nearby grass plane)
  phase   geometry/phase2 three-view common player phase at FOV 70: (.8118, .8185)
  camera  LINK scorer model: f = 180/tan(35deg), pixel centres, 0.3-block setback, zero roll.
Pixel classes: SKY (B>200 & B-R>60), GREEN (G>R+15 & G>B+15), else OPEN-EVIDENCE (stone grey or dark = something
below the grass plane seen through the plane).  Every non-sky ray is intersected with the grass-top plane (camera
height h above it); the crossing column gets one vote of its class.  Under GEN's flat-ceiling family a crossing
column is OPEN iff the surface cell is air.  In the footage the plane is NOT flat (1-2 block terraces), which makes
this an approximation: green side faces of raised terraces vote "closed" for the column in front of them.

Outputs c5/footage-obs.json: per column (dx,dz relative to floor(player)) votes per frame; consensus labels;
cross-frame consistency (the registration check) under pose and phase perturbations.
"""
import json, math
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
ROOT = PROOF.parent
F = 180 / math.tan(math.radians(35))
PHASE = (.8118, .8185)
NAMES = ['r2-153.054', 'r2-153.158', 'r2-153.313', 'r2-153.520']
DMAX = 32.0


def classes(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    sky = (B > 200) & (B - R > 60)
    green = (~sky) & (G > R + 15) & (G > B + 15)
    openev = (~sky) & (~green)
    return sky, green, openev


def project(yaw, pitch, h, phase=PHASE, dyaw=0.0, dpitch=0.0, dh=0.0):
    yaw += dyaw; pitch += dpitch; h += dh
    X, Y = np.meshgrid(np.arange(640) + .5, np.arange(360) + .5)
    rx = (X - 320) / F; ry = (Y - 180) / F
    s, c, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    dx = c * rx - s * sp * ry + s * cp
    dy = -cp * ry - sp
    dz = s * rx + c * sp * ry - c * cp
    camx = phase[0] - .3 * s * cp; camz = phase[1] + .3 * c * cp
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(dy < -1e-6, -h / dy, np.nan)
    Xp = camx + t * dx; Zp = camz + t * dz
    dist = np.hypot(Xp - camx, Zp - camz)
    ok = np.isfinite(t) & (dist <= DMAX)
    return np.floor(Xp).astype(np.int64), np.floor(Zp).astype(np.int64), ok


def column_votes(perturb=None):
    poses = json.loads((ROOT / 'geometry/edge-registration/four-texture-poses.json').read_text())['frames']
    per = []
    for name, fr in zip(NAMES, poses):
        yaw, pitch, h = fr['yaw_pitch_height']
        kw = (perturb or {}).get(name, {})
        cx, cz, ok = project(yaw, pitch, h, **kw)
        sky, green, openev = classes(PROOF / 'link/frames' / f'{name}.png')
        d = {}
        for lab, m in (('closed', green), ('open', openev)):
            sel = ok & m
            keys, cnt = np.unique(np.stack([cx[sel], cz[sel]], 1), axis=0, return_counts=True)
            for (x, z), n in zip(keys.tolist(), cnt.tolist()):
                d.setdefault((x, z), [0, 0])[0 if lab == 'closed' else 1] += n
        per.append(d)
    return per


def labels(d, minpix=12, frac=.6):
    out = {}
    for k, (c, o) in d.items():
        n = c + o
        if n < minpix: continue
        if o / n >= frac: out[k] = 1
        elif c / n >= frac: out[k] = 0
    return out


def consistency(per):
    L = [labels(d) for d in per]
    res = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            common = set(L[i]) & set(L[j])
            if not common: res.append({'pair': [NAMES[i], NAMES[j]], 'common': 0}); continue
            agree = sum(L[i][k] == L[j][k] for k in common)
            oi = [k for k in common if L[i][k] == 1]; oj = [k for k in common if L[j][k] == 1]
            inter = len(set(oi) & set(oj)); uni = len(set(oi) | set(oj))
            res.append({'pair': [NAMES[i], NAMES[j]], 'common': len(common), 'agree': agree / len(common),
                        'open_iou': inter / uni if uni else None})
    return res


def best_shift(per, i, j, R=3):
    """integer shift of frame j's map that best aligns its open labels with frame i (registration residual)."""
    Li, Lj = labels(per[i]), labels(per[j])
    best = None
    for sx in range(-R, R + 1):
        for sz in range(-R, R + 1):
            common = [k for k in Li if (k[0] + sx, k[1] + sz) in Lj]
            if len(common) < 50: continue
            oi = {k for k in common if Li[k] == 1}; oj = {k for k in common if Lj[(k[0] + sx, k[1] + sz)] == 1}
            u = len(oi | oj); iou = len(oi & oj) / u if u else 0
            if best is None or iou > best[2]: best = (sx, sz, iou)
    return best


def components(lab):
    opn = {k for k, v in lab.items() if v == 1}
    seen, comps = set(), []
    for k in opn:
        if k in seen: continue
        st, cc = [k], []
        seen.add(k)
        while st:
            x, z = st.pop(); cc.append((x, z))
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (x + d[0], z + d[1])
                if q in opn and q not in seen: seen.add(q); st.append(q)
        comps.append(cc)
    return sorted(comps, key=len, reverse=True)


def main():
    per = column_votes()
    pooled = {}
    for d in per:
        for k, v in d.items():
            p = pooled.setdefault(k, [0, 0]); p[0] += v[0]; p[1] += v[1]
    lab = labels(pooled, minpix=24)
    comps = components(lab)
    cons = consistency(per)
    shifts = {f'{NAMES[0]}->{NAMES[j]}': best_shift(per, 0, j) for j in range(1, 4)}
    shifts.update({f'{NAMES[2]}->{NAMES[3]}': best_shift(per, 2, 3)})
    # sensitivity: pose perturbations (yaw +-1 deg, pitch +-1 deg, h +-0.5) -> how far do hole centroids move?
    base_c = [(np.mean([p[0] for p in c]), np.mean([p[1] for p in c]), len(c)) for c in comps if len(c) >= 4]
    sens = []
    for dyaw, dpitch, dh in [(math.radians(1), 0, 0), (-math.radians(1), 0, 0), (0, math.radians(1), 0), (0, -math.radians(1), 0), (0, 0, .5), (0, 0, -.5), (0, 0, 1.0), (0, 0, -1.0)]:
        pp = column_votes({n: {'dyaw': dyaw, 'dpitch': dpitch, 'dh': dh} for n in NAMES})
        pl = {}
        for d in pp:
            for k, v in d.items():
                q = pl.setdefault(k, [0, 0]); q[0] += v[0]; q[1] += v[1]
        cc = [(np.mean([p[0] for p in c]), np.mean([p[1] for p in c]), len(c)) for c in components(labels(pl, minpix=24)) if len(c) >= 4]
        moves = []
        for bx, bz, n in base_c:
            if not cc: break
            m = min(cc, key=lambda q: (q[0] - bx) ** 2 + (q[1] - bz) ** 2)
            moves.append(math.hypot(m[0] - bx, m[1] - bz))
        sens.append({'dyaw_deg': round(math.degrees(dyaw), 2), 'dpitch_deg': round(math.degrees(dpitch), 2), 'dh': dh,
                     'n_comp': len(cc), 'median_centroid_move': float(np.median(moves)) if moves else None,
                     'max_centroid_move': float(np.max(moves)) if moves else None})
    out = {'phase': PHASE, 'dmax': DMAX,
           'n_labelled_columns': len(lab), 'n_open': sum(lab.values()), 'n_closed': len(lab) - sum(lab.values()),
           'components_ge4': [{'size': len(c), 'centroid_rel': [round(float(np.mean([p[0] for p in c])), 2), round(float(np.mean([p[1] for p in c])), 2)]} for c in comps if len(c) >= 4],
           'component_size_hist': {str(s): sum(1 for c in comps if len(c) == s) for s in range(1, 4)},
           'n_components_total': len(comps),
           'frame_pair_consistency': cons, 'best_integer_shift_between_frames': shifts,
           'pose_sensitivity': sens,
           'labels': [[k[0], k[1], v] for k, v in sorted(lab.items())]}
    (HERE / 'footage-obs.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'labels'}, indent=1))
    # picture
    xs = [k[0] for k in lab]; zs = [k[1] for k in lab]
    x0, z0 = min(xs), min(zs)
    img = np.full((max(zs) - z0 + 1, max(xs) - x0 + 1, 3), 255, np.uint8)
    for (x, z), v in lab.items(): img[z - z0, x - x0] = (60, 60, 60) if v else (90, 200, 90)
    img[-z0, -x0] = (255, 0, 0)
    Image.fromarray(img).resize((img.shape[1] * 6, img.shape[0] * 6), Image.NEAREST).save(HERE / 'footage-obs-map.png')


if __name__ == '__main__':
    main()
