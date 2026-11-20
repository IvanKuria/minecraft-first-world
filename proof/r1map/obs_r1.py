"""R1MAP task 1-2: surface open/closed column map from the FIRST-reset fall, registered to the player.

Same conventions as c5/obs_footage.py (pixel classes, plane crossing, 32-block cut, vote thresholds, holes >= 4 kept,
closed columns 8-adjacent to open dropped), applied to four first-reset frames of the 2011 640x360 channel copy:
  n1662 85.967 s  pose: generator/first-reset-texture/backward-holdout.json (phase fixed, angles+height free; held-out)
  n1663 86.019 s  pose: shared-fixed70.json  (frame '86.00')
  n1666 86.174 s  pose: shared-fixed70.json  (frame '86.15')
  n1668 86.277 s  pose: shared-fixed70.json  (frame '86.30')
Frames were re-extracted with ffmpeg select=eq(n,N) from proof/link/footage/wayback-fakeurl.mp4; the same command
reproduces LINK's r2-153.054.png byte-for-byte (n2959), so indices match frame-pts.txt.
Poses were fitted BEFORE this lane by the authentic-grass method with corrected half-pixel sampling (README.md).
Three near-identical shared-phase modes exist; mode 0 is the main map, modes 1-2 are registration alternatives.

Output: r1map/footage-obs-r1.json (labels relative to floor(player x1), floor(z1)), r1map/footage-r1.obs.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
ROOT = PROOF.parent
sys.path.insert(0, str(PROOF / 'c5'))
import obs_footage as OF  # noqa: E402  (classes, labels, components, F, DMAX)

TEX = ROOT / 'generator/first-reset-texture'
FRAMES = [('n1662', 'backward', None), ('n1663', 'shared', '86.00'), ('n1666', 'shared', '86.15'), ('n1668', 'shared', '86.30')]


def poses(mode=0):
    sh = json.loads((TEX / 'shared-fixed70.json').read_text())[mode]
    bk = json.loads((TEX / 'backward-holdout.json').read_text())[mode]
    assert bk['fixed_player_phase'] == sh['shared_player_phase']
    phase = tuple(sh['shared_player_phase'])
    out = []
    for name, src, key in FRAMES:
        if src == 'backward': p = bk['pose']
        else: p = next(fr['camera_parameters'] for fr in sh['frames'] if fr['frame'] == key)
        out.append((name, p[0], p[1], p[2]))
    return phase, out


def column_votes(mode=0, perturb=None, frames=None):
    phase, P = poses(mode)
    per = []
    for name, yaw, pitch, h in P:
        if frames and name not in frames: continue
        kw = (perturb or {}).get(name, perturb.get('*', {}) if perturb else {})
        cx, cz, ok = OF.project(yaw, pitch, h, phase=phase, **kw)
        sky, green, openev = OF.classes(HERE / 'frames' / f'{name}.png')
        d = {}
        for lab, m in (('closed', green), ('open', openev)):
            sel = ok & m
            keys, cnt = np.unique(np.stack([cx[sel], cz[sel]], 1), axis=0, return_counts=True)
            for (x, z), n in zip(keys.tolist(), cnt.tolist()):
                d.setdefault((x, z), [0, 0])[0 if lab == 'closed' else 1] += n
        per.append((name, d))
    return phase, per


def pool(per):
    pooled = {}
    for _, d in per:
        for k, v in d.items():
            p = pooled.setdefault(k, [0, 0]); p[0] += v[0]; p[1] += v[1]
    return pooled


def consistency(per):
    L = [(n, OF.labels(d)) for n, d in per]
    res = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            (ni, li), (nj, lj) = L[i], L[j]
            common = set(li) & set(lj)
            if not common: res.append({'pair': [ni, nj], 'common': 0}); continue
            agree = sum(li[k] == lj[k] for k in common)
            oi = {k for k in common if li[k] == 1}; oj = {k for k in common if lj[k] == 1}
            u = len(oi | oj)
            res.append({'pair': [ni, nj], 'common': len(common), 'agree': agree / len(common), 'open_iou': len(oi & oj) / u if u else None})
    return res


def best_shift(li, lj, R=3):
    best = None
    for sx in range(-R, R + 1):
        for sz in range(-R, R + 1):
            common = [k for k in li if (k[0] + sx, k[1] + sz) in lj]
            if len(common) < 50: continue
            oi = {k for k in common if li[k] == 1}; oj = {k for k in common if lj[(k[0] + sx, k[1] + sz)] == 1}
            u = len(oi | oj); iou = len(oi & oj) / u if u else 0
            if best is None or iou > best[2]: best = (sx, sz, round(iou, 3))
    return best


def centroids(lab):
    return [(float(np.mean([p[0] for p in c])), float(np.mean([p[1] for p in c])), len(c)) for c in OF.components(lab) if len(c) >= 4]


def centroid_moves(base, cc):
    moves = []
    for bx, bz, n in base:
        if not cc: break
        m = min(cc, key=lambda q: (q[0] - bx) ** 2 + (q[1] - bz) ** 2)
        moves.append(math.hypot(m[0] - bx, m[1] - bz))
    return moves


def to_obs(lab, header):
    opn = {k for k, v in lab.items() if v == 1}
    big = [c for c in OF.components(lab) if len(c) >= 4]
    cid = {p: i for i, c in enumerate(big) for p in c}
    lines = [header, 'R 1']; n_open = n_closed = 0
    for (x, z), v in sorted(lab.items()):
        if v == 1:
            if (x, z) in cid: lines.append(f'O {x} {z} 1 {cid[(x, z)]}'); n_open += 1
        else:
            if any((x + dx, z + dz) in opn for dx in (-1, 0, 1) for dz in (-1, 0, 1)): continue
            lines.append(f'O {x} {z} 0 -1'); n_closed += 1
    for c in big:
        lines.append(f'C {sum(p[0] + .5 for p in c) / len(c):.3f} {sum(p[1] + .5 for p in c) / len(c):.3f} {len(c)}')
    return lines, {'open': n_open, 'closed': n_closed, 'components': [len(c) for c in big]}


def picture(lab, path, mark=(0, 0)):
    xs = [k[0] for k in lab]; zs = [k[1] for k in lab]
    x0, z0 = min(xs), min(zs)
    img = np.full((max(zs) - z0 + 1, max(xs) - x0 + 1, 3), 255, np.uint8)
    for (x, z), v in lab.items(): img[z - z0, x - x0] = (60, 60, 60) if v else (90, 200, 90)
    if 0 <= mark[1] - z0 < img.shape[0] and 0 <= mark[0] - x0 < img.shape[1]: img[mark[1] - z0, mark[0] - x0] = (255, 0, 0)
    Image.fromarray(img).resize((img.shape[1] * 6, img.shape[0] * 6), Image.NEAREST).save(path)


def main():
    phase, per = column_votes(0)
    lab = OF.labels(pool(per), minpix=24)
    comps = OF.components(lab)
    base_c = centroids(lab)
    L = {n: OF.labels(d) for n, d in per}
    names = [n for n, _ in per]
    shifts = {f'{names[i]}->{names[j]}': best_shift(L[names[i]], L[names[j]]) for i in range(4) for j in range(i + 1, 4)}
    # registration alternatives: the other two shared-phase modes (whole-map shift + relabel)
    modes = []
    for m in (1, 2):
        ph, pm = column_votes(m)
        lm = OF.labels(pool(pm), minpix=24)
        common = set(lab) & set(lm)
        modes.append({'mode': m, 'phase': ph, 'agree': sum(lab[k] == lm[k] for k in common) / len(common),
                      'median_centroid_move': float(np.median(centroid_moves(base_c, centroids(lm))))})
    # pose sensitivity, same perturbation set as obs_footage.py, applied to all four frames
    sens = []
    for dyaw, dpitch, dh in [(math.radians(1), 0, 0), (-math.radians(1), 0, 0), (0, math.radians(1), 0), (0, -math.radians(1), 0), (0, 0, .5), (0, 0, -.5), (0, 0, 1.0), (0, 0, -1.0)]:
        _, pp = column_votes(0, {'*': {'dyaw': dyaw, 'dpitch': dpitch, 'dh': dh}})
        cc = centroids(OF.labels(pool(pp), minpix=24))
        mv = centroid_moves(base_c, cc)
        sens.append({'dyaw_deg': round(math.degrees(dyaw), 2), 'dpitch_deg': round(math.degrees(dpitch), 2), 'dh': dh, 'n_comp': len(cc),
                     'median_centroid_move': float(np.median(mv)) if mv else None, 'max_centroid_move': float(np.max(mv)) if mv else None})
    # leave-one-frame-out: does each frame's own map agree with the map built from the other three?
    loo = []
    for n in names:
        rest = OF.labels(pool([q for q in per if q[0] != n]), minpix=18)
        own = L[n]; common = set(rest) & set(own)
        oi = {k for k in common if rest[k] == 1}; oj = {k for k in common if own[k] == 1}; u = len(oi | oj)
        loo.append({'frame': n, 'common': len(common), 'agree': sum(rest[k] == own[k] for k in common) / max(1, len(common)), 'open_iou': len(oi & oj) / u if u else None})
    lines, obs_summary = to_obs(lab, '# first-reset surface map (r1map/obs_r1.py), relative to floor(x1), floor(z1); O dx dz lab comp | C cdx cdz size')
    (HERE / 'footage-r1.obs').write_text('\n'.join(lines) + '\n')
    out = {'phase_mode0': phase, 'dmax': OF.DMAX, 'frames': [list(p) for p in poses(0)[1]],
           'n_labelled_columns': len(lab), 'n_open': sum(lab.values()), 'n_closed': len(lab) - sum(lab.values()),
           'components_ge4': [{'size': n, 'centroid_rel': [round(x, 2), round(z, 2)]} for x, z, n in base_c],
           'component_size_hist': {str(s): sum(1 for c in comps if len(c) == s) for s in range(1, 4)},
           'n_components_total': len(comps), 'obs_file': obs_summary,
           'frame_pair_consistency': consistency(per), 'best_integer_shift_between_frames': shifts,
           'leave_one_frame_out': loo, 'phase_mode_alternatives': modes, 'pose_sensitivity': sens,
           'labels': [[k[0], k[1], v] for k, v in sorted(lab.items())]}
    (HERE / 'footage-obs-r1.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'labels'}, indent=1))
    picture(lab, HERE / 'footage-obs-r1-map.png')
    for n in names: picture(L[n], HERE / f'map-{n}.png')


if __name__ == '__main__':
    main()
