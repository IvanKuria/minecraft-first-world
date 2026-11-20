"""Planted-seed check of two-map C5 under the FULL-HEIGHT shell model, with footage-faithful synthetic maps:
render the 4 second-reset and 4 first-reset views with trender (authentic textures, 'top' grass rule, blur+noise as
terrace/synth.py), classify pixels exactly as c5/obs_footage.py (SKY / GREEN / OPEN-EVIDENCE), project every non-sky
pixel onto the grass-top plane with obs_footage.project (camera h above it, player phase = fractional spawn position),
pool votes, labels(minpix=24, frac=.6), components >= 4, rim drop -> c5search obs files (make_inputs.obs rule).
Then rank the planted seed among L +- HALF literal seeds with c5gpu5 (top-N) and record its per-map hits / IoU.
usage: planted_render.py <variant-key> <trials> [half]"""
import json, math, os, random, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
sys.path.insert(0, str(HERE / 'full')); sys.path.insert(0, str(PROOF / 'c5'))
import synth_pc                      # noqa: F401  (points terrace/synth.render at trender.exe)
import synth as SY                   # terrace/synth.py (render with blur + noise)
import sweep as SW                   # terrace/sweep.py (r1/r2 pose sets as used by synth)
import obs_footage as OF             # c5/obs_footage.py (classes, project, labels)
from planted_full import VARS, tgen_spec, MULT, MASK, POINTS
B = HERE / 'build'; FX = HERE / 'planted'; FX.mkdir(exist_ok=True)
MINCOMP = 4


def view_votes(world, mode, px, pz, frames):
    phase = (px - math.floor(px), pz - math.floor(pz))
    pooled = {}
    tmp = FX / f'r_{os.getpid()}.ppm'
    for name, path, yaw, pitch, h in frames:
        py = mode + 1 + h - .3 * math.sin(pitch)
        im = SY.render(str(world), px, py, pz, yaw, pitch, 'top', mode, tmp)
        p = FX / f'r_{os.getpid()}.png'; im.save(p)
        sky, green, openev = OF.classes(p)
        cx, cz, ok = OF.project(yaw, pitch, h, phase=phase)
        for lab, m in (('closed', green), ('open', openev)):
            sel = ok & m
            keys, cnt = np.unique(np.stack([cx[sel], cz[sel]], 1), axis=0, return_counts=True)
            for (x, z), n in zip(keys.tolist(), cnt.tolist()):
                q = pooled.setdefault((x, z), [0, 0]); q[0 if lab == 'closed' else 1] += n
    return OF.labels(pooled, minpix=24)


def write_obs(lab, path, ref):
    opn = {k for k, v in lab.items() if v == 1}
    comps, seen = [], set()
    for k in sorted(opn):
        if k in seen: continue
        st, cc = [k], []; seen.add(k)
        while st:
            x, z = st.pop(); cc.append((x, z))
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (x + dx, z + dz)
                if q in opn and q not in seen: seen.add(q); st.append(q)
        comps.append(cc)
    big = [c for c in comps if len(c) >= 4]
    cid = {p: i for i, c in enumerate(big) for p in c}
    lines = [f'# synthetic full-height map (planted_render.py), rendered + obs_footage classes', f'R {ref}']
    no = nc = 0
    for (x, z), v in sorted(lab.items()):
        if v == 1:
            if (x, z) in cid: lines.append(f'O {x} {z} 1 {cid[(x, z)]}'); no += 1
        else:
            if any((x + dx, z + dz) in opn for dx in (-1, 0, 1) for dz in (-1, 0, 1)): continue
            lines.append(f'O {x} {z} 0 -1'); nc += 1
    for c in big: lines.append(f'C {sum(p[0] + .5 for p in c) / len(c):.3f} {sum(p[1] + .5 for p in c) / len(c):.3f} {len(c)}')
    Path(path).write_text('\n'.join(lines) + '\n')
    return {'open': no, 'closed': nc, 'ncomp': len(big)}


def main():
    key, trials = sys.argv[1], int(sys.argv[2]); half = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    xs, base, T = VARS[key]
    rng = random.Random(f'render|{key}')
    r1set, r2set = SW.r1_poses()[1], SW.r2_poses()[1]
    out = []; t = 0
    while len(out) < trials and t < trials * 4:
        t += 1
        L = rng.randrange(1 << 40) + half; G0 = (L ^ MULT) & MASK
        w = FX / f'w_{os.getpid()}.blocks'
        subprocess.run([str(B / 'tgen.exe'), 'run', tgen_spec(base, T), str(G0), str(w)], capture_output=True, check=True)
        a = np.fromfile(w, np.uint8).reshape(64, 256, 256); sol = a != 0
        top = np.where(sol.any(0), 63 - np.argmax(sol[::-1], 0), -1)
        vals, cnt = np.unique(top[8:-8, 8:-8], return_counts=True); mode = int(vals[np.argmax(cnt)])
        def plain():
            for _ in range(5000):
                x, z = rng.uniform(40, 216), rng.uniform(40, 216)
                xs_ = range(int(math.floor(x - .3)), int(math.floor(x + .3)) + 1); zs_ = range(int(math.floor(z - .3)), int(math.floor(z + .3)) + 1)
                if max(top[zz, xx] for xx in xs_ for zz in zs_) == mode: return x, z
        got = None
        for _ in range(6):
            p2, p1 = plain(), plain()
            if not p2 or not p1 or math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < 60: continue
            o2 = write_obs(view_votes(w, mode, *p2, r2set), FX / f'w2_{os.getpid()}.obs', 2)
            o1 = write_obs(view_votes(w, mode, *p1, r1set), FX / f'w1_{os.getpid()}.obs', 1)
            if o2['ncomp'] >= MINCOMP and o1['ncomp'] >= MINCOMP: got = (p2, p1, o2, o1); break
        if not got: continue
        p2, p1, o2, o1 = got
        cf = FX / f'w_{os.getpid()}.cands'; cf.write_text(f'0 0 1 0 {p2[0]!r} {p2[1]!r} {p1[0]!r} {p1[1]!r}\n')
        maps = f"{FX / f'w2_{os.getpid()}.obs'},{FX / f'w1_{os.getpid()}.obs'}"
        env = {**os.environ, 'C5_SEEDMODE': f'lit:{L - half}:{L + half}', 'C5_TOPN': '2000', 'C5_RANK_CMIN': '1,1'}
        r = subprocess.run([str(B / 'c5gpu5.exe'), xs, str(cf), maps, '1', '10', '40', '1', '3,3', '0.5,0.4', '1'], capture_output=True, text=True, env=env, check=True).stdout.splitlines()
        rank = [l.split() for l in r if l.startswith('RANK')]
        me = [i for i, x in enumerate(rank) if int(x[3]) == L]
        env1 = {**os.environ, 'C5_SEEDMODE': f'lit:{L}:{L}'}
        one = [l.split() for l in subprocess.run([str(B / 'c5gpu5.exe'), xs, str(cf), maps, '1', '10', '40', '1', '0,0', '0,0', '1'], capture_output=True, text=True, env=env1, check=True).stdout.splitlines() if l.startswith('PASS')]
        h2, i2, h1, i1 = int(one[0][5]), float(one[0][6]), int(one[0][7]), float(one[0][8])
        rec = {'trial': len(out), 'variant': key, 'L': L, 'mode': mode, 'map2': o2, 'map1': o1,
               'planted': {'hits2': h2, 'iou2': i2, 'hits1': h1, 'iou1': i1},
               'passes': {f'c{c[0]},{c[1]}|i{i[0]},{i[1]}': (h2 >= c[0] and i2 >= i[0] and h1 >= c[1] and i1 >= i[1]) for c, i in POINTS},
               'rank_in_window': (me[0] + 1) if me else None, 'window': 2 * half + 1}
        out.append(rec); print(json.dumps(rec), flush=True)
    if out:
        print(json.dumps({'variant': key, 'trials': len(out), 'pass_rate': {k: float(np.mean([r['passes'][k] for r in out])) for k in out[0]['passes']},
                          'in_top2000': float(np.mean([r['rank_in_window'] is not None for r in out])),
                          'ranks': [r['rank_in_window'] for r in out]}), flush=True)


if __name__ == '__main__':
    main()
