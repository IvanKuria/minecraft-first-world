"""PC planted_render.py-style fixtures, with everything saved.  Same map-making pipeline as pc-c5/planted_render.py:
literal seed L (G0 = (L ^ 0x5DEECE66D) mod 2^48), the PC's tgen spec at the variant's T (lgen = tgen bit-exact), two
plain spawns >= 60 apart in [40,216]^2 (landing column top == world mode), the 4 second-reset and 4 first-reset views
rendered with terrace/synth.render (trender, authentic textures, 'top' grass rule, blur + noise), pixels classified
exactly as c5/obs_footage.classes, projected with obs_footage.project, pooled, labels(minpix=24, frac=.6), components
>= 4 on BOTH maps required (as the PC).  In addition the SAME rendered frames go through terrace/sweep.py
(frame_scores/combine), so each fixture also carries the footage-style sweep scores.  True column depths are stored for
calibration only (never given to the scorer).
usage: pc_fixtures.py <variant-key> <n> <stream-tag> [half=500000]"""
import json, math, os, random, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
sys.path.insert(0, str(PROOF / 'terrace')); sys.path.insert(0, str(PROOF / 'c5'))
import synth as SY, sweep as SW, obs_footage as OF
import lworld as Lw
MULT = 0x5DEECE66D; MASK = (1 << 48) - 1
VARS = {  # PC planted_full.VARS tgen bases (+ the PC's T)
    'Vfull-K4096-nt8-scan3.5': ('ntypes=8,nb=scan,K=4096,margin=0', 3.5),
    'Vfull-K1024-nt8-scan5': ('ntypes=8,nb=scan,K=1024,margin=0', 5.0),
    'Vfull-K1024-nt8-scan4.5': ('ntypes=8,nb=scan,K=1024,margin=0', 4.5),
    'Vfull-K4096-nt8-rand16': ('ntypes=8,nb=rand,K=4096,margin=0', 16.0),
}
def tgen_spec(base, T):
    return f'init=unset,rng=int,pick=xyz,rule=mono,unsetfill=0,cap=fixed,capy=63,shell=1,{base},sweeps={T}'
def view(world, mode, px, pz, frames, tmp):
    phase = (px - math.floor(px), pz - math.floor(pz))
    pooled, per = {}, []
    for name, path, yaw, pitch, h in frames:
        py = mode + 1 + h - .3 * math.sin(pitch)
        im = SY.render(str(world), px, py, pz, yaw, pitch, 'top', mode, tmp)
        a = np.asarray(im).astype(float); R, G, B = a[..., 0], a[..., 1], a[..., 2]
        sky = (B > 200) & (B - R > 60); green = (~sky) & (G > R + 15) & (G > B + 15); openev = (~sky) & (~green)
        cx, cz, ok = OF.project(yaw, pitch, h, phase=phase)
        for lab, m in (('closed', green), ('open', openev)):
            sel = ok & m
            keys, cnt = np.unique(np.stack([cx[sel], cz[sel]], 1), axis=0, return_counts=True)
            for (x, z), n in zip(keys.tolist(), cnt.tolist()):
                q = pooled.setdefault((x, z), [0, 0]); q[0 if lab == 'closed' else 1] += n
        per.append(SW.frame_scores(im, yaw, pitch, h, phase))
    lab = OF.labels(pooled, minpix=24)
    res = SW.combine(per)
    return lab, res, phase
def comps(lab):
    opn = {k for k, v in lab.items() if v == 1}; seen, out = set(), []
    for k in sorted(opn):
        if k in seen: continue
        st, cc = [k], []; seen.add(k)
        while st:
            x, z = st.pop(); cc.append((x, z))
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (x + dx, z + dz)
                if q in opn and q not in seen: seen.add(q); st.append(q)
        out.append(cc)
    return out
def obs_lines(lab):
    """exact make_inputs.obs / planted_render.write_obs rule -> c5search O/C lines"""
    opn = {k for k, v in lab.items() if v == 1}
    big = [c for c in comps(lab) if len(c) >= 4]; cid = {p: i for i, c in enumerate(big) for p in c}
    L = []
    for (x, z), v in sorted(lab.items()):
        if v == 1:
            if (x, z) in cid: L.append(f'O {x} {z} 1 {cid[(x, z)]}')
        elif not any((x + dx, z + dz) in opn for dx in (-1, 0, 1) for dz in (-1, 0, 1)): L.append(f'O {x} {z} 0 -1')
    C = [f'C {sum(p[0] + .5 for p in c) / len(c):.3f} {sum(p[1] + .5 for p in c) / len(c):.3f} {len(c)}' for c in big]
    return L, C, len(big)
def main():
    key, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]; half = int(sys.argv[4]) if len(sys.argv) > 4 else 500000
    base, T = VARS[key]
    rng = random.Random(f'likelihood|{key}|{tag}')
    od = HERE / 'fixtures' / key / tag; od.mkdir(parents=True, exist_ok=True)
    r1set, r2set = SW.r1_poses()[1], SW.r2_poses()[1]
    tmp = od / f'r_{os.getpid()}.ppm'; w = od / f'w_{os.getpid()}.blocks'
    done = len(list(od.glob('fx*.json'))); t = 0
    while done < n and t < n * 6:
        t += 1
        L = rng.randrange(1 << 40) + half; G0 = (L ^ MULT) & MASK
        subprocess.run([str(Lw.LGEN), 'run', tgen_spec(base, T), str(G0), str(w)], capture_output=True, check=True)
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
            lab2, res2, ph2 = view(w, mode, *p2, r2set, tmp); O2, C2, n2 = obs_lines(lab2)
            if n2 < 4: continue
            lab1, res1, ph1 = view(w, mode, *p1, r1set, tmp); O1, C1, n1 = obs_lines(lab1)
            if n1 >= 4: got = True; break
        if not got: continue
        rec = {'key': key, 'T': T, 'L': L, 'G0': G0, 'mode': mode, 'half': half, 'maps': {}}
        for which, p, lab, res, ph, Ol, Cl in (('r2', p2, lab2, res2, ph2, O2, C2), ('r1', p1, lab1, res1, ph1, O1, C1)):
            fx, fz = math.floor(p[0]), math.floor(p[1])
            cols = set(lab) | set(res)
            truth = {}
            for (x, z) in cols:
                X, Z = fx + x, fz + z
                truth[f'{x},{z}'] = int(min(mode - top[Z, X], 8)) if 0 <= X < 256 and 0 <= Z < 256 else 8
            rec['maps'][which] = {'pos': list(p), 'phase': list(ph), 'labels': [[x, z, v] for (x, z), v in sorted(lab.items())],
                                  'scores': [[x, z, {str(k): [round(r, 4), int(nn)] for k, (r, nn) in d.items()}] for (x, z), d in sorted(res.items())],
                                  'O': Ol, 'C': Cl, 'truth': truth}
        (od / f'fx{done:03d}.json').write_text(json.dumps(rec) + '\n')
        print(json.dumps({'fixture': done, 'L': L, 'mode': mode, 'ncomp': [len(rec['maps'][m]['C']) for m in ('r2', 'r1')]}), flush=True)
        done += 1
    for f in (tmp, w): f.unlink(missing_ok=True)
if __name__ == '__main__':
    main()
