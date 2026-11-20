"""Stage-2 prototype (lead's plan, Oct 16): exact full-height regrowth + map likelihood, vs C5's seed-only rank.
Note: an exact tgen regrowth at the fitted T costs ~0.3 s (same order as a 2-view local Monte-Carlo), so Stage 2 here is
EXACT regrowth of each candidate followed by a column likelihood against the two observed (rendered, footage-faithful)
maps: P(label open | regrown column depth d) from a calibration table built on separate planted worlds.
Per fixture: planted seed L + NRAND random seeds from the same literal window; report the planted seed's rank by
log-likelihood (1 = best) and, for comparison, its C5 combined-score rank among the same candidates.
usage: stage2_proto.py <variant-key> <fixtures> [nrand]"""
import json, math, os, random, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import planted_render as PR
from planted_full import VARS, tgen_spec, MULT, MASK
B = HERE / 'build'; FX = HERE / 'planted' / 'fx'; FX.mkdir(parents=True, exist_ok=True)
DMAXD = 8


def depth_map(blocks_path):
    a = np.fromfile(blocks_path, np.uint8).reshape(64, 256, 256); sol = a != 0
    top = np.where(sol.any(0), 63 - np.argmax(sol[::-1], 0), -1)
    vals, cnt = np.unique(top[8:-8, 8:-8], return_counts=True); mode = int(vals[np.argmax(cnt)])
    return np.clip(mode - top, 0, DMAXD), mode, top


def read_obs(path):
    O = []
    for l in Path(path).read_text().splitlines():
        if l.startswith('O '):
            _, x, z, lab, c = l.split(); O.append((int(x), int(z), int(lab)))
    return O


def grow(spec, G0, out):
    subprocess.run([str(B / 'tgen.exe'), 'run', spec, str(G0), str(out)], capture_output=True, check=True)


def make_fixture(args):
    key, i = args
    f = FX / f'{key}_{i}.json'
    if f.exists(): return json.loads(f.read_text())
    xs, base, T = VARS[key]
    rng = random.Random(f'stage2|{key}|{i}')
    for _ in range(30):
        L = rng.randrange(1 << 40) + 10 ** 7; G0 = (L ^ MULT) & MASK
        w = FX / f'{key}_{i}.blocks'; grow(tgen_spec(base, T), G0, w)
        dm, mode, top = depth_map(w)
        def plain():
            for _ in range(5000):
                x, z = rng.uniform(40, 216), rng.uniform(40, 216)
                if max(top[int(math.floor(zz)), int(math.floor(xx))] for xx in (x - .3, x + .3) for zz in (z - .3, z + .3)) == mode: return x, z
        p2, p1 = plain(), plain()
        if not p2 or not p1 or math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < 60: continue
        o2 = PR.write_obs(PR.view_votes(w, mode, *p2, PR.SW.r2_poses()[1]), FX / f'{key}_{i}.w2.obs', 2)
        o1 = PR.write_obs(PR.view_votes(w, mode, *p1, PR.SW.r1_poses()[1]), FX / f'{key}_{i}.w1.obs', 1)
        if o2['ncomp'] >= 4 and o1['ncomp'] >= 4:
            rec = {'key': key, 'i': i, 'L': L, 'p2': p2, 'p1': p1, 'o2': o2, 'o1': o1}
            f.write_text(json.dumps(rec)); return rec
    return None


def calib(fixtures):
    """P(open | depth) from the fixtures' own true worlds (calibration set = first half of fixtures)."""
    cnt = np.zeros((DMAXD + 1, 2))
    for r in fixtures:
        dm, mode, _ = depth_map(FX / f"{r['key']}_{r['i']}.blocks")
        for which, p in (('w2', r['p2']), ('w1', r['p1'])):
            fx, fz = math.floor(p[0]), math.floor(p[1])
            for x, z, lab in read_obs(FX / f"{r['key']}_{r['i']}.{which}.obs"):
                X, Z = x + fx, z + fz
                if 0 <= X < 256 and 0 <= Z < 256: cnt[dm[Z, X], lab] += 1
    p = (cnt[:, 1] + 1) / (cnt.sum(1) + 2)
    return p.tolist()


def loglik(args):
    spec, G0, r, pop = args
    w = FX / f"cand_{os.getpid()}.blocks"; grow(spec, G0, w)
    dm, mode, _ = depth_map(w)
    ll = 0.0; pop = np.array(pop)
    for which, p in (('w2', r['p2']), ('w1', r['p1'])):
        fx, fz = math.floor(p[0]), math.floor(p[1])
        for x, z, lab in read_obs(FX / f"{r['key']}_{r['i']}.{which}.obs"):
            X, Z = x + fx, z + fz
            if 0 <= X < 256 and 0 <= Z < 256:
                q = pop[dm[Z, X]]; ll += math.log(q if lab else 1 - q)
    return ll


def c5_scores(r, cands):
    """C5 combined score (ranked mode) for the given literal seeds: run c5gpu5 on each seed's one-seed window."""
    xs = VARS[r['key']][0]
    cf = FX / f"c_{r['key']}_{r['i']}.cands"; cf.write_text(f"0 0 1 0 {r['p2'][0]!r} {r['p2'][1]!r} {r['p1'][0]!r} {r['p1'][1]!r}\n")
    maps = f"{FX / (r['key'] + '_' + str(r['i']) + '.w2.obs')},{FX / (r['key'] + '_' + str(r['i']) + '.w1.obs')}"
    out = {}
    for L in cands:
        env = {**os.environ, 'C5_SEEDMODE': f'lit:{L}:{L}', 'C5_TOPN': '1', 'C5_RANK_CMIN': '0,0'}
        lines = subprocess.run([str(B / 'c5gpu5.exe'), xs, str(cf), maps, '1', '10', '40', '1', '0,0', '0,0', '1'], capture_output=True, text=True, env=env).stdout.splitlines()
        rk = [l.split() for l in lines if l.startswith('RANK')]
        out[L] = float(rk[0][5]) if rk else -1
    return out


def main():
    key, nfx = sys.argv[1], int(sys.argv[2]); nrand = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    xs, base, T = VARS[key]; spec = tgen_spec(base, T)
    with Pool(min(nfx, 16)) as p:
        fxs = [r for r in p.map(make_fixture, [(key, i) for i in range(nfx)]) if r]
    half = len(fxs) // 2
    pop = calib(fxs[:half])
    print(json.dumps({'calibration_P_open_by_depth': [round(x, 3) for x in pop], 'fixtures': len(fxs)}), flush=True)
    res = []
    with Pool(24) as p:
        for r in fxs[half:]:
            rng = random.Random(r['L'])
            cands = [r['L']] + [r['L'] + rng.randrange(-5_000_000, 5_000_000) for _ in range(nrand)]
            lls = p.map(loglik, [(spec, (L ^ MULT) & MASK, r, pop) for L in cands])
            c5 = c5_scores(r, cands)
            rank_ll = 1 + sum(1 for v in lls[1:] if v > lls[0])
            rank_c5 = 1 + sum(1 for L in cands[1:] if c5[L] > c5[cands[0]])
            rec = {'fixture': r['i'], 'rank_loglik': rank_ll, 'rank_c5': rank_c5, 'n': len(cands), 'll_true': lls[0],
                   'll_best_random': max(lls[1:]), 'll_median_random': float(np.median(lls[1:]))}
            res.append(rec); print(json.dumps(rec), flush=True)
    print(json.dumps({'variant': key, 'n_test': len(res), 'loglik_rank1_frac': float(np.mean([r['rank_loglik'] == 1 for r in res])),
                      'loglik_median_rank': float(np.median([r['rank_loglik'] for r in res])),
                      'c5_median_rank': float(np.median([r['rank_c5'] for r in res]))}), flush=True)


if __name__ == '__main__':
    main()
