"""Planted-seed check of C5 (two maps) under the FULL-HEIGHT shell model.
Per trial: random literal seed L (generator = new Random(L), G0 = (L ^ 0x5DEECE66D) mod 2^48), grow with tgen at the
variant's fitted T, pick two plain spawns >= 60 apart (landing column top == world mode) whose synthetic maps each have
>= MINCOMP open components (as both real maps do), observe them footage-style with fullobs (plane 64 or 63 for E62 = mode+1,
label rule), then run c5gpu5 over L +- HALF in literal mode with ranking. Reports for the planted seed: its per-map hits /
IoU, whether it passes each threshold point, and its rank among all 2*HALF+1 seeds (top-N list).
usage: planted_full.py <variant-key> <labrule> <trials> [half]"""
import json, random, subprocess, sys, math, os
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
B = HERE / 'build'; FX = HERE / 'planted'; FX.mkdir(exist_ok=True)
MULT = 0x5DEECE66D; MASK = (1 << 48) - 1
VARS = {  # key: (kernel X spec, tgen spec without sweeps, T)
    'Vfull-K4096-nt8-scan3.5': ('X:K=4096,nt=8,m=0,yb=64,tf=0,capy=63,surf=63', 'ntypes=8,nb=scan,K=4096,margin=0', 3.5),
    'E62-K4096-nt6-scan3.5': ('X:K=4096,nt=6,m=1,yb=62,tf=0,capy=63,surf=62', 'ntypes=6,nb=scan,K=4096,margin=1', 3.5),
    'E63-K4096-nt8-scan3.5': ('X:K=4096,nt=8,m=1,yb=63,tf=0,capy=63,surf=63', 'ntypes=8,nb=scan,K=4096,margin=1,sy=63,sy0=1,py=63,py0=1', 3.5),
    'Vfull-K4096-nt8-rand16': ('X:K=4096,nt=8,m=0,yb=64,tf=0,capy=63,surf=63', 'ntypes=8,nb=rand,K=4096,margin=0', 16.0),
    'Vfull-K1024-nt8-scan5': ('X:K=1024,nt=8,m=0,yb=64,tf=0,capy=63,surf=63', 'ntypes=8,nb=scan,K=1024,margin=0', 5.0),
}
POINTS = [((3, 3), (0.5, 0.4)), ((3, 3), (0.3, 0.3)), ((2, 2), (0.3, 0.3)), ((4, 4), (0.5, 0.5))]
MINCOMP = 4


def tgen_spec(base, T):
    return f'init=unset,rng=int,pick=xyz,rule=mono,unsetfill=0,cap=fixed,capy=63,shell=1,{base},sweeps={T}'


def main():
    key, lab, trials = sys.argv[1], sys.argv[2], int(sys.argv[3]); half = int(sys.argv[4]) if len(sys.argv) > 4 else 500000
    xs, base, T = VARS[key]
    rng = random.Random(f'{key}|{lab}')
    out = []; t = 0
    while len(out) < trials and t < trials * 6:
        t += 1
        L = rng.randrange(1 << 40) + half; G0 = (L ^ MULT) & MASK
        w = FX / 'w.blocks'
        subprocess.run([str(B / 'tgen.exe'), 'run', tgen_spec(base, T), str(G0), str(w)], capture_output=True, check=True)
        a = np.fromfile(w, np.uint8).reshape(64, 256, 256)
        sol = a != 0
        top = np.where(sol.any(0), 63 - np.argmax(sol[::-1], 0), -1)
        vals, cnt = np.unique(top[8:-8, 8:-8], return_counts=True); mode = int(vals[np.argmax(cnt)])
        def plain():
            for _ in range(5000):
                x, z = rng.uniform(40, 216), rng.uniform(40, 216)
                xs_ = range(int(math.floor(x - .3)), int(math.floor(x + .3)) + 1); zs_ = range(int(math.floor(z - .3)), int(math.floor(z + .3)) + 1)
                if max(top[zz, xx] for xx in xs_ for zz in zs_) == mode: return x, z
            return None
        got = None
        for _ in range(40):
            p2 = plain(); p1 = plain()
            if not p2 or not p1 or math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < 60: continue
            o = json.loads(subprocess.run([str(B / 'fullobs.exe'), str(w), str(mode + 1), lab, *map(repr, (*p2, *p1)), str(FX / 'w2.obs'), str(FX / 'w1.obs')],
                                          capture_output=True, text=True, check=True).stdout)
            if o['ncomp2'] >= MINCOMP and o['ncomp1'] >= MINCOMP: got = (p2, p1, o); break
        if not got: continue
        (p2, p1, o) = got
        cf = FX / 'w.cands'; cf.write_text(f'0 0 1 0 {p2[0]!r} {p2[1]!r} {p1[0]!r} {p1[1]!r}\n')
        maps = f"{FX / 'w2.obs'},{FX / 'w1.obs'}"
        env = {**os.environ, 'C5_SEEDMODE': f'lit:{L - half}:{L + half}', 'C5_TOPN': '2000', 'C5_RANK_CMIN': '1,1'}
        r = subprocess.run([str(B / 'c5gpu5.exe'), xs, str(cf), maps, '1', '10', '40', '1', '3,3', '0.5,0.4', '1'], capture_output=True, text=True, env=env, check=True).stdout.splitlines()
        rank = [l.split() for l in r if l.startswith('RANK')]
        me = [i for i, x in enumerate(rank) if int(x[3]) == L]
        # the planted seed's own values (window of one seed, everything passes)
        env1 = {**os.environ, 'C5_SEEDMODE': f'lit:{L}:{L}'}
        one = [l.split() for l in subprocess.run([str(B / 'c5gpu5.exe'), xs, str(cf), maps, '1', '10', '40', '1', '0,0', '0,0', '1'], capture_output=True, text=True, env=env1, check=True).stdout.splitlines() if l.startswith('PASS')]
        h2, i2, h1, i1 = int(one[0][5]), float(one[0][6]), int(one[0][7]), float(one[0][8])
        passes = {f'c{c[0]},{c[1]}|i{i[0]},{i[1]}': (h2 >= c[0] and i2 >= i[0] and h1 >= c[1] and i1 >= i[1]) for c, i in POINTS}
        rec = {'trial': len(out), 'variant': key, 'lab': lab, 'L': L, 'mode': mode, 'obs': o, 'planted': {'hits2': h2, 'iou2': i2, 'hits1': h1, 'iou1': i1},
               'passes': passes, 'rank_in_window': (me[0] + 1) if me else None, 'window': 2 * half + 1}
        out.append(rec); print(json.dumps(rec), flush=True)
    summ = {'variant': key, 'lab': lab, 'trials': len(out), 'pass_rate': {k: float(np.mean([r['passes'][k] for r in out])) for k in out[0]['passes']} if out else {},
            'in_top2000_of_window': float(np.mean([r['rank_in_window'] is not None for r in out])) if out else None,
            'median_rank': float(np.median([r['rank_in_window'] or 1e9 for r in out])) if out else None}
    print(json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
