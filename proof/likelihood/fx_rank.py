"""Confusion tables from CALIBRATION fixtures, then rank planted literal seeds of TEST fixtures among their L +- half
window with lscore (stage 1 exact rank over the whole window, stage 2 = full scorer on the stage-1 top-N).
usage: fx_rank.py <variant-key> <calib-model-family> <T_score> <topN> <threads> <test-tag> [half] [regshift-sigma=0] [S2=1]
Writes results/fxrank-<key>-<tag>.jsonl"""
import json, math, sys, subprocess, random, os
from pathlib import Path
import numpy as np
import lobsio as O
HERE = Path(__file__).resolve().parent
key, fam, Tsc, topN, nth, tag = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
half = int(sys.argv[7]) if len(sys.argv) > 7 else None
sreg = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
S2 = int(sys.argv[9]) if len(sys.argv) > 9 else 1
FX = HERE / 'fixtures' / key
def classes(m):
    ph = m['phase']; sweep, unl = {}, set()
    for x, z, s in m['scores']:
        if math.hypot(x + .5 - ph[0], z + .5 - ph[1]) >= 10: continue
        it = sorted(((int(k), v) for k, v in s.items()), key=lambda kv: -kv[1][0])
        if it and it[0][1][0] >= .35 and (len(it) < 2 or it[0][1][0] - it[1][1][0] >= .12): sweep[(x, z)] = it[0][0]
        else: unl.add((x, z))
    mraw = {(x, z): v for x, z, v in m['labels']}
    return O.classify(sweep, unl, mraw, ph)
# ---- confusion from calibration fixtures (never the test fixtures)
cp = FX / 'conf.json'
if not cp.exists():
    cnt = {w: np.zeros((O.NCLS, O.LD + 1)) for w in ('r1', 'r2')}
    for f in sorted((FX / 'calib').glob('fx*.json')):
        rec = json.loads(f.read_text())
        for w, m in rec['maps'].items():
            for (x, z), c in classes(m).items(): cnt[w][c, m['truth'].get(f'{x},{z}', 8)] += 1
    conf, lnull, p0s = {}, {}, {}
    for w in cnt:
        c = cnt[w] + 0.5; pcd = c / c.sum(0, keepdims=True); p0 = cnt[w].sum(0) / cnt[w].sum()
        conf[w] = pcd.tolist(); p0s[w] = p0.tolist(); lnull[w] = np.log(pcd @ p0).tolist()
    cp.write_text(json.dumps({'conf': conf, 'lnull': lnull, 'p0': p0s, 'counts': {w: cnt[w].tolist() for w in cnt},
                              'n_fixtures': len(list((FX / 'calib').glob('fx*.json')))}))
CF = json.loads(cp.read_text())
cal = json.loads((HERE / f'calib2-{fam}.json').read_text())['params']
K = int(key.split('-K')[1].split('-')[0]); nt = int(key.split('-nt')[1].split('-')[0])
work = HERE / 'work' / f'fx-{key}-{tag}'; work.mkdir(parents=True, exist_ok=True)
prm = work / 'params.txt'
prm.write_text(''.join(f'{k} {cal[k]}\n' for k in ('a', 'b', 'p', 'su', 'sv', 'lam', 'kap')) + f'T {Tsc}\nytop 63\nmargin 0\nK {K}\nnt {nt}\ntf 0\nrmax1 12\n')
(HERE / 'results').mkdir(exist_ok=True)
out = (HERE / 'results' / f"fxrank-{key}-{tag}{os.environ.get('FX_SUFFIX', '')}.jsonl").open('a')
rng = random.Random(f'shift|{key}|{tag}')
import os
ONLY = set(os.environ['FX_ONLY'].split(',')) if os.environ.get('FX_ONLY') else None
SUFFIX = os.environ.get('FX_SUFFIX', '')
for f in sorted((FX / tag).glob('fx*.json')):
    if ONLY and f.stem not in ONLY: continue
    rec = json.loads(f.read_text())
    maps = []
    for w in ('r1', 'r2'):
        m = rec['maps'][w]; mp = work / f'{f.stem}-{w}.map'
        O.write_map(mp, 1 if w == 'r1' else 2, classes(m), CF['conf'][w], CF['lnull'][w], m['O'], m['C'])
        maps.append(str(mp))
    sh = [int(round(rng.gauss(0, sreg))) for _ in range(4)] if sreg > 0 else [0, 0, 0, 0]
    p1, p2 = rec['maps']['r1']['pos'], rec['maps']['r2']['pos']
    pos = [p1[0] + sh[0], p1[1] + sh[1], p2[0] + sh[2], p2[1] + sh[3]]
    h = half if half is not None else rec['half']
    r = subprocess.run([str(HERE / 'lscore'), 'rank', str(prm), ','.join(maps), str(rec['L']), str(h), *[f'{x!r}' for x in pos], str(topN), str(nth), str(S2)],
                       capture_output=True, text=True, check=True, env={**os.environ, 'LS_TOPDUMP': str(work / f"{f.stem}{os.environ.get('FX_SUFFIX', '')}.top")})
    res = json.loads(r.stdout); res.update({'fixture': f.stem, 'key': key, 'model_family': fam, 'T_score': Tsc, 'shift': sh, 'S2': S2,
                                            'ncomp': [len(rec['maps'][w]['C']) for w in ('r1', 'r2')]})
    out.write(json.dumps(res) + '\n'); out.flush()
    print(json.dumps({k: res[k] for k in ('fixture', 'rank1', 'window_others', 'true_in_topN', 'rank2_in_topN', 's1_true', 's2_true', 's2_best_other', 'shift', 'stage1_sec_per_seed_per_thread')}), flush=True)
