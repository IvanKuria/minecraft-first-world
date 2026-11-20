"""Synthetic ground-truth evaluation of the likelihood scorer vs C5 statistics on full-height worlds.
usage: synth_eval.py <family> <T_sweeps> <ntrials> <nnull> <S> <sigma_reg> <tag> [threads=6] [T_score=T] [conf_worlds=4]
 - confusion tables P(class | true depth) come from lobs observations of CALIBRATION worlds (states disjoint from the
   evaluation worlds), per reset (r1/r2 poses);
 - each trial: a NEW world (random generator state), lgen at fixed T, footage-like spawns (R1 near (197,230), R2 near
   (224,125), jitter +-8, footage phases), lobs observation, registration error: integer shift ~ round(N(0, sigma_reg))
   per map; lscore scores the true state and <nnull> random states at the same (mis)registered positions.
Output: results/<tag>.jsonl (one line per trial)."""
import sys, json, math, subprocess, time, os
from pathlib import Path
import numpy as np
import lworld as Lw, lobsio as O
HERE = Path(__file__).resolve().parent
fam, T, ntr, nnull, S, sreg, tag = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
nth = int(sys.argv[8]) if len(sys.argv) > 8 else 6
Tsc = float(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] != '-' else T
ncw = int(sys.argv[10]) if len(sys.argv) > 10 else 4
import gen_worlds as GW
tpl, nb, K, nt, _ = GW.SETS[fam]
spec = tpl.format(nt=nt, nb=nb, K=K)
margin = 1 if fam.startswith('fh1') else 0; ytop = 62 if margin else 63
cal = json.loads((HERE / f'calib2-{fam}.json').read_text())['params']
work = HERE / 'work' / tag; work.mkdir(parents=True, exist_ok=True)
(HERE / 'results').mkdir(exist_ok=True)
prm = work / 'params.txt'
prm.write_text(''.join(f'{k} {cal[k]}\n' for k in ('a', 'b', 'p', 'su', 'sv', 'lam', 'kap')) + f'T {Tsc}\nytop {ytop}\nmargin {margin}\nK {K}\nnt {nt}\ntf 0\n')
rng = np.random.default_rng(abs(hash(tag)) % (2**32))
def splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & (2**64 - 1); z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1); z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
    return z ^ (z >> 31)
def make_world(state, path):
    subprocess.run([str(Lw.LGEN), 'run', f'{spec},M={int(T * Lw.V)}', str(state), str(path)], check=True, capture_output=True)
def spawns(r):
    j = r.integers(-8, 9, 4)
    return ((197 + j[0] + O.PH['r1'][0], 230 + j[1] + O.PH['r1'][1]), (224 + j[2] + O.PH['r2'][0], 125 + j[3] + O.PH['r2'][1]))
def observe(wpath, sp, name):
    spf = work / f'{name}.spawns'; outf = work / f'{name}.obs.txt'
    with open(spf, 'w') as f:
        f.write(f'0 r1 {sp[0][0]:.6f} {sp[0][1]:.6f}\n0 r2 {sp[1][0]:.6f} {sp[1][1]:.6f}\n')
    subprocess.run([str(HERE / 'lobs'), str(wpath), str(ytop), str(spf), str(outf), '0.8', '1'], check=True)
    return O.parse_lobs(outf)
# ---- confusion tables from calibration worlds (states with a fixed separate stream)
confp = work / 'conf.json'
if not confp.exists():
    cnt = {w: np.zeros((O.NCLS, O.LD + 1)) for w in ('r1', 'r2')}
    cr = np.random.default_rng(12345)
    for i in range(ncw):
        st = splitmix(0xC0FFEE + i) & ((1 << 48) - 1)
        wp = work / 'calw.blocks'; make_world(st, wp)
        for k in range(6):
            for rec in observe(wp, spawns(cr), 'cal'):
                ph = (rec['px'] - math.floor(rec['px']), rec['pz'] - math.floor(rec['pz']))
                cls = O.classify(rec['S'], rec['U'], rec['M'], ph)
                for c, v in cls.items():
                    D = rec['T'].get(c, O.LD); cnt[rec['which']][v, D] += 1
    conf, lnull, p0s = {}, {}, {}
    for w in cnt:
        c = cnt[w] + 0.5                        # Laplace smoothing
        pcd = c / c.sum(0, keepdims=True)       # P(class | D)
        p0 = cnt[w].sum(0) / cnt[w].sum()       # marginal D over observed columns
        conf[w] = pcd.tolist(); p0s[w] = p0.tolist(); lnull[w] = np.log(pcd @ p0).tolist()
    confp.write_text(json.dumps({'conf': conf, 'lnull': lnull, 'p0': p0s, 'counts': {w: cnt[w].tolist() for w in cnt}}))
CF = json.loads(confp.read_text())
out = open(HERE / 'results' / f'{tag}.jsonl', 'a')
for tr in range(ntr):
    t0 = time.time()
    st = splitmix(0xABCDEF + 7919 * tr + int(T * 1000) + K) & ((1 << 48) - 1)
    wp = work / 'evw.blocks'; make_world(st, wp)
    sp = spawns(rng)
    recs = observe(wp, sp, 'ev')
    maps = []
    for rec in recs:
        w = rec['which']; ph = (rec['px'] - math.floor(rec['px']), rec['pz'] - math.floor(rec['pz']))
        cls = O.classify(rec['S'], rec['U'], rec['M'], ph)
        mp = work / f'ev-{w}.map'
        O.write_map(mp, 1 if w == 'r1' else 2, cls, CF['conf'][w], CF['lnull'][w], rec['O'], rec['C'])
        maps.append(str(mp))
    sh = [int(round(x)) for x in rng.normal(0, sreg, 4)] if sreg > 0 else [0, 0, 0, 0]
    pos = [sp[0][0] + sh[0], sp[0][1] + sh[1], sp[1][0] + sh[2], sp[1][1] + sh[3]]
    r = subprocess.run([str(HERE / 'lscore'), 'synth', str(prm), ','.join(maps), str(st), *[f'{p:.6f}' for p in pos], str(nnull), str(1000 + tr), str(S), '1.0', str(nth)],
                       capture_output=True, text=True, check=True, env={**os.environ, 'LS_DUMP': str(work / f'null-{tr}.txt')})
    res = json.loads(r.stdout)
    res.update({'fam': fam, 'T': T, 'T_score': Tsc, 'trial': tr, 'state': st, 'spawns': sp, 'shift': sh, 'S': S, 'sigma_reg': sreg,
                'nobs': [sum(1 for l in open(m) if l.startswith('Q')) for m in maps],
                'ncomp': [len(rec['C']) for rec in recs], 'nsweep': [len(rec['S']) for rec in recs], 'sec': round(time.time() - t0, 1)})
    # null tail: keep top values for empirical rank checks
    nv = np.loadtxt(work / f'null-{tr}.txt', ndmin=2)
    res['null_q'] = [float(x) for x in np.quantile(nv[:, 0], [.5, .9, .99, .999])]
    res['c5_true_pass'] = {str(tau): all(res['true_c5'][m][0] >= math.ceil(2 * res['true_c5'][m][2] / 3) and res['true_c5'][m][1] >= tau for m in range(len(maps))) for tau in (.5, .4, .3)}
    out.write(json.dumps(res) + '\n'); out.flush()
    print(json.dumps({k: res[k] for k in ('trial', 'true_llr', 'null_max', 'null_mean', 'null_sd', 'z', 'null_ge_true', 'true_c5', 'c5_true_pass', 'sec')}), flush=True)
