"""Full-height shell model (lead's correction, Oct 15 evening): fit the fixed-T windows against the footage pit statistics.
World model (tgen): seeds and picks over the full height, no discard above 42 (cap=fixed capy=63 = whole world), shell = side
walls (x,z = 0/255, all y) + floor (y=0) rock from the start, TOP OPEN, interior unset, GEN C4 rule (scan | random
neighbour), fixed T, leftover unset -> AIR, grass = column top (synth 'top' rule). Seed forms:
  Vfull : x=nextInt(256), y=nextInt(64), z=nextInt(256)                    (margin=0)
  E62   : nextInt(254)+1, nextInt(62)+1, nextInt(254)+1                    (margin=1)
  E63   : nextInt(254)+1, nextInt(63)+1, nextInt(254)+1                    (margin=1, sy=py=63, sy0=py0=1)
Statistic = terrace/compare.py + pertail.py, unchanged (synthetic observation through terrace/synth.py via synth_pc.py):
near-field (d<10) label fractions 0/-1/-2/-3/<=-4 per reset view set vs footage, and P(r1-like & r2-like) for the
shallow-depression fraction (footage r1 .36, r2 .22).
usage: fit_full.py grid|score [nspawn] [nstates]
"""
import json, os, subprocess, sys, glob
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TERR = HERE.parents[1] / 'terrace'
sys.path.insert(0, str(TERR))
TGEN = HERE.parent / 'build' / 'tgen.exe'
FORMS = {'Vfull': 'margin=0', 'E62': 'margin=1', 'E63': 'margin=1,sy=63,sy0=1,py=63,py0=1'}
TGRID = {('scan', 256): [5, 6, 7, 8, 9, 10], ('scan', 1024): [3.5, 4, 4.5, 5, 5.5, 6, 7], ('scan', 4096): [2.5, 3, 3.5, 4, 4.5, 5],
         ('rand', 256): [20, 25, 30, 35, 40, 50], ('rand', 1024): [12, 15, 18, 21, 25, 30], ('rand', 4096): [8, 10, 12, 14, 16, 20]}
NTS = (4, 6, 8, 12)


def spec(form, rule, K, nt, Tmax, snaps):
    return (f'init=unset,ntypes={nt},rng=int,pick=xyz,nb={rule},rule=mono,K={K},unsetfill=0,cap=fixed,capy=63,shell=1,'
            f'{FORMS[form]},sweeps={Tmax},snap=' + '/'.join(f'{t}v' for t in snaps))


def tag(form, rule, K, nt, s):
    return f'{form}_{rule}_K{K}_nt{nt}_s{s}'


def gen_one(a):
    form, rule, K, nt, s = a
    Ts = TGRID[(rule, K)]
    tg = tag(form, rule, K, nt, s)
    if all((HERE / 'synth' / f'{tg}.T{t}.json').exists() for t in Ts): return []
    env = {**os.environ, 'TGEN_SNAP': str(HERE / 'worlds' / tg)}
    subprocess.run([str(TGEN), 'run', spec(form, rule, K, nt, max(Ts), Ts), str(0x5EED00 + 7919 * s + K), '-'], env=env,
                   capture_output=True, check=True)
    V = 256 * 256 * 64
    return [(tg, t, HERE / 'worlds' / f'{tg}.{int(t * V + .5)}.blocks') for t in Ts]


def synth_one(a):
    tg, t, world, nsp = a
    out = HERE / 'synth' / f'{tg}.T{t}.json'
    if not out.exists():
        subprocess.run([sys.executable, str(HERE / 'synth_pc.py'), str(world), str(nsp), '7', str(out), 'top'],
                       capture_output=True, check=True, cwd=str(TERR))
    try: os.remove(world)
    except OSError: pass
    return str(out)


def grid(nsp, nstates):
    (HERE / 'worlds').mkdir(exist_ok=True); (HERE / 'synth').mkdir(exist_ok=True)
    cfg = [(f, r, K, nt, s) for s in range(nstates) for f in FORMS for (r, K) in TGRID for nt in NTS]
    with Pool(28) as p:
        worlds = [w for ws in p.imap_unordered(gen_one, cfg) for w in ws]
        print(json.dumps({'worlds': len(worlds)}), flush=True)
        for i, o in enumerate(p.imap_unordered(synth_one, [(tg, t, w, nsp) for tg, t, w in worlds])):
            if i % 50 == 0: print(i, o, flush=True)


def score():
    import compare as CP
    F = CP.footage('0.35')
    fb = {w: F[w]['bins'] for w in ('r1', 'r2')}
    fs = {w: fb[w]['-1'] + fb[w]['-2'] + fb[w]['-3'] for w in ('r1', 'r2')}
    rows = []
    for p in sorted(glob.glob(str(HERE / 'synth' / '*.json'))):
        d, agg, tru, per = CP.synth(p, '0.35')
        name = Path(p).stem                       # form_rule_Kk_ntn_ss.Tt
        base, T = name.rsplit('.T', 1)
        form, rule, K, nt, s = base.split('_')
        b = {w: CP.bins(agg[w]) for w in ('r1', 'r2')}
        if not b['r1'] or not b['r2']: continue
        l1 = sum(abs(b[w][k] - fb[w][k]) for w in ('r1', 'r2') for k in ('0', '-1', '-2', '-3', '<=-4'))
        vals = {w: [(x['-1'] + x['-2'] + x['-3']) if x else 0.0 for x in per[w]] for w in ('r1', 'r2')}
        p1 = np.mean([v >= fs['r1'] for v in vals['r1']]); p2 = np.mean([v >= fs['r2'] for v in vals['r2']])
        rows.append({'form': form, 'rule': rule, 'K': int(K[1:]), 'nt': int(nt[2:]), 'state': s, 'T': float(T),
                     'L1_bins_vs_footage': round(l1, 3), 'P_r1_and_r2_like': round(float(p1 * p2), 3),
                     'r1': {k: round(b['r1'][k], 3) for k in ('0', '-1', '-2', '-3', '<=-4')},
                     'r2': {k: round(b['r2'][k], 3) for k in ('0', '-1', '-2', '-3', '<=-4')},
                     'mode': d['mode'], 'truth_surface': {k: round(v, 3) for k, v in d['surface'].items() if isinstance(v, float)}})
    (HERE / 'fit-full.json').write_text(json.dumps({'footage': fb, 'footage_shallow': fs, 'rows': rows}, indent=1))
    # best T per (form, rule, K, nt): pooled over states, max P then min L1
    best = {}
    for r in rows:
        k = (r['form'], r['rule'], r['K'], r['nt'], r['T'])
        best.setdefault(k, []).append(r)
    agg = []
    for k, rs in best.items():
        agg.append((k, float(np.mean([r['P_r1_and_r2_like'] for r in rs])), float(np.mean([r['L1_bins_vs_footage'] for r in rs])), len(rs)))
    agg.sort(key=lambda a: (-a[1], a[2]))
    print('footage r1', fb['r1'], '\nfootage r2', fb['r2'])
    print('%-6s %-4s %5s %3s %5s  %6s %6s %s' % ('form', 'rule', 'K', 'nt', 'T/V', 'P', 'L1', 'n'))
    for (form, rule, K, nt, T), P, L1, n in agg[:40]:
        print('%-6s %-4s %5d %3d %5.1f  %6.3f %6.3f %d' % (form, rule, K, nt, T, P, L1, n))


if __name__ == '__main__':
    if sys.argv[1] == 'grid': grid(int(sys.argv[2]) if len(sys.argv) > 2 else 12, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    else: score()
