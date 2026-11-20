"""All-rock window fit (lead, Oct 16: Notch's 2009 annotation 'grown from a few hundred seed blocks of rock').
Worlds: tgen full height, ntypes=1 (every seed rock, no type draw), seeds+picks per form (Vfull / E62 / E63), shell on
(walls+floor rock, top open) or off, scan | random neighbour, fixed T, leftover unset -> AIR (caves = unreached cells).
Step 1 (truth): every snapshot's true surface profile (terrace surf.stats). Step 2 (synth): terrace synth.py near-field
statistic (via synth_pc.py) on the shortlisted T per config (truth f0 in the window that matched before), scored with
compare/pertail exactly as fit_full.py. usage: fit_rock.py gen | synth | score"""
import json, os, subprocess, sys, glob, re
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TERR = HERE.parents[1] / 'terrace'; sys.path.insert(0, str(TERR))
TGEN = HERE.parent / 'build' / 'tgen.exe'
FORMS = {'Vfull': 'margin=0', 'E62': 'margin=1', 'E63': 'margin=1,sy=63,sy0=1,py=63,py0=1'}
KS = (128, 200, 256, 300, 400, 500, 512, 600, 800, 1000)
TG = {'scan': [4, 6, 8, 10, 12, 15, 20, 25, 30, 40], 'rand': [15, 25, 35, 50, 70, 100, 130, 160]}
V = 256 * 256 * 64
W = HERE / 'rworlds'; SY = HERE / 'rsynth'


def tag(form, shell, rule, K): return f'{form}_sh{shell}_{rule}_K{K}'


def spec(form, shell, rule, K):
    Ts = TG[rule]
    return (f'init=unset,ntypes=1,rng=int,pick=xyz,nb={rule},rule=mono,K={K},unsetfill=0,cap=fixed,capy=63,shell={shell},'
            f'{FORMS[form]},sweeps={max(Ts)},snap=' + '/'.join(f'{t}v' for t in Ts))


def gen_one(c):
    form, shell, rule, K = c
    tg = tag(*c); out = []
    files = [W / f'{tg}.{int(t * V + .5)}.blocks' for t in TG[rule]]
    if not all(f.exists() for f in files):
        subprocess.run([str(TGEN), 'run', spec(*c), str(0xA11C0C + K * 31 + len(form) * 7 + shell), '-'],
                       env={**os.environ, 'TGEN_SNAP': str(W / tg)}, capture_output=True, check=True)
    import surf
    for t, f in zip(TG[rule], files):
        st, _ = surf.stats(surf.load(f))
        out.append({'form': form, 'shell': shell, 'rule': rule, 'K': K, 'T': t, 'mode': st['mode'],
                    'profile': [round(st[k], 3) for k in ('f0', 'f-1', 'f-2', 'f-3', 'f_deep(<=-4)')], 'file': str(f)})
    return out


def gen():
    W.mkdir(exist_ok=True)
    cfg = [(f, s, r, K) for f in FORMS for s in (1, 0) for r in TG for K in KS]
    with Pool(24) as p: rows = [x for xs in p.map(gen_one, cfg) for x in xs]
    (HERE / 'rock-truth.json').write_text(json.dumps(rows, indent=1))
    print(len(rows), 'snapshots;', sum(r['mode'] == 63 or r['mode'] == 62 for r in rows), 'with plain at the top')


def synth_one(a):
    f, out = a
    if not Path(out).exists():
        subprocess.run([sys.executable, str(HERE / 'synth_pc.py'), f, '12', '7', out, 'top'], capture_output=True, cwd=str(TERR))
    return out


def synth():
    SY.mkdir(exist_ok=True)
    rows = json.loads((HERE / 'rock-truth.json').read_text())
    # shortlist: plain at the top (mode >= 62) and truth f0 in [.36, .58] (where the multi-type fits landed: .39-.50)
    pick = [r for r in rows if r['mode'] >= 62 and .36 <= r['profile'][0] <= .58]
    jobs = [(r['file'], str(SY / f"{tag(r['form'], r['shell'], r['rule'], r['K'])}.T{r['T']}.json")) for r in pick]
    print(len(jobs), 'worlds to synthesise', flush=True)
    with Pool(26) as p:
        for i, o in enumerate(p.imap_unordered(synth_one, jobs)):
            if i % 25 == 0: print(i, o, flush=True)


def score():
    import compare as CP
    F = CP.footage('0.35'); fb = {w: F[w]['bins'] for w in ('r1', 'r2')}
    fs = {w: fb[w]['-1'] + fb[w]['-2'] + fb[w]['-3'] for w in ('r1', 'r2')}
    out = []
    for p in sorted(glob.glob(str(SY / '*.json'))):
        try: d, agg, tru, per = CP.synth(p, '0.35')
        except Exception: continue
        b = {w: CP.bins(agg[w]) for w in ('r1', 'r2')}
        if not b['r1'] or not b['r2']: continue
        m = re.match(r'(\w+?)_sh(\d)_(scan|rand)_K(\d+)\.T([\d.]+?)(?:\.s\d)?$', Path(p).stem)
        form, shell, rule, K, T = m.groups()
        l1 = sum(abs(b[w][k] - fb[w][k]) for w in ('r1', 'r2') for k in ('0', '-1', '-2', '-3', '<=-4'))
        vals = {w: [(x['-1'] + x['-2'] + x['-3']) if x else 0.0 for x in per[w]] for w in ('r1', 'r2')}
        P = float(np.mean([v >= fs['r1'] for v in vals['r1']]) * np.mean([v >= fs['r2'] for v in vals['r2']]))
        out.append({'form': form, 'shell': int(shell), 'rule': rule, 'K': int(K), 'T': float(T), 'L1': round(l1, 3), 'P': round(P, 3),
                    'r1': [round(b['r1'][k], 3) for k in ('0', '-1', '-2', '-3', '<=-4')], 'r2': [round(b['r2'][k], 3) for k in ('0', '-1', '-2', '-3', '<=-4')]})
    out.sort(key=lambda r: r['L1'])
    (HERE / 'fit-rock.json').write_text(json.dumps({'footage': fb, 'rows': out}, indent=1))
    print('footage r1', [round(fb['r1'][k], 2) for k in ('0', '-1', '-2', '-3', '<=-4')], 'r2', [round(fb['r2'][k], 2) for k in ('0', '-1', '-2', '-3', '<=-4')])
    for r in out[:30]: print(r)


REFINE = [('Vfull', 1, 'scan', 300), ('Vfull', 1, 'scan', 512), ('E63', 1, 'scan', 256), ('E62', 1, 'scan', 256),
          ('E62', 1, 'scan', 300), ('Vfull', 0, 'rand', 600), ('Vfull', 0, 'scan', 500), ('E62', 0, 'rand', 600)]
TR = {'scan': [5.0, 5.5, 6.0, 6.5, 7.0], 'rand': [20.0, 22.5, 25.0, 27.5, 30.0]}


def refine_one(a):
    (form, shell, rule, K), s = a
    tg = f'{tag(form, shell, rule, K)}_s{s}'; Ts = TR[rule]
    sp = (f'init=unset,ntypes=1,rng=int,pick=xyz,nb={rule},rule=mono,K={K},unsetfill=0,cap=fixed,capy=63,shell={shell},'
          f'{FORMS[form]},sweeps={max(Ts)},snap=' + '/'.join(f'{t}v' for t in Ts))
    subprocess.run([str(TGEN), 'run', sp, str(0xBEEF00 + 7919 * s + K), '-'], env={**os.environ, 'TGEN_SNAP': str(W / tg)}, capture_output=True, check=True)
    return [(str(W / f'{tg}.{int(t * V + .5)}.blocks'), str(SY / f'{tag(form, shell, rule, K)}.T{t}.s{s}.json')) for t in Ts]


def refine():
    with Pool(16) as p: jobs = [j for js in p.map(refine_one, [(c, s) for c in REFINE for s in (1, 2)]) for j in js]
    print(len(jobs), 'refine worlds', flush=True)
    with Pool(26) as p:
        for i, o in enumerate(p.imap_unordered(synth_one, jobs)):
            if i % 20 == 0: print(i, o, flush=True)


if __name__ == '__main__':
    {'gen': gen, 'synth': synth, 'score': score, 'refine': refine}[sys.argv[1]]()
