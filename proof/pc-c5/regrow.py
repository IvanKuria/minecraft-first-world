"""Regrow C5 survivors and score them with the FROZEN LINK scorer (scorer.py + raycast DLL built from unchanged raycast.c).
Lead's rule for this own-Random case: the RNG self-consistency check does not apply (separate Random). PASS iff
T_dark >= 0.30 AND T_dark_r2 >= 0.30 (FROZEN.json floors). A PASS triggers the per-hit null immediately (same logic as
link/null_for_hit.py: same world at 200 random positions + 20 same-variant worlds from random states at the hit's
positions; must beat max by margin 0.15), with c5gen standing in for `gen run` and without the self-consistency term.

usage: python regrow.py <survivor-file> [<survivor-file> ...] [--procs N]
Each survivor (PASS line) is regrown with both growth rules (scan, rand), until filled, ceiling 42, with the SAME
seed-draw spec the GPU search used (from the file header). Output: regrow/<survivor-file-stem>.jsonl (resumable).
"""
import argparse, json, math, os, random, subprocess, sys, tempfile, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pc-replay'))
import pc_replay                                    # noqa: E402  (loads raycast.dll into the frozen scorer)
import scorer                                       # noqa: E402  (frozen, unchanged)
from calibrate import rand_pos
from lcg import reset_xz                            # noqa: E402  (frozen)                      # noqa: E402  (frozen null position sampler)

C5GEN = HERE / 'build' / 'c5gen.exe'
TGEN = HERE / 'build' / 'tgen.exe'   # terrace/tgen.c (GEN + shell keys), MSVC build; matches terrace/validate-tgen.txt
FROZEN = json.loads((HERE.parent / 'link' / 'FROZEN.json').read_text())['thresholds']
STATES = {r['s_open']: r for r in json.loads((HERE.parent / 'link' / 'states-by-hypothesis.json').read_text())['rows'] if r['set'] == 'baseline175'}


# TERRACE lane shell model (terrace/NOTES.txt): slab walls+floor (y<=42) rock from the start, interior unset, GEN C4 rule,
# FIXED T (sweeps) then leftover unset -> AIR. Fitted windows: scan K4096 3.5V, scan K1024 4.5-5V, rand K4096 15V.
SHELL_T = {('scan', 1024): (4.5, 5.0), ('scan', 4096): (3.5,), ('rand', 4096): (15.0,), ('scan', 256): (6.0, 7.0)}


def shell_specs(search_variant):
    """(rule, T, tgen spec) list for the shell model with the SAME seed draws as the GPU search."""
    if search_variant.startswith('X:'):
        kv = dict(p.split('=') for p in search_variant[2:].split(','))
        K, nt, m, yb, tf = int(kv['K']), int(kv['nt']), int(kv['m']), int(kv['yb']), int(kv.get('tf', 0))
        if tf: return []                                   # type-first order not expressible in tgen
    else:
        kv = dict(p.split('=') for p in search_variant.split(':', 1)[1].split(','))
        K, nt, m, yb = int(kv['K']), int(kv['ntypes']), 1, 62
    extra = f',margin={m}'
    if m == 0 and yb != 64: extra += f',sy={yb},sy0=0,py={yb},py0=0'      # V-full42: y = nextInt(43) for seeds and picks
    out = []
    for (rule, k), Ts in SHELL_T.items():
        if k != K: continue
        for T in Ts:
            out.append((rule, T, f'init=unset,ntypes={nt},rng=int,pick=xyz,nb={rule},rule=mono,K={K},unsetfill=0,cap=fixed,capy=42,shell=1,sweeps={T}{extra}'))
    return out


# FULL-HEIGHT shell model (lead, Oct 15 night): plain = top of the world; tgen cap=fixed capy=63 (no discard),
# shell walls+floor, top open, fixed T, leftover -> air. Windows fitted in full/fit-full.json (terrace synth statistic).
FULL_T = {('scan', 4096): (3.0, 3.5), ('rand', 4096): (16.0,), ('scan', 1024): (4.0, 4.5), ('scan', 256): (5.0,)}


def full_specs(search_variant):
    """(rule, T, tgen spec) for an X spec with capy=63 (full-height seed draws)."""
    kv = dict(p.split('=') for p in search_variant[2:].split(','))
    K, nt, m, yb, tf = int(kv['K']), int(kv['nt']), int(kv['m']), int(kv['yb']), int(kv.get('tf', 0))
    if tf: return []
    extra = f',margin={m}' + (f',sy={yb},sy0={m},py={yb},py0={m}' if yb != 64 - 2 * m else '')
    return [(rule, T, f'init=unset,ntypes={nt},rng=int,pick=xyz,nb={rule},rule=mono,K={K},unsetfill=0,cap=fixed,capy=63,shell=1,sweeps={T}{extra}')
            for (rule, k), Ts in FULL_T.items() if k == K for T in Ts]


def grow_spec(search_variant, rule):
    """Map the GPU search's seed-draw variant to a c5gen growth spec (same seed draws, grow until filled)."""
    if search_variant.startswith('X:'):
        base = search_variant[2:]
        if ',m=0' in ',' + base:   # V-full / V-full42: edges forced solid after growth (lead's spec)
            base += ',edge=1'
    else:  # GEN form U6s:K=..,ntypes=.. -> edge-skipping draws m=1, yb=62, type after xyz
        kv = dict(p.split('=') for p in search_variant.split(':', 1)[1].split(','))
        base = f"K={kv['K']},nt={kv['ntypes']},m=1,yb=62,tf=0"
    return base + f",nb={rule},until=1"


_W = {}


def init():
    pc_replay.load_dll()
    _W['frames'] = scorer.load_frames()
    _W['tmp'] = Path(tempfile.mkdtemp(prefix='c5regrow-'))


def grow(spec, g0, path):
    exe = TGEN if 'shell=1' in spec else C5GEN
    out = subprocess.run([str(exe), 'run', spec, str(g0), str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


MAPLL = None   # (P_open_by_depth, [(obs file, fx, fz)]) set per survivor file


def map_loglik(blocks, maps, pop):
    """Stage-2 statistic (stage2_proto.py): sum over the real maps' labelled columns of log P(label | regrown depth)."""
    import numpy as np
    a = blocks.reshape(64, 256, 256) != 0
    top = np.where(a.any(0), 63 - np.argmax(a[::-1], 0), -1)
    vals, cnt = np.unique(top[8:-8, 8:-8], return_counts=True); mode = int(vals[np.argmax(cnt)])
    dm = np.clip(mode - top, 0, len(pop) - 1)
    ll = 0.0
    for obs, fx, fz in maps:
        for x, z, lab in obs:
            X, Z = x + fx, z + fz
            if 0 <= X < 256 and 0 <= Z < 256:
                q = pop[dm[Z, X]]; ll += math.log(q if lab else 1 - q)
    return ll


def job(a):
    key, spec, g0, positions = a[:4]
    p = _W['tmp'] / f'w{os.getpid()}.blocks'
    t = time.time()
    g = grow(spec, g0, p)
    blk = scorer.load_blocks(str(p))
    s = scorer.score(scorer.World(blk), positions, _W['frames'])
    mll = map_loglik(blk, a[4], a[5]) if len(a) > 5 and a[4] else None
    os.remove(p)
    passes = s['T_dark'] >= FROZEN['T_dark_floor'] and s['T_dark_r2'] >= FROZEN['T_dark_r2_floor'] and s['mIoU'] >= FROZEN['mIoU_floor']
    return {'key': key, 'spec': spec, 'G0': g0, 'positions': positions, 'gen': g, 'score': s, 'passes_floor': passes, 'map_loglik': mll,
            'seconds': round(time.time() - t, 2)}


def null_for_hit(rec):
    """Per-hit null (link/null_for_hit.py logic; c5gen instead of gen; no self-consistency term)."""
    rng = random.Random(rec['key'])
    p = _W['tmp'] / 'null.blocks'
    grow(rec['spec'], rec['G0'], p)
    world = scorer.World(scorer.load_blocks(str(p)))
    pos_null = [scorer.score(world, rand_pos(rng), _W['frames'])['T_dark'] for _ in range(FROZEN['null_positions'])]
    world_null = []
    for _ in range(FROZEN['null_worlds']):
        grow(rec['spec'], rng.randrange(1 << 48), p)
        world_null.append(scorer.score(scorer.World(scorer.load_blocks(str(p))), rec['positions'], _W['frames'])['T_dark'])
    s = rec['score']; nmax = max(pos_null + world_null)
    checks = {'T_dark_floor': s['T_dark'] >= FROZEN['T_dark_floor'], 'T_dark_r2_floor': s['T_dark_r2'] >= FROZEN['T_dark_r2_floor'],
              'margin_over_null_max': s['T_dark'] >= nmax + FROZEN['margin_over_null_max'],
              'self_consistent_rng': 'not applicable (own Random, lead instruction)'}
    return {'key': rec['key'], 'null_max': nmax, 'pos_null': pos_null, 'world_null': world_null, 'checks': checks,
            'DECISIVE': all(v is True for k, v in checks.items() if k != 'self_consistent_rng')}


def main():
    a = argparse.ArgumentParser(); a.add_argument('files', nargs='+'); a.add_argument('--procs', type=int, default=12); a.add_argument('--model', choices=['fill', 'shell', 'full'], default='full'); a.add_argument('--maxlines', type=int, default=0); a.add_argument('--fastT', action='store_true')
    args = a.parse_args()
    (HERE / 'regrow').mkdir(exist_ok=True)
    with Pool(args.procs, initializer=init) as pool:
        for fn in args.files:
            fn = Path(fn); lines = fn.read_text().splitlines()
            row = lines[0].split(':', 1)[1].split(); search_variant = lines[1].split()[1]
            s_open = int(row[0])
            ox, oz, _ = reset_xz(s_open)                     # opening = first resetPos from the row's state (frozen lcg.py)
            positions = [[ox, oz], [float(row[4]), float(row[5])]]
            if s_open in STATES: assert abs(STATES[s_open]['positions_opening_r1_r2'][0][0] - ox) < 1e-9
            maps = []
            if len(row) >= 8:     # r12 rows: real maps registered to floor(x2,z2) (R2) and floor(x1,z1) (R1)
                def rd(f): return [(int(l.split()[1]), int(l.split()[2]), int(l.split()[3])) for l in Path(f).read_text().splitlines() if l.startswith('O ')]
                maps = [(rd(HERE.parent / 'c5' / 'footage.obs'), math.floor(float(row[4])), math.floor(float(row[5]))),
                        (rd(HERE.parent / 'r1map' / 'footage-r1.obs'), math.floor(float(row[6])), math.floor(float(row[7])))]
            pop = json.loads((HERE / 'stage2-calibration.json').read_text())['P_open_by_depth']
            out = HERE / {'fill': 'regrow', 'shell': 'regrow-shell', 'full': 'regrow-full'}[args.model] / (fn.stem + '.jsonl'); out.parent.mkdir(exist_ok=True); done = set()
            if out.exists(): done = {json.loads(l)['key'] for l in out.read_text().splitlines()}
            jobs = []
            seen_q = set(); nl = 0
            for l in lines:
                if not (l.startswith('PASS') or l.startswith('RANK')): continue
                if l.split()[3] in seen_q: continue
                seen_q.add(l.split()[3]); nl += 1
                if args.maxlines and nl > args.maxlines: break
                _, ci, j, q, g0 = l.split()[:5]   # single- or multi-map PASS line
                if args.model == 'fill':
                    for rule in ('scan', 'rand'):
                        key = f'{fn.stem}|{j}|{q}|{rule}'
                        if key not in done: jobs.append((key, grow_spec(search_variant, rule), int(g0), positions))
                elif args.model == 'full':
                    fs = full_specs(search_variant)
                    if args.fastT: fs = [x for x in fs if x[0] == 'scan'][-1:]     # one scan T (the later window edge)
                    for rule, T, spec in fs:
                        key = f'{fn.stem}|{j}|{q}|full-{rule}-{T}V'
                        if key not in done: jobs.append((key, spec, int(g0), positions, maps, pop))
                else:
                    for rule, T, spec in shell_specs(search_variant):
                        key = f'{fn.stem}|{j}|{q}|shell-{rule}-{T}V'
                        if key not in done: jobs.append((key, spec, int(g0), positions, maps, pop))
            t0 = time.time(); best = (0, None); n = 0
            with out.open('a') as f:
                for rec in pool.imap_unordered(job, jobs):
                    rec['row'] = lines[0][2:]; rec['search_variant'] = search_variant
                    f.write(json.dumps(rec) + '\n'); f.flush(); n += 1
                    if rec['score']['T_dark'] > best[0]: best = (rec['score']['T_dark'], rec['key'], rec['score']['T_dark_r2'])
                    if rec['passes_floor']:
                        print('FLOOR PASSED', rec['key'], rec['score']['T_dark'], rec['score']['T_dark_r2'], flush=True)
                        (HERE / 'FLOOR_PASSED.txt').open('a').write(rec['key'] + '\n')
                        init_needed = 'frames' not in _W
                        if init_needed: init()
                        nl = null_for_hit(rec)
                        (out.parent / 'nulls.jsonl').open('a').write(json.dumps(nl) + '\n')
                        print('NULL', json.dumps({k: nl[k] for k in ('key', 'null_max', 'checks', 'DECISIVE')}), flush=True)
            print(json.dumps({'file': fn.name, 'regrown': n, 'seconds': round(time.time() - t0, 1), 'best': best}), flush=True)


if __name__ == '__main__':
    main()
