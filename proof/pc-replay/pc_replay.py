"""PC parallel replay (Windows, 32 threads). Same record format, job key, linkage and frozen decision as
link/replay_driver.py; the frozen files (scorer.py, raycast.c, lcg.py, FROZEN.json, states-by-hypothesis.json) are
imported/read unchanged. The only platform difference: scorer._lib is pre-set to raycast.dll, built with MSVC from the
unmodified link/raycast.c (scorer.lib() would otherwise try `cc` and a .dylib).

Usage:
  python pc_replay.py run --set baseline175|all --hyps H0_direct,H1_prefilm_resets_1,... --variants U6sm,U6rm:sweeps=5
                         --k 0 --out replay.jsonl --procs 28
  python pc_replay.py verify          # re-run link/replay/e2e-test.jsonl keys and compare with the Mac results
"""
import argparse, ctypes, json, os, subprocess, sys, tempfile, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINK = HERE.parent / 'link'
GEN = HERE.parent / 'gen' / 'gen.exe'
sys.path.insert(0, str(LINK))
import scorer                      # noqa: E402  (frozen)
from lcg import advance            # noqa: E402  (frozen)


def load_dll():
    lib = ctypes.CDLL(str(HERE / 'raycast.dll'))
    d = ctypes.POINTER(ctypes.c_double); u8 = ctypes.POINTER(ctypes.c_uint8)
    lib.cast_rays.argtypes = [u8, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                              ctypes.c_double, d, d, d, ctypes.c_int, ctypes.c_double, u8, d, u8,
                              ctypes.POINTER(ctypes.c_int32)]
    scorer._lib = lib


def gen_cmd(*a):
    return subprocess.run([str(GEN), *map(str, a)], capture_output=True, text=True, check=True).stdout


def decide(s, frozen):   # identical to replay_driver.decide
    t = frozen['thresholds']
    return s['T_dark'] >= t['T_dark_floor'] and s['mIoU'] >= t['mIoU_floor']


_W = {}


def init_worker():
    load_dll()
    _W['frames'] = scorer.load_frames()
    _W['frozen'] = json.loads((LINK / 'FROZEN.json').read_text())
    _W['tmp'] = Path(tempfile.mkdtemp(prefix='pcreplay-'))


def run_job(job):
    r, hyp, v, spec, N, k = job
    key = f"{r['id']}|{hyp}|{v}|{k}"
    s_end = advance(r['s_gen_end'][hyp], -2 * k)
    s_start = advance(s_end, -N)
    rec = {'key': key, 'row': r['id'], 'set': r['set'], 'gaps': [r['gap_before_first'], r['gap_between']],
           'hyp': hyp, 'variant': v, 'spec': spec, 'k': k, 'N': N, 's_gen_start': s_start, 's_gen_end': s_end,
           'positions': [r['positions_opening_r1_r2'][0], r['positions_opening_r1_r2'][2]]}
    wpath = _W['tmp'] / f'w{os.getpid()}.blocks'
    t0 = time.time()
    g = json.loads(gen_cmd('run', v, s_start, wpath).strip().splitlines()[-1])
    rec['gen'] = {x: g.get(x) for x in ['steps', 'state_out', 'solid', 'seconds', 'fnv64']}
    rec['self_consistent'] = (g['state_out'] == s_end and g['steps'] == N)
    if rec['self_consistent']:
        s = scorer.score(scorer.World(scorer.load_blocks(str(wpath))), rec['positions'], _W['frames'])
        rec['score'] = s; rec['passes_frozen_floor'] = decide(s, _W['frozen'])
    os.remove(wpath)
    rec['wall_seconds'] = round(time.time() - t0, 3)
    return rec


def registry():
    return {l.split()[0]: l.split()[1] for l in gen_cmd('list').splitlines() if l.strip()}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('mode', choices=['run', 'verify'])
    a.add_argument('--set', default='baseline175')
    a.add_argument('--hyps', default='H0_direct')
    a.add_argument('--variants', default='U6sm')
    a.add_argument('--k', default='0')
    a.add_argument('--out', default=str(HERE / 'replay.jsonl'))
    a.add_argument('--procs', type=int, default=28)
    args = a.parse_args()
    states = json.loads((LINK / 'states-by-hypothesis.json').read_text())
    byid = {r['id']: r for r in states['rows']}
    reg = registry()
    if args.mode == 'verify':
        ref = [json.loads(l) for l in (LINK / 'replay/e2e-test.jsonl').read_text().splitlines()]
        jobs = []
        for x in ref:
            N = json.loads(gen_cmd('steps', x['variant']))['steps_formula']
            jobs.append((byid[x['row']], x['hyp'], x['variant'], reg.get(x['variant'], x['variant']), N, x['k']))
        with Pool(min(len(jobs), args.procs), initializer=init_worker) as p:
            got = p.map(run_job, jobs)
        bad = 0
        for x, y in zip(ref, got):
            same = (x['gen']['fnv64'] == y['gen']['fnv64'] and x['gen']['state_out'] == y['gen']['state_out']
                    and abs(x['score']['T_dark'] - y['score']['T_dark']) < 1e-9 and abs(x['score']['mIoU'] - y['score']['mIoU']) < 1e-9)
            bad += not same
            print(x['key'], 'MATCH' if same else 'DIFF', x['score']['T_dark'], y['score']['T_dark'], x['score']['mIoU'], y['score']['mIoU'])
        print(json.dumps({'verify_cases': len(ref), 'mismatches': bad}))
        return
    rows = [r for r in states['rows'] if args.set == 'all' or r['set'] == args.set]
    ks = [int(x) for x in args.k.split(',')]
    jobs = []
    for v in args.variants.split(','):
        meta = json.loads(gen_cmd('steps', v))
        spec = reg.get(v, v)
        if meta['rng'] != 'math':
            print(f'skip {v}: not a Math.random variant (no linkage)'); continue
        if meta['steps_formula'] < 0:
            print(f'skip {v}: no closed-form N'); continue
        for hyp in args.hyps.split(','):
            for k in ks:
                for r in rows:
                    jobs.append((r, hyp, v, spec, meta['steps_formula'], k))
    done = set()
    out = Path(args.out)
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)['key'])
    todo = [j for j in jobs if f"{j[0]['id']}|{j[1]}|{j[2]}|{j[5]}" not in done]
    print(json.dumps({'jobs': len(jobs), 'already_done': len(jobs) - len(todo)}), flush=True)
    t0 = time.time(); n = 0; best = None
    with Pool(args.procs, initializer=init_worker) as p, out.open('a') as f:
        for rec in p.imap_unordered(run_job, todo, chunksize=1):
            f.write(json.dumps(rec) + '\n'); f.flush(); n += 1
            if not rec['self_consistent']:
                print('NOT SELF-CONSISTENT', rec['key'], flush=True)
            elif best is None or rec['score']['T_dark'] > best[0]:
                best = (rec['score']['T_dark'], rec['key'])
            if rec.get('passes_frozen_floor'):
                print('FLOOR PASSED ->', rec['key'], json.dumps({x: rec['score'][x] for x in ['T_dark', 'T_dark_r2', 'mIoU']}), flush=True)
                (HERE / 'FLOOR_PASSED.txt').open('a').write(rec['key'] + '\n')
            if n % 200 == 0:
                print(f'{n}/{len(todo)} {(time.time()-t0):.0f}s best={best}', flush=True)
    print(json.dumps({'ran': n, 'seconds': round(time.time() - t0, 1), 'best': best}), flush=True)


if __name__ == '__main__':
    main()
