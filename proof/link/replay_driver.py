"""Replay driver: candidate RNG state x call-order hypothesis x GEN variant x extra offset -> world -> frozen score.

Usage (always under the machine-wide heavy lock; the whole driver is ONE heavy job, strictly sequential):
  video/scripts/heavy-lock.sh PY replay_driver.py --set baseline175 --hyp H0_direct --variants G1b,G2c --k 0 \
        --out replay/results.jsonl [--limit N] [--dry-run]
Resumable: a job key already present in --out is skipped. World files are written to a temp path and deleted.

Linkage rule: S_gen_end = s_gen_end[hyp] rewound by 2k (k extra Math.random calls); S_gen_start = S_gen_end rewound by
N(variant). N comes from `gen steps <variant>` when GEN reports a closed form. After `gen run`, the driver REQUIRES
state_out == S_gen_end (exact self-consistency); variants whose consumption depends on the data (rejection loops,
draw-if-free) are therefore rejected unless --variable-n-window is given, in which case N is scanned over the window
and only self-consistent N are scored.

Only variants whose RNG is the JVM-global Math.random stream (rng=math) are meaningful for linkage; rng=int variants
model a private Random and are skipped unless --allow-private (then the result is a structural control only).
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import scorer
from lcg import advance

P = Path(__file__).parent
GEN = P.parent / 'gen' / 'gen'
FROZEN = P / 'FROZEN.json'


def gen_list():
    out = subprocess.run([str(GEN), 'list'], capture_output=True, text=True, check=True).stdout.split('\n')
    return {l.split()[0]: l.split()[1] for l in out if l.strip()}


def gen_steps(variant):
    r = json.loads(subprocess.run([str(GEN), 'steps', variant], capture_output=True, text=True, check=True).stdout)
    return r.get('steps_formula'), r


def gen_run(variant, state, path):
    r = subprocess.run([str(GEN), 'run', variant, str(state), path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def decide(s, frozen):
    t = frozen['thresholds']
    return s['T_dark'] >= t['T_dark_floor'] and s['mIoU'] >= t['mIoU_floor']


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--set', default='baseline175', choices=['baseline175', 'hidden_gap_pairs', 'all'])
    a.add_argument('--hyp', default='H0_direct')
    a.add_argument('--variants', required=True)
    a.add_argument('--k', default='0', help='comma list or a-b range of extra Math.random calls')
    a.add_argument('--out', default=str(P / 'replay/results.jsonl'))
    a.add_argument('--limit', type=int, default=0)
    a.add_argument('--ids', default='')
    a.add_argument('--allow-private', action='store_true')
    a.add_argument('--variable-n-window', type=int, default=0)
    a.add_argument('--dry-run', action='store_true')
    args = a.parse_args()
    ks = (list(range(int(args.k.split('-')[0]), int(args.k.split('-')[1]) + 1)) if '-' in args.k
          else [int(x) for x in args.k.split(',')])
    frozen = json.loads(FROZEN.read_text())
    states = json.loads((P / 'states-by-hypothesis.json').read_text())
    rows = [r for r in states['rows'] if args.set == 'all' or r['set'] == args.set]
    if args.ids:
        keep = {int(x) for x in args.ids.split(',')}; rows = [r for r in rows if r['id'] in keep]
    registry = gen_list()
    variants = args.variants.split(',')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(args.out).exists():
        for line in Path(args.out).read_text().splitlines():
            done.add(json.loads(line)['key'])
    frames = scorer.load_frames()
    jobs = [(r, v, k) for v in variants for k in ks for r in rows]
    if args.limit:
        jobs = jobs[:args.limit]
    print(json.dumps({'jobs': len(jobs), 'already_done': sum(f"{r['id']}|{args.hyp}|{v}|{k}" in done for r, v, k in jobs)}), flush=True)
    tmp = Path(tempfile.mkdtemp(prefix='link-replay-'))
    t0 = time.time(); n = 0
    with open(args.out, 'a') as out:
        for r, v, k in jobs:
            key = f"{r['id']}|{args.hyp}|{v}|{k}"
            if key in done:
                continue
            spec = registry.get(v, v)
            if 'rng=math' not in spec and not args.allow_private:
                print(f'skip {v}: not a Math.random variant (no linkage)', flush=True); continue
            N, meta = gen_steps(v)
            s_end = advance(r['s_gen_end'][args.hyp], -2 * k)
            trials = [N] if not args.variable_n_window else range(N - args.variable_n_window, N + args.variable_n_window + 1)
            for Ntry in trials:
                s_start = advance(s_end, -Ntry)
                rec = {'key': key, 'row': r['id'], 'set': r['set'], 'gaps': [r['gap_before_first'], r['gap_between']],
                       'hyp': args.hyp, 'variant': v, 'spec': spec, 'k': k, 'N': Ntry, 's_gen_start': s_start,
                       's_gen_end': s_end, 'positions': [r['positions_opening_r1_r2'][0], r['positions_opening_r1_r2'][2]]}
                if args.dry_run:
                    out.write(json.dumps({**rec, 'dry_run': True}) + '\n'); continue
                wpath = str(tmp / 'w.blocks')
                g = gen_run(v, s_start, wpath)
                rec['gen'] = {x: g.get(x) for x in ['steps', 'state_out', 'solid', 'seconds', 'fnv64']}
                rec['self_consistent'] = (g['state_out'] == s_end and g['steps'] == Ntry)
                if rec['self_consistent'] or args.allow_private:
                    s = scorer.score(scorer.World(scorer.load_blocks(wpath)), rec['positions'], frames)
                    rec['score'] = s; rec['passes_frozen_floor'] = decide(s, frozen)
                os.remove(wpath)
                out.write(json.dumps(rec) + '\n'); out.flush()
                n += 1
                if rec.get('passes_frozen_floor'):
                    print('FLOOR PASSED -> run null_for_hit.py on', key, flush=True)
            if n and n % 20 == 0:
                print(f'{n} jobs, {(time.time()-t0)/n:.2f} s/job', flush=True)
    print(json.dumps({'ran': n, 'seconds': time.time() - t0}))


if __name__ == '__main__':
    main()
