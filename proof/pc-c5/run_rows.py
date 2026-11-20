"""Real-window C5 stage-1+2 search on the GPU, one candidate row at a time, in the lead's priority order.
Survivors only (no regrowth). Resumable: a (row, variant) already in progress.jsonl is skipped.
Row identity = the exact cands line (s_open calls k_m q_m x2 z2) of its timer; a row present in both the 24 h and 78 h
files is run once (24 h pass). Output: survivors/<timer>-<s_open>-c<calls>k<k_m>-<variant>.txt (PASS lines, cand index 0).
Parameters (lead's spec): J=2, Delta 10-40 s, cmin=4 (of 6 components), imin=0.5, variants U6s K=1024 / K=4096 air 1/8."""
import json, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
C5 = HERE.parent / 'c5'
TIMERS = {'PIT': 1193182, 'PM': 3579545, 'HPET': 14318180}
VARIANTS = ['U6s:K=1024,ntypes=8', 'U6s:K=4096,ntypes=8']
J, DMIN, DMAX, CMIN, IMIN = 2, 10, 40, 4, 0.5
PRIORITY = [('PM', '88325364808660', 7), ('PM', '12004448324090', 4),
            ('PIT', '128528575296967', None), ('PIT', '228497643919992', None), ('PIT', '68704517177180', None)]


def rows_of(timer, window):
    return [l.strip() for l in (C5 / f'cands-{timer}-{window}.txt').read_text().splitlines() if l.strip()]


def plan():
    seen, order = set(), []
    def add(timer, window, line):
        if (timer, line) in seen: return
        seen.add((timer, line)); order.append((timer, window, line))
    for timer, state, k in PRIORITY:
        for window in ('24h', '78h'):
            for l in rows_of(timer, window):
                p = l.split()
                if p[0] == state and (k is None or int(p[2]) == k): add(timer, window, l)
    for window in ('24h', '78h'):
        for timer in ('PM', 'PIT', 'HPET'):
            for l in rows_of(timer, window): add(timer, window, l)
    return order


def main():
    prog = HERE / 'progress.jsonl'; done = set()
    if prog.exists():
        for l in prog.read_text().splitlines():
            d = json.loads(l); done.add((d['timer'], d['row'], d['variant']))
    (HERE / 'survivors').mkdir(exist_ok=True); (HERE / 'rows').mkdir(exist_ok=True)
    order = plan()
    print(json.dumps({'rows': len(order), 'jobs': len(order) * len(VARIANTS)}), flush=True)
    for n, (timer, window, line) in enumerate(order):
        p = line.split(); tag = f'{timer}-{p[0]}-c{p[1]}k{p[2]}'
        cf = HERE / 'rows' / f'{tag}.cands'; cf.write_text(line + '\n')
        for v in VARIANTS:
            if (timer, line, v) in done: continue
            t = time.time()
            r = subprocess.run([str(HERE / 'build/c5gpu.exe'), v, str(cf), str(C5 / 'footage.obs'), str(TIMERS[timer]), str(DMIN), str(DMAX),
                                str(J), str(CMIN), str(IMIN), '1'], capture_output=True, text=True)
            if r.returncode != 0:
                print(json.dumps({'error': tag, 'variant': v, 'rc': r.returncode, 'stderr': r.stderr[-500:]}), flush=True); raise SystemExit(1)
            out = r.stdout.splitlines(); passes = [l for l in out if l.startswith('PASS')]
            vtag = v.replace(':', '_').replace(',', '_').replace('=', '')
            (HERE / 'survivors' / f'{tag}-{vtag}.txt').write_text(f'# row {timer} {window}: {line}\n# {v} J={J} delta={DMIN}-{DMAX}s cmin={CMIN} imin={IMIN}\n' + '\n'.join(out) + '\n')
            summ = [l for l in out if l.startswith('SUMMARY')][0]
            rec = {'n': n, 'timer': timer, 'window': window, 'row': line, 'variant': v, 'tested': int(summ.split('tested=')[1].split()[0]),
                   'pass_stage1': int(summ.split('pass_stage1=')[1].split()[0]), 'survivors': len(passes),
                   'seconds': round(time.time() - t, 2), 'gpu': r.stderr.strip().splitlines()[-1]}
            with prog.open('a') as f: f.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
