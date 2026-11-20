"""C5 GPU search, round 3 (lead's clean-model priorities, Oct 15 evening). Two maps (R2 footage.obs + R1 footage-r1.obs)
at the r1map/NOTES.txt operating points, plus the single-map R2 list for the top rows, plus the new seed hypotheses.
Survivors -> survivors3/*.txt ; every finished job's survivors are queued for regrow.py (frozen LINK scorer), one
regrow process at a time on the CPU, if the job is flagged regrow and has <= REGROW_MAX survivors.
Resumable: jobs listed in progress3.jsonl are skipped; regrow resumes from regrow/<stem>.jsonl.

Variants (seed-draw forms x K, air 1/8, type after xyz):
  GEN edge-skipping   U6s:K=..,ntypes=8                 (x=nextInt(254)+1, y=nextInt(62)+1, z=nextInt(254)+1)
  V-full              X:K=..,nt=8,m=0,yb=64,tf=0         (rd-160052 idiom: nextInt(256), nextInt(64), nextInt(256))
  V-full42            X:K=..,nt=8,m=0,yb=43,tf=0
Extended later: tf=1 (type first) and nt=6.
"""
import json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
C5 = HERE.parent / 'c5'; R1 = HERE.parent / 'r1map'
GPU = HERE / 'build' / 'c5gpu3.exe'
TIMERS = {'PIT': 1193182, 'PM': 3579545, 'HPET': 14318180}
MAPS2 = f"{C5 / 'footage.obs'},{R1 / 'footage-r1.obs'}"
OP = {('PIT', '24h'): ('3,3', '0.4,0.5'), ('PIT', '78h'): ('3,3', '0.4,0.5'), ('PM', '24h'): ('3,3', '0.5,0.4'),
      ('PM', '78h'): ('3,3', '0.5,0.5'), ('HPET', '24h'): ('3,3', '0.6,0.4'), ('HPET', '78h'): ('3,3', '0.6,0.4')}
STRICT = ('3,3', '0.6,0.4')            # literal-seed scans (many rows x many seeds)
SINGLE = ('4', '0.5')                  # C5 single-map operating point (R2 only)
V6 = [f'{f}{k}' for k in (1024, 4096) for f in ('U6s:K=', 'X:K=')]
VARS = []
for K in (1024, 4096):
    VARS += [f'U6s:K={K},ntypes=8', f'X:K={K},nt=8,m=0,yb=64,tf=0', f'X:K={K},nt=8,m=0,yb=43,tf=0']
EXTRA = []
for K in (1024, 4096):
    for f in ('m=1,yb=62', 'm=0,yb=64', 'm=0,yb=43'):
        EXTRA += [f'X:K={K},nt=8,{f},tf=1', f'X:K={K},nt=6,{f},tf=0']
REGROW_MAX = 10000


def rows(timer, window):
    return [l.strip() for l in (R1 / f'cands-{timer}-{window}-r12.txt').read_text().splitlines() if l.strip()]


def window_of(timer, line):
    return '24h' if line in rows(timer, '24h') else '78h'


def plan():
    J = []   # (label, timer, cands_line(s), maps, cmin, imin, variant, env, regrow)
    def row_jobs(timer, line, variants, single, regrow, window=None):
        w = window or window_of(timer, line)
        for v in variants:
            J.append((f'{timer}{w}-two', timer, [line], MAPS2, *OP[(timer, w)], v, {}, regrow))
            if single: J.append((f'{timer}{w}-R2', timer, [line], str(C5 / 'footage.obs'), *SINGLE, v, {}, regrow))
    def pick(timer, state, calls, k):
        out = []
        for w in ('24h', '78h'):
            for l in rows(timer, w):
                p = l.split()
                if p[0] == state and int(p[1]) == calls and int(p[2]) == k and l not in out: out.append(l)
        return out
    # (1) fresh-launch row, PM and HPET grids; (2) HPET 266841333641652
    for timer in ('PM', 'HPET'):
        for l in pick(timer, '99100327271809', 0, 2): row_jobs(timer, l, VARS, True, True)
    for l in pick('HPET', '266841333641652', 0, 2): row_jobs('HPET', l, VARS, True, True)
    # (3) seed hypotheses that do not touch seedUniquifier -> Math's Random is the JVM's first: rows with k_m == 1, even calls
    k1 = []
    for timer in ('PM', 'PIT', 'HPET'):
        for l in rows(timer, '78h'):
            p = l.split()
            if int(p[2]) == 1 and int(p[1]) % 2 == 0 and (timer, l) not in k1: k1.append((timer, l))
    k1.sort(key=lambda t: int(t[1].split()[1]))          # calls 0 first
    k1_lines, regs = [], set()                            # literal seeds ignore q_m: one line per distinct registration
    for _, l in k1:
        r = tuple(l.split()[4:8])
        if r not in regs: regs.add(r); k1_lines.append(l)
    for v in VARS:
        J.append(('ms-May13', 'PM', k1_lines, MAPS2, *STRICT, v, {'C5_SEEDMODE': 'lit:1242165600000:1242251999999'}, True))
        J.append(('const-2^24', 'PM', k1_lines, MAPS2, *STRICT, v, {'C5_SEEDMODE': 'lit:0:16777215'}, True))
    for timer, l in k1:
        for v in VARS: J.append((f'nano-{timer}', timer, [l], MAPS2, *OP[(timer, window_of(timer, l))], v, {'C5_SEEDMODE': 'nano'}, True))
    # (4) remaining even-calls rows, k<=3 (PM, PIT, HPET), then even k>3, then odd; GEN + V-full forms
    done = {('PM', l) for l in pick('PM', '99100327271809', 0, 2)} | {('HPET', l) for l in pick('HPET', '99100327271809', 0, 2) + pick('HPET', '266841333641652', 0, 2)}
    def tier(pred):
        for timer in ('PM', 'PIT', 'HPET'):
            seen = []
            for w in ('24h', '78h'):
                for l in rows(timer, w):
                    p = l.split()
                    if (timer, l) in done or l in seen or not pred(int(p[1]), int(p[2])): continue
                    seen.append(l); done.add((timer, l)); row_jobs(timer, l, VARS, False, True, w)
    tier(lambda c, k: c % 2 == 0 and k <= 3)
    tier(lambda c, k: c % 2 == 0 and k > 3)
    # secondary order (Math's Random first: k_g = k_m + 1, +0..5 s) on the clean rows, then extended draw forms, then odd rows
    for timer in ('PM', 'HPET'):
        for l in pick(timer, '99100327271809', 0, 2):
            for v in VARS: J.append((f'{timer}-after5s', timer, [l], MAPS2, *OP[(timer, window_of(timer, l))], v, {'C5_AFTER': '5'}, True))
    for timer in ('PM', 'HPET'):
        for l in pick(timer, '99100327271809', 0, 2): row_jobs(timer, l, EXTRA, False, True)
    tier(lambda c, k: c % 2 == 1)
    return J


def key_of(j):
    label, timer, lines, maps, cmin, imin, v, env, _ = j
    return '|'.join([label, timer, ';'.join(l.split()[0] + ':' + l.split()[1] + ':' + l.split()[2] + ':' + l.split()[3] for l in lines)[:200],
                     str(len(lines)), cmin, imin, v, json.dumps(env, sort_keys=True)])


def main():
    prog = HERE / 'progress3.jsonl'
    done = {json.loads(l)['key'] for l in prog.read_text().splitlines()} if prog.exists() else set()
    (HERE / 'survivors3').mkdir(exist_ok=True); (HERE / 'rows3').mkdir(exist_ok=True)
    J = plan()
    print(json.dumps({'jobs': len(J), 'done': sum(key_of(j) in done for j in J)}), flush=True)
    regrow_q, regrow_p = [], None
    def pump():
        nonlocal regrow_p
        if regrow_p and regrow_p.poll() is None: return
        if regrow_q:
            f = regrow_q.pop(0)
            regrow_p = subprocess.Popen(['python', str(HERE / 'regrow.py'), str(f), '--procs', '10'],
                                        stdout=open(HERE / 'regrow3.log', 'a'), stderr=subprocess.STDOUT, cwd=str(HERE))
    for n, j in enumerate(J):
        k = key_of(j)
        label, timer, lines, maps, cmin, imin, v, env, regrow = j
        if k in done:
            continue
        p0 = lines[0].split()
        tag = f"{n:04d}-{label}-{p0[0]}-c{p0[1]}k{p0[2]}" + (f"+{len(lines) - 1}rows" if len(lines) > 1 else '')
        vtag = v.replace(':', '_').replace(',', '_').replace('=', '')
        cf = HERE / 'rows3' / f'{tag}.cands'; cf.write_text('\n'.join(lines) + '\n')
        t = time.time()
        r = subprocess.run([str(GPU), v, str(cf), maps, str(TIMERS[timer]), '10', '40', '2', cmin, imin, '1'],
                           capture_output=True, text=True, env={**os.environ, **env})
        if r.returncode != 0:
            print(json.dumps({'error': tag, 'variant': v, 'rc': r.returncode, 'stderr': r.stderr[-400:]}), flush=True); raise SystemExit(1)
        out = r.stdout.splitlines(); passes = [l for l in out if l.startswith('PASS')]
        sf = HERE / 'survivors3' / f'{tag}-{vtag}.txt'
        if len(lines) == 1:
            sf.write_text(f'# row {timer}: {lines[0]}\n# {v} J=2 delta=10-40s maps={maps} cmin={cmin} imin={imin} env={json.dumps(env)}\n' + '\n'.join(out) + '\n')
        else:   # multi-row literal scans: one survivor file per row so regrow gets the right positions
            sf.write_text(f'# rows {timer}: {len(lines)} (see rows3/{cf.name})\n# {v} maps={maps} cmin={cmin} imin={imin} env={json.dumps(env)}\n' + '\n'.join(out) + '\n')
        summ = [l for l in out if l.startswith('SUMMARY')][0]
        rec = {'key': k, 'n': n, 'label': label, 'timer': timer, 'rows': len(lines), 'row0': lines[0], 'variant': v, 'env': env,
               'cmin': cmin, 'imin': imin, 'tested': int(summ.split('tested=')[1].split()[0]),
               'pass_stage1': int(summ.split('pass_stage1=')[1].split()[0]), 'survivors': len(passes),
               'seconds': round(time.time() - t, 2), 'summary': summ}
        with prog.open('a') as f: f.write(json.dumps(rec) + '\n')
        print(json.dumps({x: rec[x] for x in ('n', 'label', 'row0', 'variant', 'tested', 'survivors', 'seconds')}), flush=True)
        if regrow and 0 < len(passes) <= REGROW_MAX:
            if len(lines) == 1: regrow_q.append(sf)
            else:
                for ci, l in enumerate(lines):          # split multi-row survivors per row
                    mine = [x for x in passes if x.split()[1] == str(ci)]
                    if not mine: continue
                    rf = HERE / 'survivors3' / f'{tag}-{vtag}-row{ci}.txt'
                    rf.write_text(f'# row {timer}: {l}\n# {v} maps={maps} cmin={cmin} imin={imin} env={json.dumps(env)}\n' + '\n'.join(mine) + '\n')
                    regrow_q.append(rf)
        pump()
    while regrow_q or (regrow_p and regrow_p.poll() is None):
        pump(); time.sleep(5)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
