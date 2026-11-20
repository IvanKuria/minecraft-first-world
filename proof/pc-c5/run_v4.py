"""Round 4 (full-height model): two-map C5 with ranking on the clean calls-0 rows and the hidden-reset (held-R) rows,
regrowth of the ranked lists with full-height tgen at the fitted windows, frozen LINK scorer (regrow.py --model full).
Order: named clean rows (calls 0, k 2/3) -> hidden-ext PM -> PIT -> HPET. Resumable via progress4.jsonl / regrow-full/."""
import json, os, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
C5 = HERE.parent / 'c5'; R1 = HERE.parent / 'r1map'; HX = HERE.parent / 'hidden-ext'
GPU = HERE / 'build' / 'c5gpu5.exe'
TIMERS = {'PIT': 1193182, 'PM': 3579545, 'HPET': 14318180}
MAPS2 = f"{C5 / 'footage.obs'},{R1 / 'footage-r1.obs'}"
OP = {'PIT': ('3,3', '0.4,0.5'), 'PM': ('3,3', '0.5,0.5'), 'HPET': ('3,3', '0.6,0.4')}
VARS = ['X:K=4096,nt=8,m=0,yb=64,tf=0,capy=63,surf=63', 'X:K=4096,nt=6,m=1,yb=62,tf=0,capy=63,surf=62',
        'X:K=4096,nt=8,m=1,yb=63,tf=0,capy=63,surf=63', 'X:K=4096,nt=12,m=0,yb=64,tf=0,capy=63,surf=63',
        'X:K=1024,nt=12,m=1,yb=62,tf=0,capy=63,surf=62', 'X:K=1024,nt=8,m=0,yb=64,tf=0,capy=63,surf=63']
CLEAN = [('PM', '99100327271809', 2), ('HPET', '99100327271809', 2), ('HPET', '266841333641652', 2),
         ('PM', '233513965841111', 3), ('HPET', '233513965841111', 3), ('HPET', '14593523350806', 3), ('HPET', '170375616623189', 3)]


def plan():
    J = []
    for timer, st, k in CLEAN:
        for w in ('24h', '78h'):
            for l in (R1 / f'cands-{timer}-{w}-r12.txt').read_text().splitlines():
                p = l.split()
                if p and p[0] == st and p[1] == '0' and int(p[2]) == k and not any(j[2] == l and j[1] == timer for j in J):
                    for v in VARS: J.append(('clean', timer, l, v, 2000, '3,2', 2000, False))
    # hidden-ext: only the lead's top class (k in {2,3}, 2..9 hidden resets per press: tag 'pri(0, 0, ...)'), PM then PIT,
    # two best-recall variants (planted check: E62 K4096 nt6 55%, Vfull K4096 nt8 40% in top-2000 of 1e6). HPET deferred.
    EXT_VARS = ['X:K=4096,nt=6,m=1,yb=62,tf=0,capy=63,surf=62', 'X:K=4096,nt=8,m=0,yb=64,tf=0,capy=63,surf=63']
    for timer in ('PM', 'PIT'):
        f = HX / f'cands-{timer}-78h-ext.txt'; tg = HX / f'cands-{timer}-78h-ext.tags'
        if not f.exists(): continue
        for l, t in zip(f.read_text().splitlines(), tg.read_text().splitlines()):
            if l.strip() and 'pri(0, 0,' in t:
                for v in EXT_VARS: J.append(('ext', timer, l.strip(), v, 300, '3,3', 200, True))
    return J


def key_of(j): return '|'.join([j[0], j[1], j[2], j[3]])


def main():
    prog = HERE / 'progress4.jsonl'
    done = {json.loads(l)['key'] for l in prog.read_text().splitlines()} if prog.exists() else set()
    (HERE / 'survivors4').mkdir(exist_ok=True); (HERE / 'rows4').mkdir(exist_ok=True)
    J = plan(); print(json.dumps({'jobs': len(J), 'done': sum(key_of(j) in done for j in J)}), flush=True)
    q, proc = [], None
    def pump():
        nonlocal proc
        if proc and proc.poll() is None: return
        if q:
            f, n, fast = q.pop(0)
            proc = subprocess.Popen(['python', str(HERE / 'regrow.py'), str(f), '--procs', '20', '--model', 'full', '--maxlines', str(n)] + (['--fastT'] if fast else []),
                                    stdout=open(HERE / 'regrow4.log', 'a'), stderr=subprocess.STDOUT, cwd=str(HERE),
                                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    for n, j in enumerate(J):                       # restart: re-queue regrowth of finished jobs (regrow.py resumes by key)
        if key_of(j) in done:
            p0 = j[2].split(); vt = j[3].replace(':', '_').replace(',', '_').replace('=', '')
            sf = HERE / 'survivors4' / f'{n:05d}-{j[0]}-{j[1]}-{p0[0]}-c{p0[1]}k{p0[2]}-{vt}.txt'
            if sf.exists(): q.append((sf, j[6], j[7]))
    pump()
    for n, j in enumerate(J):
        label, timer, line, v, topn, rcmin, nreg, fast = j
        if key_of(j) in done: continue
        p0 = line.split(); tag = f'{n:05d}-{label}-{timer}-{p0[0]}-c{p0[1]}k{p0[2]}'
        vtag = v.replace(':', '_').replace(',', '_').replace('=', '')
        cf = HERE / 'rows4' / f'{tag}.cands'; cf.write_text(line + '\n')
        cmin, imin = OP[timer]; t = time.time()
        r = subprocess.run([str(GPU), v, str(cf), MAPS2, str(TIMERS[timer]), '10', '40', '2', cmin, imin, '1'], capture_output=True, text=True,
                           env={**os.environ, 'C5_TOPN': str(topn), 'C5_RANK_CMIN': rcmin})
        if r.returncode != 0: print(json.dumps({'error': tag, 'rc': r.returncode, 'err': r.stderr[-300:]}), flush=True); raise SystemExit(1)
        out = r.stdout.splitlines()
        sf = HERE / 'survivors4' / f'{tag}-{vtag}.txt'
        sf.write_text(f'# row {timer}: {line}\n# {v} J=2 delta=10-40s maps=two cmin={cmin} imin={imin} topN={topn} rank_cmin={rcmin}\n' + '\n'.join(out) + '\n')
        summ = [l for l in out if l.startswith('SUMMARY')][0]
        rk = [l.split() for l in out if l.startswith('RANK')]
        rec = {'key': key_of(j), 'n': n, 'label': label, 'timer': timer, 'row': line, 'variant': v, 'tested': int(summ.split('tested=')[1].split()[0]),
               'pass': sum(l.startswith('PASS') for l in out), 'ranked': len(rk), 'best_rank_score': float(rk[0][5]) if rk else None,
               'seconds': round(time.time() - t, 1), 'gpu': r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}
        with prog.open('a') as f: f.write(json.dumps(rec) + '\n')
        print(json.dumps({k: rec[k] for k in ('n', 'label', 'timer', 'variant', 'tested', 'pass', 'ranked', 'best_rank_score', 'seconds')}), flush=True)
        if rk or rec['pass']: q.append((sf, nreg, fast))
        pump()
    while q or (proc and proc.poll() is None): pump(); time.sleep(10)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
