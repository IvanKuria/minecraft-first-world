"""Frozen fingerprint scan of the footage RNG candidates (JDK6 new Random() hypothesis).

Phase 1 (`python3 run_scan.py freeze`): writes FROZEN_RULES.json (families, timer
scenarios, decoys, decision rule) and the derived state files. Nothing is scanned.
Phase 2 (`python3 run_scan.py scan`): refuses to run unless FROZEN_RULES.json
matches the hash recorded at freeze time; writes scan-results.json.
"""
import hashlib, json, random, subprocess, sys, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RT = HERE.parents[1] / 'runtime/conditional-two-reset'
A, C, M48 = 0x5DEECE66D, 11, 1 << 48
U = 8682522807148012           # JDK6 java.util.Random seedUniquifier initial value (sources/jdk6-Random.java:78)
W, H, D = 256, 256, 64

def back(s, n):
    ainv = pow(A, -1, M48)
    for _ in range(n): s = ainv * (s - C) % M48
    return s

def families():
    fam = {}
    # A: generator used its OWN Random (GEN rng=int variants); Math.random's Random is created lazily
    #    by the first resetPos, so 0..16 Math.random calls precede the opening x draw.
    fam['A_math_random_first_used_at_spawn'] = {'calls': list(range(0, 17)),
        'why': 'GEN rng=int variants / LINK H0-H1: Math.random first called by Player.resetPos (0) or after a few prefilm draws'}
    # B: GEN registry Math.random variants exactly as registered (placeholder K=1000, sweeps=32), plus LINK H0/H1 prefilm resets.
    regsteps = [805312368, 1073749824]   # gen steps G1b ; G2c=G3c   (LCG steps = 2*calls)
    fam['B_gen_registry_exact'] = {'calls': sorted({s // 2 + 2 * j for s in regsteps for j in range(9)}),
        'why': 'gen steps G1b/G2c/G3c (K=1000,sweeps=32) + 0..8 prefilm resets (2 calls each)'}
    # C: GEN structural formula calls = K*sd + Mi*pd with plausible round K and iteration counts.
    Ks = [1, 10, 16, 32, 50, 64, 100, 128, 200, 256, 500, 512, 1000, 1024, 2000, 2048, 4096, 5000, 10000, 65536]
    Ms = sorted({s * W * H * D for s in [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100]} |
                {m * 10**6 for m in [1, 2, 5, 10, 20, 50, 100, 200]} | {2**j for j in range(20, 29)})
    fam['C_gen_structured'] = {'calls': sorted({K * sd + Mi * pd for K in Ks for Mi in Ms for sd in (3, 4) for pd in (3, 4)}),
        'why': 'calls = K*seedDraws + M*(pickDraws+nbDraws), seedDraws,pick+nb in {3,4}; K,M round values'}
    return fam

TIMERS = {'PIT_1193182': 1193182, 'ACPI_PM_3579545': 3579545, 'HPET_14318180': 14318180, 'TSC_2394MHz_dense': 2394000000}
WINDOWS = {'uptime_le_1h': 3600 * 10**9, 'uptime_le_24h': 86400 * 10**9, 'uptime_le_2^48ns_78h': M48 - 1}
TMIN = 30 * 10**9
KRANGE = [1, 8]

def freeze():
    hold = json.loads((RT / 'opening-holdout-results.json').read_text())
    gaps = json.loads((RT / 'hidden-reset-gap-results.json').read_text())
    big = json.loads((RT / 'candidate-internal-states.json').read_text())
    sets = {}
    sets['baseline175'] = sorted({back(s['state_before_first_observed_reset'], 4) for s in hold['survivors']})
    sets['hidden_gap_4385pairs_unique'] = sorted({s for t in gaps['tests'] for s in t['initial_internal_state_candidates']})
    sets['all63074'] = sorted({back(s, 4) for s in big})
    rng = random.Random(2)
    sets['control_uniform_4385'] = sorted(rng.randrange(M48) for _ in range(len(sets['hidden_gap_4385pairs_unique'])))
    for name, st in sets.items():
        (HERE / f'states-{name}.txt').write_text(''.join(f'{s}\n' for s in st))
    drng = random.Random(1)
    decoys = [U + drng.randrange(1, M48) for _ in range(10)]
    fam = families()
    for name, f in fam.items():
        (HERE / f'nlist-{name}.txt').write_text(''.join(f'{n}\n' for n in f['calls']))
    rules = {
        'hypothesis': 'Math.random Random = JDK6 new Random(): seed=(U+k)+nanoTime, nanoTime=(jlong)((qpc/(double)f)*1e9) [HotSpot6 os_windows.cpp]; '
                      'state before opening x draw = X0 advanced by 2*calls LCG steps',
        'U': U, 'k_range': KRANGE, 'tmin_ns': TMIN, 'timers': TIMERS, 'windows': WINDOWS, 'decoy_U': decoys,
        'families': {k: {'n_values': len(v['calls']), 'why': v['why'],
                         'sha256': hashlib.sha256((HERE / f'nlist-{k}.txt').read_bytes()).hexdigest()} for k, v in fam.items()},
        'state_sets': {k: {'n': len(v), 'sha256': hashlib.sha256((HERE / f'states-{k}.txt').read_bytes()).hexdigest()} for k, v in sets.items()},
        'inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [RT / 'opening-holdout-results.json', RT / 'hidden-reset-gap-results.json', RT / 'candidate-internal-states.json']},
        'runs': 'family A,B on all sets; family C on baseline175, hidden_gap_4385pairs_unique and control_uniform_4385 (cost)',
        'decision_rule': ('A cell (set,family,timer,window) shows SET-LEVEL evidence only if the real-U pass count exceeds BOTH the max over the 10 decoy-U '
                          'counts AND the Poisson upper 0.999 quantile of the analytic expectation lambda. Individual passes are listed as '
                          '"consistent with H", never as authentication. Pre-registered prediction: since H predicts at most ONE true pass per set, '
                          'no cell with lambda >= ~0.1 can show set-level evidence; only cells with lambda << 1 are informative.'),
    }
    txt = json.dumps(rules, indent=2) + '\n'
    (HERE / 'FROZEN_RULES.json').write_text(txt)
    (HERE / 'FROZEN_RULES.sha256').write_text(hashlib.sha256(txt.encode()).hexdigest() + '\n')
    print(json.dumps({k: v['n'] for k, v in rules['state_sets'].items()}), {k: v['n_values'] for k, v in rules['families'].items()})

def grid_count(f, tmin, tmax):
    # number of distinct HotSpot nanoTime values in [tmin,tmax]: spacing 1e9/f > 1 -> f*(span)/1e9 ; else every integer
    return (tmax - tmin) * min(f, 10**9) / 10**9

def poisson_upper(lam, q=0.999):
    if lam > 50: return lam + 3.09 * math.sqrt(lam) + 1
    k, p, cdf = 0, math.exp(-lam), math.exp(-lam)
    while cdf < q:
        k += 1; p *= lam / k; cdf += p
    return k

def scan():
    txt = (HERE / 'FROZEN_RULES.json').read_text()
    assert hashlib.sha256(txt.encode()).hexdigest() == (HERE / 'FROZEN_RULES.sha256').read_text().strip(), 'rules changed after freeze'
    rules = json.loads(txt)
    exe = str(HERE / 'build/fpscan')
    cells = []
    for setname in rules['state_sets']:
        for fam in rules['families']:
            if fam.startswith('C_') and setname == 'all63074': continue
            nset = rules['state_sets'][setname]['n']; nfam = rules['families'][fam]['n_values']
            trials = nset * nfam * (KRANGE[1] - KRANGE[0] + 1)
            for tname, f in rules['timers'].items():
                for wname, tmax in rules['windows'].items():
                    lam = trials * grid_count(f, TMIN, tmax) / M48
                    maxprint = 200 if lam < 5000 else 0
                    def run(u, mp):
                        r = subprocess.run([exe, str(HERE / f'states-{setname}.txt'), str(HERE / f'nlist-{fam}.txt'),
                                            str(KRANGE[0]), str(KRANGE[1]), str(f), str(TMIN), str(tmax), str(u), str(mp)],
                                           capture_output=True, text=True, check=True)
                        lines = r.stdout.splitlines()
                        passes = int(lines[-1].split('passes=')[1].split()[0])
                        return passes, [l.split()[1:] for l in lines if l.startswith('PASS')]
                    if lam > 1e6:   # pass probability ~1: no information; skip decoys to save time
                        cells.append({'set': setname, 'family': fam, 'timer': tname, 'window': wname, 'trials': trials,
                                      'lambda': lam, 'skipped': 'lambda>1e6, test carries no information'}); continue
                    real, listed = run(rules['U'], maxprint)
                    dec = [run(u, 0)[0] for u in rules['decoy_U']]
                    upper = poisson_upper(lam)
                    cell = {'set': setname, 'family': fam, 'timer': tname, 'window': wname, 'trials': trials,
                            'lambda': lam, 'real_passes': real, 'decoy_passes': dec, 'poisson_0.999_upper': upper,
                            'set_level_evidence': real > max(dec) and real > upper}
                    if listed:
                        cell['listed_passes(idx,calls,k,nanos,qpc)'] = listed
                    cells.append(cell)
                    print(json.dumps({k: v for k, v in cell.items() if not k.startswith('listed')}), flush=True)
    (HERE / 'scan-results.json').write_text(json.dumps({'rules_sha256': (HERE / 'FROZEN_RULES.sha256').read_text().strip(),
                                                        'cells': cells}, indent=1) + '\n')

if __name__ == '__main__':
    {'freeze': freeze, 'scan': scan}[sys.argv[1]]()
