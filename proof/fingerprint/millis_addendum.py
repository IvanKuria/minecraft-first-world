"""POST-FREEZE EXPLORATORY addendum (not in FROZEN_RULES.json; labelled as such).

HotSpot6 Windows javaTimeNanos falls back to javaTimeMillis()*1e6 when
QueryPerformanceFrequency fails (os_windows.cpp:824-825). Then nanoTime is
wall-clock-anchored and the filming day bounds it tightly, which would make the
test strong.  On ACPI XP hardware QPF essentially always succeeds, so the prior
for this branch is very low; it is checked only because it is cheap.

t = ms*10^6 with ms in a wall-clock window; solve ms from t mod 2^48 exactly:
10^6 = 2^6 * 15625, so r must be divisible by 64 and ms is unique mod 2^42.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
A, C, M48 = 0x5DEECE66D, 11, 1 << 48
U = 8682522807148012
AINV = pow(A, -1, M48)
INV15625 = pow(15625, -1, 1 << 42)
MS0 = int(datetime(2009, 5, 12, 12, tzinfo=timezone.utc).timestamp() * 1000)
MS1 = int(datetime(2009, 5, 14, 0, tzinfo=timezone.utc).timestamp() * 1000)

def back_aff(steps):
    m, b, am, ab = 1, 0, AINV, AINV * (-C) % M48
    while steps:
        if steps & 1: m, b = am * m % M48, (am * b + ab) % M48
        am, ab = am * am % M48, (am * ab + ab) % M48
        steps >>= 1
    return m, b

def ms_solutions(r):
    if r % 64: return []
    base = (r // 64) * INV15625 % (1 << 42)
    out = []; ms = base + ((MS0 - base + (1 << 42) - 1) // (1 << 42)) * (1 << 42) if base < MS0 else base
    while ms <= MS1:
        if ms >= MS0: out.append(ms)
        ms += 1 << 42
    return out

def run(setname, fam, u, krange=(1, 8)):
    states = [int(l) for l in (HERE / f'states-{setname}.txt').read_text().split()]
    ns = [int(l) for l in (HERE / f'nlist-{fam}.txt').read_text().split()]
    passes = []; trials = 0
    for n in ns:
        m, b = back_aff(2 * n)
        for i, s in enumerate(states):
            x0 = (m * s + b) % M48 ^ (A % M48)
            for k in range(krange[0], krange[1] + 1):
                trials += 1
                for ms in ms_solutions((x0 - (u + k)) % M48):
                    passes.append([i, n, k, ms])
    return trials, passes

def main():
    rules = json.loads((HERE / 'FROZEN_RULES.json').read_text())
    res = []
    window_vals = MS1 - MS0
    for setname in ['baseline175', 'hidden_gap_4385pairs_unique']:
        for fam in ['A_math_random_first_used_at_spawn', 'B_gen_registry_exact', 'C_gen_structured']:
            if fam.startswith('C') and setname != 'baseline175': continue
            trials, passes = run(setname, fam, U)
            dec = [len(run(setname, fam, d)[1]) for d in rules['decoy_U'][:3]]
            lam = trials * window_vals / M48
            res.append({'set': setname, 'family': fam, 'trials': trials, 'lambda': lam, 'real_passes': len(passes),
                        'passes(idx,calls,k,ms)': passes, 'decoy_passes_first3': dec})
            print(json.dumps({k: v for k, v in res[-1].items() if not k.startswith('passes')}), flush=True)
    (HERE / 'millis-addendum.json').write_text(json.dumps({
        'status': 'POST-FREEZE exploratory; QPC-unavailable branch, very low prior on XP ACPI hardware',
        'ms_window_utc': ['2009-05-12T12:00Z', '2009-05-14T00:00Z'], 'results': res}, indent=1) + '\n')

if __name__ == '__main__':
    main()
