"""Turn the extended hidden-reset scenarios into C5 candidate rows under the clean model:
fresh launch (opening = first resetPos, calls = 0) + JDK6 `new Random()` fingerprint for Math's Random (k = 1..4),
for PIT / ACPI-PM / HPET QPC grids, uptime 30 s .. 78 h.  Row format (as c5/r1map cands): state calls k q x2 z2 x1 z1  + scenario tag."""
import json, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R / 'runtime'))
from benchmark_spawn_recovery import advance, coordinate
OUT = Path(__file__).resolve().parent
lines = [json.loads(l) for l in (R / 'runtime/conditional-two-reset/hidden-reset-gap-results-ext.jsonl').read_text().splitlines()]
scen = {}  # S0 -> list of (g1, g2)
for r in lines:
    for s in r['initial_internal_state_candidates']:
        scen.setdefault(s, []).append((r['hidden_resets_before_first_observed'], r['hidden_resets_between_observed']))
states = sorted(scen); (OUT / 'states.txt').write_text('\n'.join(map(str, states)) + '\n'); (OUT / 'nlist0.txt').write_text('0\n')
fp = R / 'proof/fingerprint/build/fpscan'
U, TMIN, TMAX = 8682522807148012, 30_000_000_000, 78 * 3600 * 10**9
def pos(S, i): return coordinate(advance(S, 2 * i + 1))
summary = {}
for name, f in [('PIT', 1193182), ('PM', 3579545), ('HPET', 14318180)]:
    out = subprocess.run([str(fp), str(OUT / 'states.txt'), str(OUT / 'nlist0.txt'), '1', '4', str(f), str(TMIN), str(TMAX), str(U), '100000000'],
                         capture_output=True, text=True).stdout
    rows = []
    for l in out.splitlines():
        if not l.startswith('PASS'): continue
        _, idx, N, k, t, q = l.split()[:6]; S = states[int(idx)]
        for g1, g2 in scen[S]:
            x1, z1 = pos(S, 2 + 2 * g1), pos(S, 3 + 2 * g1)
            x2, z2 = pos(S, 4 + 2 * (g1 + g2)), pos(S, 5 + 2 * (g1 + g2))
            pri = (0 if k in ('2', '3') else 1, 0 if (2 <= g1 <= 9 and 2 <= g2 <= 9) else 1, abs(g1 - 5) + abs(g2 - 5))
            rows.append((pri, f'{S} 0 {k} {q} {x2} {z2} {x1} {z1}', f'{S} k{k} g{g1}_{g2} pri{pri}'))
    rows.sort(key=lambda r: r[0])
    (OUT / f'cands-{name}-78h-ext.txt').write_text(''.join(r[1] + '\n' for r in rows))
    (OUT / f'cands-{name}-78h-ext.tags').write_text(''.join(r[2] + '\n' for r in rows))
    summary[name] = {'rows': len(rows), 'summary_line': [l for l in out.splitlines() if l.startswith('SUMMARY')]}
summary['scenarios'] = len(lines); summary['unique_states'] = len(states)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))
