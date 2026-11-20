"""Build C5 search inputs from existing lane outputs (no new measurements).

1. Candidate rows (cands-<timer>-<window>.txt): the fingerprint grid test (fpscan, family A = 0..16 Math.random calls
   before the opening x draw, k_m = 1..8) on the 175 baseline opening states.  Under H3 Math.random's Random was
   created by JDK6 new Random() at the first resetPos, so (state, calls, k_m) is admissible ONLY if its implied
   nanoTime lies on the QPC grid of the timer; each admissible row fixes q_m exactly.  The second-reset spawn
   (x2, z2) comes from the same state: opening reset, first observed reset, second reset are consecutive
   (zero-gap baseline assumption, same as the 175 set).
2. footage.obs: the second-reset surface map of obs_footage.py, in c5search format, same post-processing rule as
   c5sim build_obs (open components >= 4 columns; closed columns 8-adjacent to any open column dropped).
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
sys.path.insert(0, str(PROOF / 'link'))
from lcg import reset_xz, back  # noqa: E402

FP = PROOF / 'fingerprint'
TIMERS = {'PIT': 1193182, 'PM': 3579545, 'HPET': 14318180}
WINDOWS = {'24h': 86400 * 10**9, '78h': (1 << 48) - 1, '1h': 3600 * 10**9}
U = 8682522807148012


def cands(timer, window):
    f = TIMERS[timer]
    r = subprocess.run([str(FP / 'build/fpscan'), str(FP / 'states-baseline175.txt'), str(FP / 'nlist-A_math_random_first_used_at_spawn.txt'),
                        '1', '8', str(f), str(30 * 10**9), str(WINDOWS[window]), str(U), '1000000'], capture_output=True, text=True, check=True)
    states = [int(l) for l in (FP / 'states-baseline175.txt').read_text().split()]
    rows, chk = [], []
    for l in r.stdout.splitlines():
        if not l.startswith('PASS'): continue
        _, idx, calls, k, nanos, q = l.split()
        s_open = states[int(idx)]
        x0, z0, s1 = reset_xz(s_open)
        x1, z1, s2 = reset_xz(s1)
        x2, z2, _ = reset_xz(s2)
        chk.append(195 <= x1 <= 201 and 222 <= x2 <= 225 and 113 <= z2 <= 135)
        rows.append(f'{s_open} {calls} {k} {q} {x2!r} {z2!r}')
    (HERE / f'cands-{timer}-{window}.txt').write_text('\n'.join(rows) + '\n')
    return len(rows), all(chk), r.stdout.splitlines()[-1]


def obs():
    d = json.loads((HERE / 'footage-obs.json').read_text())
    lab = {(x, z): v for x, z, v in d['labels']}
    opn = {k for k, v in lab.items() if v == 1}
    comps, seen = [], set()
    for k in sorted(opn):
        if k in seen: continue
        st, cc = [k], []; seen.add(k)
        while st:
            x, z = st.pop(); cc.append((x, z))
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (x + dx, z + dz)
                if q in opn and q not in seen: seen.add(q); st.append(q)
        comps.append(cc)
    big = [c for c in comps if len(c) >= 4]
    cid = {p: i for i, c in enumerate(big) for p in c}
    lines = ['# second-reset surface map (obs_footage.py), relative to floor(x2), floor(z2); O dx dz lab comp | C cdx cdz size']
    n_open = n_closed = 0
    for (x, z), v in sorted(lab.items()):
        if v == 1:
            if (x, z) in cid: lines.append(f'O {x} {z} 1 {cid[(x, z)]}'); n_open += 1
        else:
            if any((x + dx, z + dz) in opn for dx in (-1, 0, 1) for dz in (-1, 0, 1)): continue
            lines.append(f'O {x} {z} 0 -1'); n_closed += 1
    for c in big:
        lines.append(f'C {sum(p[0] + .5 for p in c) / len(c):.3f} {sum(p[1] + .5 for p in c) / len(c):.3f} {len(c)}')
    (HERE / 'footage.obs').write_text('\n'.join(lines) + '\n')
    return {'open': n_open, 'closed': n_closed, 'components': [len(c) for c in big]}


if __name__ == '__main__':
    out = {'obs': obs()}
    for t in TIMERS:
        for w in ('1h', '24h', '78h'):
            n, ok, summ = cands(t, w)
            out[f'{t}-{w}'] = {'rows': n, 'reset_ranges_consistent': ok, 'fpscan': summ}
    (HERE / 'inputs-summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
