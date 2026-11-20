"""ROWS lane (15 Oct 2026): widened, tagged, edge-scored candidate rows for the PC C5 kernel.

Inputs (unchanged, no new measurements):
  runtime/conditional-two-reset/hidden-reset-gap-results-ext.jsonl  441 hidden-reset scenarios (g1,g2 = 0..20), 76,619 unique
      states; each state = Math.random's internal state immediately before the OPENING resetPos x draw.
  fingerprint/build/fpscan  (JDK6 new Random() + HotSpot6 Windows QPC nanoTime grid test; wraps handled natively: it tests
      t = r + w*2^48 on the grid for every w with t in [tmin, tmax], so adding wraps re-tests, it does not relabel).
  caves/edge_rows.score  frozen world-edge model, same-height (delta=0) variant, called unchanged.

Model per row: fresh launch / clean model (calls before opening = 0) for all 76,619 ext states; additionally the 175 baseline
states (scenario 0,0) with calls 1..16 (family A, as in c5/make_inputs.py and r1map) so no existing row family is lost.
Timers: PM 3,579,545 Hz (the /usepmtimer case), PIT 1,193,182 Hz (low prior). HPET DROPPED (XP never used HPET for QPC).
k_m = 1..12 (was 1..8 in c5, 1..4 in hidden-ext). j is a kernel parameter (c5search J; k_g = k_m - j >= 1), so a row with
k_m supports j <= k_m - 1; tagged as jmax. Uptime windows: wrap 0 = [30 s, 2^48 ns), wrap 1 = [2^48, 2^49), wrap 2 = [2^49, 3*2^48).

Position convention (hidden-ext/make_rows.py): draw i -> coordinate(advance(S, 2i+1)); opening = draws 0,1;
R1 = draws 2+2g1, 3+2g1; R2 = draws 4+2(g1+g2), 5+2(g1+g2).  Verified below against link/lcg.reset_xz and r1map rows.
Row format (r12): state calls k q x2 z2 x1 z1.
"""
import json, struct, subprocess, sys, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
PROOF = OUT.parent
R = PROOF.parent
sys.path.insert(0, str(PROOF / 'caves')); sys.path.insert(0, str(PROOF / 'link'))
from edge_rows import score  # frozen model, not refit
from lcg import reset_xz

A, C, M = 0x5DEECE66D, 11, 1 << 48


def affine(steps):  # copied verbatim from runtime/benchmark_spawn_recovery.py (that module imports fpylll at load time)
    mul, add, a, c = 1, 0, A, C
    while steps:
        if steps & 1:
            mul, add = a * mul % M, (a * add + c) % M
        a, c = a * a % M, c * (a + 1) % M
        steps >>= 1
    return mul, add


def advance(state, steps):
    mul, add = affine(steps)
    return (mul * state + add) % M


def coordinate(first_state):
    second = (A * first_state + C) % M
    double = ((first_state >> 22) * 2**27 + (second >> 21)) / 2**53
    return struct.unpack('f', struct.pack('f', double))[0] * 256


def pos(S, i): return coordinate(advance(S, 2 * i + 1))


FP = PROOF / 'fingerprint/build/fpscan'
U = 8682522807148012
TIMERS = {'PM': 3579545, 'PIT': 1193182}
WRAPS = {0: (30_000_000_000, M - 1), 1: (M, 2 * M - 1), 2: (2 * M, 3 * M - 1)}
KMAX = 12
EDGE_GOOD = 0.4


def fpscan(states_file, nlist_file, f, tmin, tmax):
    out = subprocess.run([str(FP), str(states_file), str(nlist_file), '1', str(KMAX), str(f), str(tmin), str(tmax), str(U), '1000000000'],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    passes = [l.split() for l in out if l.startswith('PASS')]
    summ = [l for l in out if l.startswith('SUMMARY')][0]
    assert int(summ.split('passes=')[1].split()[0]) == len(passes), 'maxprint truncated'
    return [(int(i), int(N), int(k), int(t), int(q)) for _, i, N, k, t, q in passes], summ


def main():
    t0 = time.time()
    lines = [json.loads(l) for l in (R / 'runtime/conditional-two-reset/hidden-reset-gap-results-ext.jsonl').read_text().splitlines()]
    assert all(r['covered_all_boxes_without_output_cap'] for r in lines)
    scen = {}
    for r in lines:
        g = (r['hidden_resets_before_first_observed'], r['hidden_resets_between_observed'])
        for s in r['initial_internal_state_candidates']:
            scen.setdefault(s, []).append(g)
    states = sorted(scen)
    base175 = [int(l) for l in (PROOF / 'fingerprint/states-baseline175.txt').read_text().split()]
    zero = sorted(s for s in states if (0, 0) in scen[s])
    checks = {'scenarios': len(lines), 'unique_states': len(states), 'state_scenario_pairs': sum(len(v) for v in scen.values()),
              'scenario_00_equals_baseline175': zero == sorted(base175)}

    # --- convention check: pos() vs lcg.reset_xz (opening, R1, R2 consecutive) and vs r1map r12 rows
    bad = 0
    for s in base175:
        x0, z0, s1 = reset_xz(s); x1, z1, s2 = reset_xz(s1); x2, z2, _ = reset_xz(s2)
        bad += (x0, z0, x1, z1, x2, z2) != tuple(pos(s, i) for i in range(6))
    r1 = [l.split() for l in (PROOF / 'r1map/cands-PM-78h-r12.txt').read_text().splitlines() if l.strip()]
    bad_r1 = sum((float(p[4]), float(p[5]), float(p[6]), float(p[7])) != (pos(int(p[0]), 4), pos(int(p[0]), 5), pos(int(p[0]), 2), pos(int(p[0]), 3)) for p in r1)
    checks.update({'pos_vs_reset_xz_mismatches_175': bad, 'pos_vs_r1map_PM78h_mismatches': bad_r1, 'r1map_rows_checked': len(r1)})
    # ext convention vs the scenario's own coordinate_indices
    ci_bad = 0
    for r in lines:
        g1, g2 = r['hidden_resets_before_first_observed'], r['hidden_resets_between_observed']
        ci_bad += r['coordinate_indices'] != [0, 1, 2 + 2 * g1, 3 + 2 * g1, 4 + 2 * (g1 + g2), 5 + 2 * (g1 + g2)]
    checks['coordinate_index_mismatches'] = ci_bad
    print(json.dumps(checks), flush=True)

    # --- frozen edge score of every unique state's opening position
    ecache = OUT / 'edge-all-states.json'
    if ecache.exists():
        E = {int(k): v for k, v in json.loads(ecache.read_text()).items()}
    else:
        E = {}
        for n, s in enumerate(states):
            x0, z0 = pos(s, 0), pos(s, 1)
            sc = score([x0, z0], deltas=(0,))[0]
            E[s] = [x0, z0, None if sc is None else sc['train_rmse_px'], None if sc is None else sc['heldout_rmse_px']]
            if n % 10000 == 0: print('edge', n, round(time.time() - t0), flush=True)
        ecache.write_text(json.dumps({str(k): v for k, v in E.items()}))
    # cross-check against the CAVES lane's own scores (4,385 pairs, g <= 4)
    cav = json.loads((PROOF / 'caves/edge-hidden-scenarios.json').read_text())
    dif = [abs(E[c['state_before_opening']][2] - c['rmse_same_height']) for c in cav
           if c['rmse_same_height'] is not None and c['state_before_opening'] in E and E[c['state_before_opening']][2] is not None]
    checks['edge_crosscheck_vs_caves'] = {'pairs_compared': len(dif), 'max_abs_diff_px': max(dif) if dif else None,
                                          'caves_pairs_with_state_absent_from_ext': sum(c['state_before_opening'] not in E for c in cav)}
    print('edge crosscheck', checks['edge_crosscheck_vs_caves'], flush=True)

    def rm(s): v = E[s][2]; return float('inf') if v is None else v

    # --- fingerprint scans
    (OUT / 'states-ext.txt').write_text('\n'.join(map(str, states)) + '\n')
    (OUT / 'states-175.txt').write_text('\n'.join(map(str, base175)) + '\n')
    (OUT / 'nlist0.txt').write_text('0\n'); (OUT / 'nlist1-16.txt').write_text('1 16\n')
    summary = {'checks': checks, 'timers': {}}
    for tname, f in TIMERS.items():
        rows, ts = {}, {}
        stats = {'fpscan': {}, 'pass_states_by_wrap': {}}
        for w, (tmin, tmax) in WRAPS.items():
            for fam, sf, nl, sl in (('ext_calls0', OUT / 'states-ext.txt', OUT / 'nlist0.txt', states),
                                    ('175_calls1-16', OUT / 'states-175.txt', OUT / 'nlist1-16.txt', base175)):
                P, summ = fpscan(sf, nl, f, tmin, tmax)
                stats['fpscan'][f'w{w}-{fam}'] = summ
                for i, N, k, t, q in P:
                    assert t >> 48 == w and tmin <= t <= tmax
                    s = sl[i]
                    stats['pass_states_by_wrap'].setdefault(w, set()).add((s, N, k))
                    gl = scen[s] if N == 0 else [(0, 0)]
                    for g1, g2 in gl:
                        x1, z1 = pos(s, 2 + 2 * g1), pos(s, 3 + 2 * g1)
                        x2, z2 = pos(s, 4 + 2 * (g1 + g2)), pos(s, 5 + 2 * (g1 + g2))
                        e = rm(s)
                        # even calls before the opening = calls/2 hidden resets at the opening press; odd calls = other Math use
                        hid = g1 + g2 + (N // 2 if N % 2 == 0 else 0)
                        kord = 0 if k in (2, 3) else (2 if k == 1 else 1)   # k=1 last: kernel needs k_g = k-j >= 1 with j >= 1
                        key = (e >= EDGE_GOOD, g1 > 3, N % 2, hid, kord, w, k, N, e)
                        row = f'{s} {N} {k} {q} {x2!r} {z2!r} {x1!r} {z1!r}'
                        tier = ('E' if e < EDGE_GOOD else 'e') + ('H' if g1 <= 3 else 'h') + ('K' if k in (2, 3) else 'k')
                        es = 'none' if e == float('inf') else f'{e:.4f}'
                        tag = (f'{s} timer={tname} calls={N} k={k} jmax={k - 1} wrap={w} t_ns={t} g={g1}_{g2} hidden_total={hid if N % 2 == 0 else "odd"} '
                               f'open={E[s][0]:.3f},{E[s][1]:.3f} edge={es} tier={tier}')
                        assert row not in rows
                        rows[row] = (key, tag)
        order = sorted(rows, key=lambda r: rows[r][0])
        (OUT / f'cands-{tname}-r12.txt').write_text(''.join(r + '\n' for r in order))
        (OUT / f'cands-{tname}-r12.tags').write_text(''.join(rows[r][1] + '\n' for r in order))
        # --- extension accounting (rows = state/scenario pairs x passing (calls,k,wrap))
        def cnt(pred): return sum(1 for r in rows if pred(rows[r][1]))
        def tg(t, key): return [x.split('=')[1] for x in t.split() if x.startswith(key + '=')][0]
        def ok(t, kmax=8, wmax=0, gmax=20, calls0=None):
            k, w, g1, g2, N = int(tg(t, 'k')), int(tg(t, 'wrap')), *map(int, tg(t, 'g').split('_')), int(tg(t, 'calls'))
            return k <= kmax and w <= wmax and max(g1, g2) <= gmax and (calls0 is None or (N == 0) == calls0)
        acc = {
            'total_rows': len(rows),
            'reproduce_hidden_ext_(calls0,k1-4,w0)': cnt(lambda t: ok(t, 4, 0, calls0=True)),
            'calls0_k1-8_w0': cnt(lambda t: ok(t, 8, 0, calls0=True)),
            'calls0_k1-12_w0': cnt(lambda t: ok(t, 12, 0, calls0=True)),
            'calls0_k1-12_w0-1': cnt(lambda t: ok(t, 12, 1, calls0=True)),
            'calls0_k1-12_w0-2': cnt(lambda t: ok(t, 12, 2, calls0=True)),
            'added_by_k9-12_(w0)': cnt(lambda t: ok(t, 12, 0, calls0=True)) - cnt(lambda t: ok(t, 8, 0, calls0=True)),
            'added_by_wrap1_(k1-12)': cnt(lambda t: ok(t, 12, 1, calls0=True)) - cnt(lambda t: ok(t, 12, 0, calls0=True)),
            'added_by_wrap2_(k1-12)': cnt(lambda t: ok(t, 12, 2, calls0=True)) - cnt(lambda t: ok(t, 12, 1, calls0=True)),
            '175_calls1-16_rows_all': cnt(lambda t: ok(t, 12, 2, calls0=False)),
            '175_calls1-16_(k1-8,w0)_reproduce_c5_minus_calls0': cnt(lambda t: ok(t, 8, 0, calls0=False)),
            'rows_k1_(kernel skips: k_g<1 for every j>=1)': cnt(lambda t: tg(t, 'k') == '1'),
            'rows_by_k': {k: cnt(lambda t, k=k: tg(t, 'k') == str(k)) for k in range(1, KMAX + 1)},
            'rows_by_wrap': {w: cnt(lambda t, w=w: tg(t, 'wrap') == str(w)) for w in WRAPS},
            'rows_by_tier': {}, 'edge_counts': {},
            'unique_states_in_rows': len({r.split()[0] for r in rows}),
        }
        for r in rows:
            ti = tg(rows[r][1], 'tier'); acc['rows_by_tier'][ti] = acc['rows_by_tier'].get(ti, 0) + 1
        for thr in (0.3, 0.35, 0.4, 0.5):
            acc['edge_counts'][f'<{thr}'] = cnt(lambda t, thr=thr: tg(t, 'edge') != 'none' and float(tg(t, 'edge')) < thr)
        acc['top_tier_EHK_rows'] = acc['rows_by_tier'].get('EHK', 0)
        acc['top_tier_EHK_k2-3_calls0_w0'] = cnt(lambda t: tg(t, 'tier') == 'EHK' and tg(t, 'wrap') == '0' and tg(t, 'calls') == '0')
        # wrap re-test: does a (state,calls,k) that passes at wrap 0 also pass at wrap 1? (independent grid draws expected)
        pw = stats['pass_states_by_wrap']
        acc['wrap_retest'] = {'w0_pass_triples': len(pw.get(0, ())), 'w1_pass_triples': len(pw.get(1, ())), 'w2_pass_triples': len(pw.get(2, ())),
                              'w0_and_w1_same_triple': len(pw.get(0, set()) & pw.get(1, set())),
                              'w0_and_w2_same_triple': len(pw.get(0, set()) & pw.get(2, set()))}
        acc['fpscan'] = stats['fpscan']
        acc['first_20_rows'] = [rows[r][1] for r in order[:20]]
        summary['timers'][tname] = acc
        print(tname, json.dumps({k: v for k, v in acc.items() if k not in ('fpscan', 'first_20_rows')}), flush=True)

    # --- TSC (no grid): footage states ranked by edge
    b = sorted(base175, key=rm)
    (OUT / 'tsc-baseline175.txt').write_text('# state opening_x opening_z edge_train_px edge_heldout_px  (calls 0, scenario 0,0; no fingerprint filter)\n' +
                                             ''.join(f'{s} {E[s][0]!r} {E[s][1]!r} {E[s][2]} {E[s][3]}\n' for s in b))
    tsc = {'baseline175': {f'<{t}': sum(rm(s) < t for s in base175) for t in (0.3, 0.4, 0.5)}}
    pairs = [(s, g) for s in states for g in scen[s]]
    tsc['ext_pairs_all'] = {f'<{t}': sum(rm(s) < t for s, _ in pairs) for t in (0.3, 0.4, 0.5)}
    tsc['ext_pairs_g1le3'] = {f'<{t}': sum(rm(s) < t and g[0] <= 3 for s, g in pairs) for t in (0.3, 0.4, 0.5)}
    tsc['ext_unique_states'] = {f'<{t}': sum(rm(s) < t for s in states) for t in (0.3, 0.4, 0.5)}
    tsc['ext_pairs_total'] = len(pairs); tsc['ext_pairs_g1le3_total'] = sum(g[0] <= 3 for _, g in pairs)
    # TSC cost: t_m is exact per (state,k) (no grid needed); generator t_g = t_m - Delta, Delta 10-40 s at 1 ns -> 3e10 seeds per (state,k,j)
    per = 3e10
    tsc['seeds_per_state_k_j'] = per
    tsc['cost_examples_seeds'] = {
        'baseline175<0.4 x k{2,3} x j{1,2}': tsc['baseline175']['<0.4'] * 2 * 2 * per,
        'baseline175<0.4 x k1-12 x j1-10 (k_g>=1)': tsc['baseline175']['<0.4'] * sum(min(10, k - 1) for k in range(1, 13)) * per,
        'ext_pairs_g1le3<0.4 x k{2,3} x j{1,2}': len({s for s, g in pairs if rm(s) < 0.4 and g[0] <= 3}) * 4 * per,
    }
    tsc['gpu_days_at_8.07e6_seeds_per_s_K4096'] = {k: v / 8.07e6 / 86400 for k, v in tsc['cost_examples_seeds'].items()}
    tsc['top20_baseline'] = [[s, round(E[s][0], 2), round(E[s][1], 2), E[s][2]] for s in b[:20]]
    summary['tsc'] = tsc
    summary['seconds'] = round(time.time() - t0, 1)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(tsc, indent=1))


if __name__ == '__main__':
    main()
