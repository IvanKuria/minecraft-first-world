"""Closed-form power analysis of the JDK6 new-Random() fingerprint.

Under the null (wrong candidate, wrong N, or wrong k), X0 = back(S, 2N) is a
uniform 48-bit value, so a single (state, N, k) trial passes with
    p = G / 2^48,   G = number of distinct HotSpot nanoTime values in [tmin, tmax]
                      = (tmax - tmin) * min(f, 1e9) / 1e9          (for tmax < 2^48)
For tmax >= 2^48 the residues wrap and p -> min(1, ...).  This was checked
empirically on Java fixtures (fixture-validation.json) and on decoy-U controls
(scan-results.json): measured rates agree with p to within sampling error.
Expected false passes lambda = n_states * n_N * n_k * p.  The truth contributes
at most ONE pass, so a set-level signal needs lambda << 1.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M48 = 1 << 48
TMIN = 30 * 10**9
timers = {'PIT 1,193,182 Hz': 1193182, 'ACPI PM 3,579,545 Hz': 3579545, 'HPET 14,318,180 Hz': 14318180, 'TSC (>=1 GHz, dense)': 2394000000}
windows = {'1 h': 3600, '6 h': 6 * 3600, '24 h': 86400, '78.2 h (=2^48 ns)': M48 / 1e9, '7 d': 7 * 86400, '30 d': 30 * 86400}
sets = {'baseline175': 175, 'hidden 4,384': 4384, 'all 63,074': 63074}

def p_trial(f, up_s):
    span = up_s * 1e9 - TMIN
    g = span * min(f, 1e9) / 1e9
    return min(1.0, g / M48)

rows = []
for tn, f in timers.items():
    for wn, up in windows.items():
        p = p_trial(f, up)
        row = {'timer': tn, 'uptime_max': wn, 'p_per_trial': p}
        for sn, n in sets.items():
            for K in (1, 8):
                lam_per_N = n * K * p
                row[f'{sn}_K{K}_lambda_per_N'] = lam_per_N
                row[f'{sn}_K{K}_max_N_values_for_lambda_0.05'] = 0.05 / lam_per_N if lam_per_N else None
        rows.append(row)

# GEN-structured N: calls = K*sd + M*pd. If K,M are uncertain over ranges, how many distinct N?
nfam = {'exactly known N': 1, 'family A (0..16 calls)': 17, 'family B (registry +prefilm resets)': 18,
        'family C (round K x round M x sd,pd)': 1800,
        'K in 1..10000 with M,sd,pd fixed': 10000, 'M within +-1% of 1.34e8 (unstructured)': 2_684_354}

# Hidden-clock search-space reduction if the generator owned its own new Random() (GEN rng=int variants):
# Math.random's Random is then created at the first resetPos (N=0), so each (candidate, k) fixes t_math;
# the generator's seed is U + k_g + t_g with t_g = t_math - Delta, Delta = generation + init time.
def gen_seed_space(f, delta_lo, delta_hi, n_cand, n_k, n_kg=1):
    return n_cand * n_k * n_kg * (delta_hi - delta_lo) * min(f, 1e9)
reduction = []
for tn, f in timers.items():
    for (lo, hi) in [(10, 40), (5, 120)]:
        for sn, n in [('baseline175', 175), ('hidden 4,384', 4384)]:
            s = gen_seed_space(f, lo, hi, n, 8)
            reduction.append({'timer': tn, 'delta_s': [lo, hi], 'set': sn, 'k_values': 8,
                              'generator_seeds_to_test': s, 'log2': __import__('math').log2(s), 'vs_blind_2^48': s / M48})

out = {'p_formula': 'p = (tmax - tmin) * min(f,1e9)/1e9 / 2^48 per (state,N,k) trial; tmin=30 s',
       'table': rows, 'N_family_sizes': nfam, 'own_generator_random_seed_space': reduction,
       'empirical_check': json.loads((HERE / 'fixture-validation.json').read_text())['per_frequency']}
(HERE / 'power-analysis.json').write_text(json.dumps(out, indent=2) + '\n')

print('| timer | uptime <= | p per trial | 175, k=1..8: lambda per N | max #N for lambda<=0.05 (175, k known) |')
for r in rows:
    print(f"| {r['timer']} | {r['uptime_max']} | {r['p_per_trial']:.3g} | {r['baseline175_K8_lambda_per_N']:.3g} | {r['baseline175_K1_max_N_values_for_lambda_0.05']:.3g} |")
for r in reduction:
    print(r['timer'], r['delta_s'], r['set'], f"2^{r['log2']:.1f}", f"{r['vs_blind_2^48']:.2g}")
