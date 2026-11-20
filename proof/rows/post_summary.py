"""Run after make_rows.py: adds PC search cost per row block and replaces make_rows.py's TSC cost examples
(which over-counted (k=2, j=2), a k_g = 0 case the kernel skips). Idempotent; reads rows/*.tags and summary.json."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
d = json.loads((H / 'summary.json').read_text())
RATE = 8.07e6  # seeds/s, K=4096, pc-c5/README.md (measured under load)
F = {'PM': 3579545, 'PIT': 1193182}
out = {}
for T, f in F.items():
    per = 30 * f  # q window per (row, j): Delta 10-40 s
    res = {'seeds_per_row_j': per}
    blocks = [('block_E_g1le3_even', lambda v: v['edge'] != 'none' and float(v['edge']) < 0.4 and int(v['g'].split('_')[0]) <= 3 and int(v['calls']) % 2 == 0),
              ('edge_lt_0.4_all', lambda v: v['edge'] != 'none' and float(v['edge']) < 0.4), ('all', lambda v: True)]
    for name, pred in blocks:
        n = s2 = s10 = 0
        for l in (H / f'cands-{T}-r12.tags').read_text().splitlines():
            v = dict(x.split('=', 1) for x in l.split()[1:])
            if not pred(v): continue
            k = int(v['k']); n += 1; s2 += max(0, min(2, k - 1)); s10 += max(0, min(10, k - 1))
        res[name] = {'rows': n, 'row_j_pairs_J2': s2, 'row_j_pairs_J10': s10, 'seeds_J2': s2 * per, 'seeds_J10': s10 * per,
                     'gpu_hours_J2_K4096': s2 * per / RATE / 3600, 'gpu_hours_J10_K4096': s10 * per / RATE / 3600}
    out[T] = res
d['pc_cost'] = out
d['pc_cost_note'] = 'q window Delta 10-40 s per (row, j) as c5search DMIN/DMAX; J capped at k-1; throughput 8.07e6 seeds/s K=4096'

E = {int(k): v for k, v in json.loads((H / 'edge-all-states.json').read_text()).items()}
L = [json.loads(l) for l in (H.parents[1] / 'runtime/conditional-two-reset/hidden-reset-gap-results-ext.jsonl').read_text().splitlines()]
S = {s for r in L if r['hidden_resets_before_first_observed'] <= 3 for s in r['initial_internal_state_candidates']
     if E[s][2] is not None and E[s][2] < 0.4}
t = d['tsc']; per = 3e10; n175 = t['baseline175']['<0.4']
kj23 = 3                                          # (k,j) = (2,1), (3,1), (3,2)
kj_all = sum(min(10, k - 1) for k in range(1, 13))  # 65
t['ext_g1le3_lt0.4_unique_states'] = len(S)
t['seeds_per_state_k_j'] = per
t['cost_examples_seeds'] = {
    'baseline175<0.4 x (k,j) in {(2,1),(3,1),(3,2)}': n175 * kj23 * per,
    'baseline175<0.4 x k1-12 x j1-min(10,k-1) [65 pairs]': n175 * kj_all * per,
    'ext g1<=3 <0.4 unique states x {(2,1),(3,1),(3,2)}': len(S) * kj23 * per,
    'ext g1<=3 <0.4 unique states x 65 (k,j) pairs': len(S) * kj_all * per}
t['gpu_days_at_8.07e6_seeds_per_s_K4096'] = {k: v / RATE / 86400 for k, v in t['cost_examples_seeds'].items()}
(H / 'summary.json').write_text(json.dumps(d, indent=1))
print(json.dumps({'pc_cost': out, 'tsc_cost': t['gpu_days_at_8.07e6_seeds_per_s_K4096']}, indent=1))
