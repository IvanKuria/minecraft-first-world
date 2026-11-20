"""Pooled synthetic power: true-seed hit rate vs per-trial false rate, split by number of observed hole components.
Rules: cent >= ceil(2n/3) (n = observed components >= 4 columns) and IoU >= tau.  'rich' = n >= 4 (the real
second-reset map has n = 6)."""
import json, glob, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
groups = {'U6s_K1024': ['A_', 'A2_', 'B_', 'B2_', 'C_', 'C2_'], 'U6s_K256': ['D_', 'D2_'], 'U6s_nt12': ['E_'], 'U4s': ['H_'],
          'U6s_K4096': ['G_'], 'U6r': ['F_']}
out = {}
for g, pre in groups.items():
    rows = []
    for p in pre:
        for f in glob.glob(str(HERE / f'runs/{p}*.jsonl')): rows += [json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    for part, sel in (('all', lambda r: r['ncomp'] >= 1), ('rich_n>=4', lambda r: r['ncomp'] >= 4), ('none', lambda r: r['ncomp'] == 0)):
        R = [r for r in rows if sel(r)]
        if part == 'none': out[f'{g}|no_hole_fraction'] = round(len(R) / len(rows), 3); continue
        for tau in (0.4, 0.5, 0.6):
            hit, rates, N = 0, [], 0
            for r in R:
                n = r['ncomp']; nd = math.ceil(2 * n / 3); tb = int(tau * 200)
                hit += (r['true_centHit'] >= nd and r['true']['iou'] >= tau)
                M = np.array(r['false_IoU_cent']); rates.append(float(M[tb:, nd:].sum()) / r['nfalse']); N += r['nfalse']
            if R:
                out[f'{g}|{part}|tau{tau}'] = {'worlds': len(R), 'sigmas': sorted({r['sigma'] for r in R}), 'true_hit': round(hit / len(R), 3),
                    'false_mean': float(np.mean(rates)), 'false_median': float(np.median(rates)), 'resolution': 1 / (N / len(R))}
(HERE / 'power-pooled.json').write_text(json.dumps(out, indent=1) + '\n')
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else f"worlds {v['worlds']} hit {v['true_hit']} false mean {v['false_mean']:.1e} median {v['false_median']:.1e} (res {v['resolution']:.0e})")
