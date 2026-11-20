"""Summarise c5sim batch runs: hit rate of the true seed list vs false-pass rate of random seed lists, per rule.
Rule = (centroid rule, IoU threshold): pass iff centHit >= need(ncomp) and IoU >= tau.
False rates are per-world rates averaged over worlds (each world = one synthetic footage scenario).
Upper bounds use 3/N when nothing passes (95%)."""
import json, sys, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RULES = [('any', lambda n: 0), ('allm1', lambda n: max(1, n - 1) if n else 0), ('all', lambda n: n)]
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def summarise(rows):
    out = []
    usable = [r for r in rows if r['ncomp'] > 0]
    for rn, need in RULES:
        for tau in TAUS:
            hits, rates, zeros, N = 0, [], 0, 0
            for r in usable:
                n = r['ncomp']; nd = need(n); tb = int(np.floor(tau * 200))
                t = r['true']; th = (r.get('true_centHit', t['compHit']) >= nd) and t['iou'] >= tau
                hits += bool(th)
                M = np.array(r['false_IoU_cent'])  # [iou_bin][cent]
                fp = M[tb:, nd:].sum() if nd <= 32 else 0
                rates.append(float(fp) / r['nfalse']); N += r['nfalse']
                zeros += int(fp == 0)
            mean = float(np.mean(rates)) if rates else None
            out.append({'rule': rn, 'tau': tau, 'worlds': len(usable), 'true_hit': hits / len(usable) if usable else None,
                        'false_rate_mean': mean, 'false_rate_median': float(np.median(rates)) if rates else None,
                        'worlds_with_zero_false': zeros, 'false_rate_upper95_if_zero': 3 / N if N else None})
    return out


def main():
    files = sorted(glob.glob(str(HERE / 'runs/*.jsonl')))
    res = {}
    for f in files:
        rows = load(f)
        if not rows or 'false_IoU_cent' not in rows[0]: continue
        name = Path(f).stem
        info = {'n_worlds': len(rows), 'worlds_with_no_hole_in_view': sum(r['ncomp'] == 0 for r in rows),
                'ncomp_median': float(np.median([r['ncomp'] for r in rows])), 'nopen_median': float(np.median([r['nopen'] for r in rows])),
                'nobs_median': float(np.median([r['nobs'] for r in rows])),
                'true_iou': [round(r['true']['iou'], 3) for r in rows], 'true_cent_over_ncomp': [f"{r.get('true_centHit')}/{r['ncomp']}" for r in rows],
                'nearest_air_seed_to_hole_centroid_dist_median': float(np.median([c[1] for r in rows for c in r['true_comp']])) if any(r['true_comp'] for r in rows) else None,
                'nearest_air_seed_depth_median_y': float(np.median([c[2] for r in rows for c in r['true_comp']])) if any(r['true_comp'] for r in rows) else None,
                'rules': summarise(rows)}
        res[name] = info
    (HERE / 'power-summary.json').write_text(json.dumps(res, indent=1) + '\n')
    for name, info in res.items():
        print(f"== {name}: worlds {info['n_worlds']} (no hole {info['worlds_with_no_hole_in_view']}), ncomp med {info['ncomp_median']}, open med {info['nopen_median']}, obs med {info['nobs_median']}")
        print('   true IoU', info['true_iou']); print('   cent', info['true_cent_over_ncomp'])
        print(f"   nearest air seed to hole centroid: median {info['nearest_air_seed_to_hole_centroid_dist_median']} blocks, depth y {info['nearest_air_seed_depth_median_y']}")
        for r in info['rules']:
            if r['tau'] in (0.4, 0.5, 0.6, 0.7):
                print(f"   {r['rule']:5s} tau {r['tau']}: hit {r['true_hit']:.2f}  false mean {r['false_rate_mean']:.2e} median {r['false_rate_median']:.2e} (zero-false worlds {r['worlds_with_zero_false']}, 3/N {r['false_rate_upper95_if_zero']:.1e})")


if __name__ == '__main__':
    main()
