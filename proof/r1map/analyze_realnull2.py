"""R1MAP task 3a: null tail of the C5 early test for random generator seeds against BOTH real maps.
Input runs/realnull2-*.txt (c5search, C5_ANY=1: every seed with cent>=3 & IoU>=.30 on either map, both maps' values).
For each per-map rule (c, t) we report the marginal false rate on each map, the observed joint rate, and the product
of marginals.  Independence (joint ~= product) is checked at loose rules where both are measurable; at strict rules
the joint is estimated by the product (the two maps lie ~85 blocks apart, so they read disjoint seed subsets)."""
import glob, gzip, json, itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS, TS = (3, 4, 5, 6), (0.3, 0.4, 0.5, 0.6, 0.7)
res = {}
for f in sorted(glob.glob(str(HERE / 'runs/realnull2-*.txt.gz'))):
    lines = gzip.open(f, 'rt').read().splitlines()
    if not lines or not lines[-1].startswith('SUMMARY'): continue
    tested = int(lines[-1].split('tested=')[1].split()[0])
    P = []
    for l in lines:
        if not l.startswith('PASS'): continue
        p = l.split(); P.append((int(p[5]), float(p[6]), int(p[7]), float(p[8])))
    m2 = {f'c{c},t{t}': sum(1 for a, b, _, _ in P if a >= c and b >= t) / tested for c in CS for t in TS}
    m1 = {f'c{c},t{t}': sum(1 for _, _, a, b in P if a >= c and b >= t) / tested for c in CS for t in TS}
    joint = {}
    for (c2, t2), (c1, t1) in itertools.product(itertools.product(CS, TS), repeat=2):
        n = sum(1 for a, b, x, y in P if a >= c2 and b >= t2 and x >= c1 and y >= t1)
        prod = m2[f'c{c2},t{t2}'] * m1[f'c{c1},t{t1}']
        joint[f'R2[c{c2},t{t2}] & R1[c{c1},t{t1}]'] = {'n': n, 'rate': n / tested, 'product': prod,
                                                       'ratio_obs_over_product': (n / tested) / prod if prod > 0 and n >= 10 else None}
    res[Path(f).name[:-7]] = {'tested': tested, 'max_iou_R2': max((b for _, b, _, _ in P), default=None), 'max_iou_R1': max((y for _, _, _, y in P), default=None),
                         'marginal_R2': m2, 'marginal_R1': m1, 'joint': joint}
(HERE / 'realnull2-summary.json').write_text(json.dumps(res, indent=1) + '\n')
for k, v in res.items():
    print(f"== {k} tested {v['tested']:.1e}  max IoU R2 {v['max_iou_R2']}  R1 {v['max_iou_R1']}")
    for c in (3, 4, 5):
        print('  R2 ', '  '.join(f"c{c} t{t}: {v['marginal_R2'][f'c{c},t{t}']:.1e}" for t in TS))
        print('  R1 ', '  '.join(f"c{c} t{t}: {v['marginal_R1'][f'c{c},t{t}']:.1e}" for t in TS))
    chk = [(kk, j) for kk, j in v['joint'].items() if j['ratio_obs_over_product'] is not None]
    for kk, j in sorted(chk, key=lambda q: -q[1]['n'])[:6]:
        print(f"   indep {kk}: n {j['n']} obs {j['rate']:.2e} product {j['product']:.2e} ratio {j['ratio_obs_over_product']:.2f}")
