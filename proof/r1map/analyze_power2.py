"""R1MAP task 3b-c: two-view synthetic true-seed hit rates (runs/sim2-*.jsonl) + real-map joint null
(realnull2-summary.json) -> search budget table.

Per-map rule (c, t) on the REAL maps (n = 6 components each): cent >= c and IoU >= t.  On a synthetic view with n
components the same rule is applied as cent >= ceil(c/6 * n), IoU >= t; only views with n >= 4 ("rich", as both real
maps are) are used, and the joint hit rate is conditional on BOTH views being rich.
Joint false rate on the real maps = product of the per-map marginals (independence verified at loose rules in
realnull2-summary.json, obs/product 0.8-1.2).  Conservative null = max over the U6 family variants measured.
Seed counts per window are C5's (NOTES.txt section 3, J=2, Delta 10-40 s, per sub-variant)."""
import glob, json, math, itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RULES = [(c, t) for c in (3, 4, 5) for t in (0.4, 0.5, 0.6, 0.7)]
SEEDS = {'PIT 24h': 1.0e9, 'PIT 78h': 1.8e9, 'PM 24h': 6.0e9, 'PM 78h': 2.0e10, 'HPET 24h': 1.0e11, 'HPET 78h': 3.2e11}
GRID = 90  # 5 K checkpoints x 6 air shares x 3 draw orders multiply false survivors (C5 NOTES)
BUDGET = 1e5


def passes(m, c, t):
    n = m['ncomp']
    return m['cent'] >= math.ceil(c / 6 * n - 1e-9) and m['iou'] >= t


def sim():
    out = {}
    for f in sorted(glob.glob(str(HERE / 'runs/sim2-*.jsonl'))):
        R = [json.loads(l) for l in open(f) if l.strip()]
        if not R: continue
        name = Path(f).stem[5:]
        r2 = [r for r in R if r['m2']['ncomp'] >= 4]; r1 = [r for r in R if r['m1']['ncomp'] >= 4]
        both = [r for r in R if r['m2']['ncomp'] >= 4 and r['m1']['ncomp'] >= 4]
        d = {'pairs': len(R), 'worlds': len({r['world_state'] for r in R}), 'rich_R2': len(r2), 'rich_R1': len(r1), 'rich_both': len(both),
             'worlds_rich_both': len({r['world_state'] for r in both}),
             'open_frac_R2_median': float(np.median([r['m2']['nopen'] / r['m2']['nobs'] for r in R])),
             'open_frac_R1_median': float(np.median([r['m1']['nopen'] / r['m1']['nobs'] for r in R])),
             'rules': {}}
        for (c2, t2), (c1, t1) in itertools.product(RULES, RULES):
            h2 = np.mean([passes(r['m2'], c2, t2) for r in r2]) if r2 else None
            h1 = np.mean([passes(r['m1'], c1, t1) for r in r1]) if r1 else None
            hj = np.mean([passes(r['m2'], c2, t2) and passes(r['m1'], c1, t1) for r in both]) if both else None
            d['rules'][f'R2[c{c2},t{t2}]&R1[c{c1},t{t1}]'] = {'hit_R2': h2, 'hit_R1': h1, 'hit_joint': hj}
        # independence of false passes in synthetic worlds (first pair only, loose rule)
        lp = [r['loose_pass'] for r in R if 'loose_pass' in r]
        if lp:
            nf = sum(r['nfalse'] for r in R if 'loose_pass' in r); a = sum(x[0] for x in lp); b = sum(x[1] for x in lp); ab = sum(x[2] for x in lp)
            d['synthetic_false_loose'] = {'nfalse': nf, 'R2': a / nf, 'R1': b / nf, 'joint': ab / nf, 'product': (a / nf) * (b / nf)}
        out[name] = d
    return out


def nulls():
    s = json.loads((HERE / 'realnull2-summary.json').read_text())
    fam = [k for k in s if 'U4s' not in k and 'random' not in k]
    res = {}
    for c, t in RULES:
        k = f'c{c},t{t}'
        res[k] = {'R2_max_U6fam': max(s[v]['marginal_R2'][k] for v in fam), 'R1_max_U6fam': max(s[v]['marginal_R1'][k] for v in fam),
                  'R2_U4s': s['realnull2-U4s']['marginal_R2'][k], 'R1_U4s': s['realnull2-U4s']['marginal_R1'][k],
                  'tested_min': min(s[v]['tested'] for v in fam)}
    return res


def floor_rate(r, n):  # a zero count is reported as its 95% upper bound 3/n
    return r if r > 0 else 3 / n


def main():
    S = sim(); N = nulls()
    pool = [k for k in S if k.endswith('_s11') and not k.startswith('U4s') and S[k]['rich_both'] >= 30]  # K256/nt12 give <5 doubly-rich pairs: unmeasurable
    table = []
    for (c2, t2), (c1, t1) in itertools.product(RULES, RULES):
        key = f'R2[c{c2},t{t2}]&R1[c{c1},t{t1}]'
        n2, n1 = N[f'c{c2},t{t2}'], N[f'c{c1},t{t1}']
        f2 = floor_rate(n2['R2_max_U6fam'], n2['tested_min']); f1 = floor_rate(n1['R1_max_U6fam'], n1['tested_min'])
        fj = f2 * f1
        hits = {k: S[k]['rules'][key]['hit_joint'] for k in S}
        hp = [S[k]['rules'][key]['hit_joint'] for k in pool if S[k]['rules'][key]['hit_joint'] is not None]
        row = {'rule': key, 'false_R2': f2, 'false_R1': f1, 'false_joint': fj, 'false_R2_is_upper_bound': n2['R2_max_U6fam'] == 0,
               'false_R1_is_upper_bound': n1['R1_max_U6fam'] == 0, 'hit_joint_U6fam_s11_mean': float(np.mean(hp)) if hp else None, 'hit_joint_by_run': hits,
               'false_survivors': {w: n * fj for w, n in SEEDS.items()}, 'false_survivors_full_grid': {w: n * fj * GRID for w, n in SEEDS.items()}}
        table.append(row)
    single = []
    for c, t in RULES:  # C5 single-map baseline for comparison
        n2 = N[f'c{c},t{t}']; f2 = floor_rate(n2['R2_max_U6fam'], n2['tested_min'])
        hp = [np.mean([passes(json.loads(l)['m2'], c, t) for l in open(HERE / f'runs/sim2-{k}.jsonl') if json.loads(l)['m2']['ncomp'] >= 4]) for k in pool]
        single.append({'rule': f'R2[c{c},t{t}] only', 'false': f2, 'hit_R2_U6fam_s11_mean': float(np.mean(hp)), 'false_survivors': {w: n * f2 for w, n in SEEDS.items()},
                       'false_survivors_full_grid': {w: n * f2 * GRID for w, n in SEEDS.items()}})
    out = {'synthetic': S, 'real_null_marginals': N, 'joint_table': table, 'single_map_baseline': single, 'seed_counts_per_subvariant': SEEDS,
           'grid_multiplier': GRID, 'regrowth_budget': BUDGET}
    (HERE / 'power2-summary.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    for k, d in S.items():
        print(k, {x: d[x] for x in d if x != 'rules'})
    print('\nOperating points (U6 family, sigma 1/1, joint hit | both maps rich), sorted by hit:')
    best = sorted([r for r in table if r['hit_joint_U6fam_s11_mean'] is not None], key=lambda r: -r['hit_joint_U6fam_s11_mean'])
    for r in best[:40]:
        fs = r['false_survivors']; fg = r['false_survivors_full_grid']
        print(f"{r['rule']:32s} hit {r['hit_joint_U6fam_s11_mean']:.2f}  false {r['false_joint']:.1e}  PM24 {fs['PM 24h']:.1e} HPET78 {fs['HPET 78h']:.1e}  grid: PM78 {fg['PM 78h']:.1e} HPET24 {fg['HPET 24h']:.1e} HPET78 {fg['HPET 78h']:.1e}")
    print('\nBest operating point per window (max pooled hit with false survivors <= 1e4 per sub-variant / <= 1e5 for the full grid):')
    pick = {}
    for w in SEEDS:
        for lab, fld, cap in (('sub-variant', 'false_survivors', 1e4), ('full grid', 'false_survivors_full_grid', 1e5)):
            ok = [r for r in best if r[fld][w] <= cap]
            if ok:
                r = ok[0]; pick[f'{w} | {lab}'] = {'rule': r['rule'], 'hit': r['hit_joint_U6fam_s11_mean'], 'false_survivors': r[fld][w], 'hit_by_run': r['hit_joint_by_run']}
                print(f"  {w:9s} {lab:12s} {r['rule']:32s} hit {r['hit_joint_U6fam_s11_mean']:.2f}  false survivors {r[fld][w]:.1e}")
            else: pick[f'{w} | {lab}'] = None; print(f'  {w:9s} {lab:12s} none')
    out['operating_points'] = pick; out['pool_runs'] = pool
    (HERE / 'power2-summary.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('\nSingle-map baseline:')
    for r in single:
        print(f"{r['rule']:22s} hit {r['hit_R2_U6fam_s11_mean']:.2f} false {r['false']:.1e} PM24 {r['false_survivors']['PM 24h']:.1e} HPET78 {r['false_survivors']['HPET 78h']:.1e}")


if __name__ == '__main__':
    main()
