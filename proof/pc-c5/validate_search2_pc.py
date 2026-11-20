"""R1MAP task 4: planted-seed validation of the TWO-MAP c5search (synthetic ground truth, GEN's generator).

Same protocol as c5/validate_search.py, with a second map: plant a JDK6 clock seed, grow with `gen run`, pick two plain
spawns >= 60 blocks apart (second reset x2,z2 and first reset x1,z1), observe both footage-style with r1sim obs2
(second-reset poses / first-reset poses), then run c5search with BOTH maps over a q_g window around the planted value.
Checks: (1) gen's world equals r1sim's world (state_out), (2) the planted q_g is reported (needs its seed list to pass
both maps), (3) false passes per tested seed with two maps vs with the second-reset map alone (same window).
usage: validate_search2.py <variant> <trials> <half_window> <cmin_rule> <imin> <threads>
"""
import json, subprocess, sys, random, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE / 'build'
GEN = HERE.parent / 'gen/gen.exe'
TMP = HERE / 'fixtures2'; TMP.mkdir(parents=True, exist_ok=True)
GPU_MISMATCH = []
U0 = 8682522807148012; MULT = 0x5DEECE66D; MASK = (1 << 48) - 1
F = 3579545


def hs_nanos(q, f=F):
    return int((q / f) * 1e9)


def plain_spawns(blocks, rng):
    b = blocks.reshape(64, 256, 256)
    top = np.where(b != 0, np.arange(64)[:, None, None], -1).max(axis=0)
    def one():
        for _ in range(100000):
            x = rng.uniform(40, 216); z = rng.uniform(40, 216)
            xs = range(int(math.floor(x - .3)), int(math.floor(x + .3)) + 1); zs = range(int(math.floor(z - .3)), int(math.floor(z + .3)) + 1)
            if max(top[zz, xx] for xx in xs for zz in zs) == 42: return x, z
        raise RuntimeError('no plain spawn')
    x2, z2 = one()
    for _ in range(1000):
        x1, z1 = one()
        if math.hypot(x1 - x2, z1 - z2) >= 60: return x2, z2, x1, z1
    raise RuntimeError('no second spawn')


def cmin_of(rule, n):
    return n if rule == 'all' else max(0, n - 1) if rule == 'allm1' else int(rule)


def search(variant, cands, obs, cmin, imin, threads, lo, hi):
    args = [variant, str(cands), obs, str(F), '10', '40', '1', cmin, imin, str(threads), str(lo), str(hi)]
    r = subprocess.run([str(B / 'c5search2.exe'), *args], capture_output=True, text=True, check=True).stdout.splitlines()
    g = subprocess.run([str(B / 'c5gpu3.exe'), *args], capture_output=True, text=True, check=True).stdout.splitlines()
    same = sorted(l for l in r if l.startswith('PASS')) == [l for l in g if l.startswith('PASS')] and r[-1] == g[-1]
    print(json.dumps({'gpu_vs_cpu': 'IDENTICAL' if same else 'DIFF', 'maps': obs.count(',') + 1, 'cpu_pass': sum(l.startswith('PASS') for l in r), 'summary': r[-1]}), flush=True)
    if not same: GPU_MISMATCH.append((obs, r[-1], g[-1]))
    return [l.split() for l in r if l.startswith('PASS')], r[-1]


def main():
    variant, trials, half, rule, imin, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
    rng = random.Random(20261016)
    out = []
    t = 0
    while len(out) < trials:
        k_m = rng.randint(2, 8); q_m = rng.randint(int(600 * F), int(86400 * F)); delta = rng.uniform(10, 40)
        q_g = q_m - int(delta * F); k_g = k_m - 1
        G0 = ((U0 + k_g + hs_nanos(q_g)) ^ MULT) & MASK
        blk = TMP / 'w.blocks'
        g = json.loads(subprocess.run([str(GEN), 'run', variant, str(G0), str(blk)], capture_output=True, text=True, check=True).stdout)
        x2, z2, x1, z1 = plain_spawns(np.fromfile(blk, np.uint8), rng)
        o2, o1 = TMP / f'w{len(out)}-{t}-2.obs', TMP / f'w{len(out)}-{t}-1.obs'
        o = json.loads(subprocess.run([str(B / 'r1sim.exe'), 'obs2', variant, str(G0), repr(x2), repr(z2), repr(x1), repr(z1), str(o2), str(o1)],
                                      capture_output=True, text=True, check=True).stdout)
        t += 1
        if o['ncomp2'] < 2 or o['ncomp1'] < 2:  # a map without holes is no test (C5 NOTES); draw another world
            print(json.dumps({'skipped_world': t, 'ncomp2': o['ncomp2'], 'ncomp1': o['ncomp1']}), flush=True); continue
        c2, c1 = cmin_of(rule, o['ncomp2']), cmin_of(rule, o['ncomp1'])
        cands = TMP / 'w.cands'
        cands.write_text(f'0 0 {k_m} {q_m} {x2!r} {z2!r} {x1!r} {z1!r}\n')
        both, sb = search(variant, cands, f'{o2},{o1}', f'{c2},{c1}', imin, threads, q_g - half, q_g + half)
        only2, s2 = search(variant, cands, str(o2), str(c2), imin, threads, q_g - half, q_g + half)
        only1, s1 = search(variant, cands, str(o1), str(c1), imin, threads, q_g - half, q_g + half)
        pl = [p for p in both if int(p[3]) == q_g]
        # exactness: the two-map PASS set must equal the intersection of the single-map PASS sets, with identical values
        d2 = {int(p[3]): tuple(p[5:7]) for p in only2}; d1 = {int(p[3]): tuple(p[5:7]) for p in only1}
        db = {int(p[3]): (tuple(p[5:7]), tuple(p[7:9])) for p in both}
        exact = set(db) == (set(d2) & set(d1)) and all(db[q] == (d2[q], d1[q]) for q in db)
        rec = {'trial': len(out), 'variant': variant, 'k_m': k_m, 'q_g': q_g, 'G0': G0, 'gen_state_out_equals_r1sim': o['fnv_state_out'] == g['state_out'],
               'x2': x2, 'z2': z2, 'x1': x1, 'z1': z1, 'obs': o, 'cmin': [c2, c1], 'imin': float(imin),
               'planted_found_two_maps': bool(pl), 'planted_line': pl[0] if pl else None,
               'planted_found_R2_only': any(int(p[3]) == q_g for p in only2), 'planted_found_R1_only': any(int(p[3]) == q_g for p in only1),
               'false_two_maps': len(both) - len(pl), 'two_map_set_equals_intersection': exact, 'false_R2_only': len(only2) - any(int(p[3]) == q_g for p in only2),
               'false_R1_only': len(only1) - any(int(p[3]) == q_g for p in only1), 'summary_two_maps': sb}
        out.append(rec); print(json.dumps(rec), flush=True)
    tested = sum(int(r['summary_two_maps'].split('tested=')[1].split()[0]) for r in out)
    print(json.dumps({'trials': trials, 'planted_found_two_maps': sum(r['planted_found_two_maps'] for r in out), 'tested': tested,
                      'false_two_maps': sum(r['false_two_maps'] for r in out), 'false_R2_only': sum(r['false_R2_only'] for r in out),
                      'false_R1_only': sum(r['false_R1_only'] for r in out), 'all_worlds_identical': all(r['gen_state_out_equals_r1sim'] for r in out),
                      'all_two_map_sets_equal_intersection': all(r['two_map_set_equals_intersection'] for r in out), 'gpu_cpu_mismatches': len(GPU_MISMATCH)}))


if __name__ == '__main__':
    main()
