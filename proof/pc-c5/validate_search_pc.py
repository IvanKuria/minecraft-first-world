"""End-to-end planted-seed validation of the C5 search (synthetic ground truth, GEN's generator).

For each trial: invent a Math.random clock row (k_m, q_m), plant a generator Random created Delta seconds earlier
(k_g = k_m - 1), grow the world with GEN's `gen` binary from that exact JDK6 seed, observe it footage-style from a
plain spawn with the second-reset poses (c5sim obs), then run c5search over a q_g window around the planted value.
Checks: (1) gen's world equals c5sim's world (state_out), (2) the planted q_g is reported iff its own seed list passes
the thresholds, (3) false passes per tested seed (empirical rate of the full pipeline).
usage: validate_search.py <variant> <trials> <half_window> <cmin_rule> <imin> <threads>
  cmin_rule: 'all' (every observed component hit), 'allm1' (all but one), or an integer
"""
import json, subprocess, sys, random, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / 'gen/gen.exe'
import tempfile; TMP = HERE / 'fixtures'; TMP.mkdir(parents=True, exist_ok=True)
U0 = 8682522807148012; MULT = 0x5DEECE66D; MASK = (1 << 48) - 1
F = 3579545  # ACPI PM


def hs_nanos(q, f=F):
    return int((q / f) * 1e9)  # matches (jlong)((double)q/(double)f*1e9) for these magnitudes; cross-checked below in C


def plain_spawn(blocks, rng):
    b = blocks.reshape(64, 256, 256)
    top = np.where(b != 0, np.arange(64)[:, None, None], -1).max(axis=0)
    for _ in range(100000):
        x = rng.uniform(40, 216); z = rng.uniform(40, 216)
        xs = range(int(math.floor(x - .3)), int(math.floor(x + .3)) + 1); zs = range(int(math.floor(z - .3)), int(math.floor(z + .3)) + 1)
        if max(top[zz, xx] for xx in xs for zz in zs) == 42: return x, z
    raise RuntimeError('no plain spawn')


def main():
    variant, trials, half, cmin_rule, imin, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5]), int(sys.argv[6])
    rng = random.Random(20261015)
    out = []
    for t in range(trials):
        k_m = rng.randint(2, 8); q_m = rng.randint(int(600 * F), int(86400 * F)); delta = rng.uniform(10, 40)
        q_g = q_m - int(delta * F); k_g = k_m - 1
        G0 = ((U0 + k_g + hs_nanos(q_g)) ^ MULT) & MASK
        blk = TMP / 'w.blocks'
        g = json.loads(subprocess.run([str(GEN), 'run', variant, str(G0), str(blk)], capture_output=True, text=True, check=True).stdout)
        x2, z2 = plain_spawn(np.fromfile(blk, np.uint8), rng)
        obs = TMP / f'w{t}.obs'
        o = json.loads(subprocess.run([str(HERE / 'build/c5sim.exe'), 'obs', variant, str(G0), str(x2), str(z2), str(obs)], capture_output=True, text=True, check=True).stdout)
        same_world = o['fnv_state_out'] == g['state_out']
        ncomp = o['ncomp']
        cmin = ncomp if cmin_rule == 'all' else max(0, ncomp - 1) if cmin_rule == 'allm1' else int(cmin_rule)
        cands = TMP / f'w{t}.cands'
        cands.write_text(f'0 0 {k_m} {q_m} {x2} {z2}\n')
        r = subprocess.run([str(HERE / 'build/c5search.exe'), variant, str(cands), str(obs), str(F), '10', '40', '1', str(cmin), str(imin), str(threads),
                            str(q_g - half), str(q_g + half)], capture_output=True, text=True, check=True).stdout.splitlines()
        passes = [l.split() for l in r if l.startswith('PASS')]
        summ = r[-1]; (TMP / f'w{t}.cpu.txt').write_text(chr(10).join(r) + chr(10)); (TMP / f'w{t}.args').write_text(json.dumps([variant, str(F), str(cmin), str(imin), str(q_g - half), str(q_g + half)]))
        planted = [p for p in passes if int(p[3]) == q_g]
        rec = {'trial': t, 'variant': variant, 'k_m': k_m, 'q_m': q_m, 'q_g': q_g, 'G0': G0, 'gen_state_out_equals_c5sim': same_world,
               'x2': x2, 'z2': z2, 'nobs': o['nobs'], 'nopen': o['nopen'], 'ncomp': ncomp, 'cmin': cmin, 'imin': imin,
               'planted_found': bool(planted), 'planted_line': planted[0] if planted else None,
               'false_passes': len(passes) - len(planted), 'summary': summ}
        out.append(rec); print(json.dumps(rec), flush=True)
    tested = sum(int(r['summary'].split('tested=')[1].split()[0]) for r in out)
    fp = sum(r['false_passes'] for r in out)
    print(json.dumps({'trials': trials, 'planted_found': sum(r['planted_found'] for r in out), 'tested': tested, 'false_passes': fp,
                      'false_rate': fp / tested if tested else None, 'all_worlds_identical': all(r['gen_state_out_equals_c5sim'] for r in out)}))


if __name__ == '__main__':
    main()
