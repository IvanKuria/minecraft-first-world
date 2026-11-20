"""Recalibration of the FROZEN LINK scorer's null on FULL-HEIGHT shell worlds (lead's overnight item B).
Same procedure as link/calibrate.py (imported, unchanged): (A) real-footage null = control worlds scored against the 7 real
frames at 100 random position pairs + the 175 baseline candidate positions; (B) synthetic positive controls with realistic
model error (calibrate.pseudo_frames). Only the control worlds differ: full-height tgen worlds of the fitted windows
(random literal seeds, NOT candidate states). The FROZEN.json thresholds are not changed; this reports whether the 0.30
floor still sits above the full-height null. Output: full-null-calibration.json"""
import json, random, sys, subprocess, tempfile, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pc-replay'))
import pc_replay
import scorer, calibrate
from planted_full import VARS, tgen_spec

CTRL = ['Vfull-K4096-nt8-scan3.5', 'E62-K4096-nt6-scan3.5', 'E63-K4096-nt8-scan3.5', 'Vfull-K4096-nt8-rand16', 'Vfull-K1024-nt8-scan5']
_W = {}


def init():
    pc_replay.load_dll(); _W['frames'] = scorer.load_frames()


def make(args):
    key, i = args
    _, base, T = VARS[key]
    p = HERE / 'full' / f'ctrl_{key}_{i}.blocks'
    if not p.exists():
        subprocess.run([str(HERE / 'build' / 'tgen.exe'), 'run', tgen_spec(base, T), str(0xC0FFEE + 7919 * i + hash(key) % 100000), str(p)], capture_output=True, check=True)
    return str(p)


def null_one(path):
    rng = random.Random(path)
    w = scorer.World(scorer.load_blocks(path))
    states = json.loads((HERE.parent / 'link' / 'states-by-hypothesis.json').read_text())
    cand = [[r['positions_opening_r1_r2'][0], r['positions_opening_r1_r2'][2]] for r in states['rows'] if r['set'] == 'baseline175']
    rnd = [scorer.score(w, calibrate.rand_pos(rng), _W['frames'])['T_dark'] for _ in range(100)]
    cnd = [scorer.score(w, p, _W['frames'])['T_dark'] for p in cand]
    return path, rnd, cnd


def pos_one(args):
    path, others, seed = args
    rng = random.Random(seed)
    w = scorer.World(scorer.load_blocks(path))
    for _ in range(200):
        pos = calibrate.rand_pos(rng)
        pf = calibrate.pseudo_frames(w, pos, _W['frames'], rng)
        if pf is None: continue
        dark = float(np.mean([(f['obs'] == 1).mean() for f in pf])); sky = float(np.mean([(f['obs'] == 0).mean() for f in pf]))
        if not (.01 <= dark <= .15 and sky < .8): continue
        true = scorer.score(w, pos, pf)['T_dark']
        wrong = max(scorer.score(w, calibrate.rand_pos(rng), pf)['T_dark'] for _ in range(30))
        other = max(scorer.score(scorer.World(scorer.load_blocks(o)), pos, pf)['T_dark'] for o in others)
        return {'world': Path(path).stem, 'true_T_dark': true, 'wrong_position_max': wrong, 'other_world_max': other, 'dark_frac': dark}
    return None


def main():
    with Pool(len(CTRL) * 3, initializer=init) as p:
        worlds = p.map(make, [(k, i) for k in CTRL for i in range(3)])
        real = {Path(a).stem: {'random_positions': b, 'candidate_positions': c} for a, b, c in p.map(null_one, worlds)}
        pos = [x for x in p.map(pos_one, [(worlds[i % len(worlds)], [w for w in worlds if w != worlds[i % len(worlds)]][:6], i) for i in range(30)]) if x]
    alln = [x for v in real.values() for k in v for x in v[k]]
    out = {'status': 'Full-height recalibration only (FROZEN.json unchanged); no candidate world scored.',
           'control_worlds': {k: VARS[k] for k in CTRL},
           'real_footage_null_summary': {'n': len(alln), 'max_T_dark': max(alln), 'p99_T_dark': float(np.percentile(alln, 99)),
                                         'p999_T_dark': float(np.percentile(alln, 99.9)), 'mean_T_dark': float(np.mean(alln))},
           'per_world_max': {k: max(v['random_positions'] + v['candidate_positions']) for k, v in real.items()},
           'synthetic_positive_controls': pos,
           'synthetic_summary': {'n': len(pos), 'true_T_dark_min': min(t['true_T_dark'] for t in pos),
                                 'true_T_dark_median': float(np.median([t['true_T_dark'] for t in pos])),
                                 'wrong_position_max': max(t['wrong_position_max'] for t in pos),
                                 'other_world_max': max(t['other_world_max'] for t in pos)} if pos else None,
           'real_footage_null': real}
    (HERE / 'full-null-calibration.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['real_footage_null_summary']), json.dumps(out['synthetic_summary']), json.dumps(out['per_world_max']))


if __name__ == '__main__':
    main()
