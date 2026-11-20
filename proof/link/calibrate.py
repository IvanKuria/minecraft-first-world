"""Null distribution + synthetic positive control for the LINK scorer. Run BEFORE any candidate world exists.

A) Real-footage null: control worlds (GEN's early variant samples, random seeds, NOT candidate states) scored against the
   actual 7 frames at (i) 100 random position pairs each and (ii) the 175 baseline candidate position pairs.
B) Synthetic positive control: pick a control world and positions, render 'pseudo-footage' with realistic model error
   (pose yaw/pitch noise sd 0.3 deg, camera height sd 0.1, horizontal position sd 0.05, pooled luminance noise sd 12),
   then score the TRUE world, the same world at 30 wrong positions, and the other worlds at the true positions.
Writes null-calibration.json.
"""
import glob, json, math, random, sys, time
from pathlib import Path
import numpy as np
import scorer

P = Path(__file__).parent
REALISTIC = ['G2a', 'G2b', 'G2c', 'G3a', 'G3b', 'G5a']


def rand_pos(rng):
    return [[rng.uniform(.5, 255.5), rng.uniform(.5, 255.5)], [rng.uniform(.5, 255.5), rng.uniform(.5, 255.5)]]


def pseudo_frames(world, pos, frames, rng):
    out = []
    for f in frames:
        g = dict(f)
        g['yaw'] += math.radians(rng.gauss(0, .3)); g['pitch'] += math.radians(rng.gauss(0, .3)); g['h'] += rng.gauss(0, .1)
        px, pz = pos[f['which']]
        r = world.render(px + rng.gauss(0, .05), pz + rng.gauss(0, .05), g, plane=world.landing_plane(px, pz))
        if r is None:
            return None
        lum = r['lum'] + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 12, r['lum'].shape)
        sky = r['cls'] == 0
        g['obs'] = np.where(sky, 0, np.where(lum < scorer.DARK_T, 1, 2)).astype(np.uint8)
        out.append(g)
    return out


def main():
    rng = random.Random(20261015)
    frames = scorer.load_frames()
    worlds = {Path(f).stem: scorer.World(scorer.load_blocks(f)) for f in sorted(glob.glob(str(P / 'control-worlds/*.blocks')))}
    states = json.loads((P / 'states-by-hypothesis.json').read_text())
    cand_pos = [[r['positions_opening_r1_r2'][0], r['positions_opening_r1_r2'][2]] for r in states['rows'] if r['set'] == 'baseline175']
    t0 = time.time()
    real = {}
    for name, w in worlds.items():
        rnd = [scorer.score(w, rand_pos(rng), frames) for _ in range(100)]
        cnd = [scorer.score(w, p, frames) for p in cand_pos]
        real[name] = {'random_positions': [[s['T_dark'], s['mIoU']] for s in rnd],
                      'candidate_positions': [[s['T_dark'], s['mIoU']] for s in cnd]}
        print(name, 'max T_dark', max(x[0] for x in real[name]['random_positions'] + real[name]['candidate_positions']),
              round(time.time() - t0), flush=True)
    pos_trials = []
    names = [n for n in REALISTIC if n in worlds]
    while len(pos_trials) < 30:
        name = rng.choice(names); w = worlds[name]; pos = rand_pos(rng)
        pf = pseudo_frames(w, pos, frames, rng)
        if pf is None:
            continue
        dark_frac = float(np.mean([(f['obs'] == 1).mean() for f in pf]))
        sky_frac = float(np.mean([(f['obs'] == 0).mean() for f in pf]))
        if not (.01 <= dark_frac <= .15 and sky_frac < .8):
            continue
        true = scorer.score(w, pos, pf)
        wrongpos = [scorer.score(w, rand_pos(rng), pf)['T_dark'] for _ in range(30)]
        otherw = [scorer.score(v, pos, pf)['T_dark'] for n, v in worlds.items() if n != name]
        pos_trials.append({'world': name, 'positions': pos, 'dark_frac': dark_frac, 'true_T_dark': true['T_dark'],
                           'true_mIoU': true['mIoU'], 'wrong_position_max': max(wrongpos), 'other_world_max': max(otherw)})
        print('pos', len(pos_trials), name, round(true['T_dark'], 3), round(max(wrongpos), 3), round(max(otherw), 3), flush=True)
    allnull = [x[0] for v in real.values() for k in v for x in v[k]]
    out = {'status': 'Calibration only; no candidate world was generated or scored.',
           'control_worlds': 'proof/link/control-worlds (copied from proof/gen/worlds, SHA256SUMS) (early GEN variants, arbitrary seeds)',
           'real_footage_null': real,
           'real_footage_null_summary': {'n': len(allnull), 'max_T_dark': max(allnull),
                                         'p99_T_dark': float(np.percentile(allnull, 99)), 'mean_T_dark': float(np.mean(allnull))},
           'synthetic_positive_controls': pos_trials,
           'synthetic_summary': {'true_T_dark_min': min(t['true_T_dark'] for t in pos_trials),
                                 'true_T_dark_median': float(np.median([t['true_T_dark'] for t in pos_trials])),
                                 'wrong_position_max': max(t['wrong_position_max'] for t in pos_trials),
                                 'other_world_max': max(t['other_world_max'] for t in pos_trials)},
           'seconds': time.time() - t0}
    (P / 'null-calibration.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['real_footage_null_summary']), json.dumps(out['synthetic_summary']))


if __name__ == '__main__':
    main()
