"""Robustness of the synthetic positive control under heavier model error, and the twin-world artifact.

1. G3a vs G3b voxel agreement (same GEN seed, near-identical worlds -> their mutual 'other world' scores are not a null).
2. Positive controls with heavier error: pose sd 1.0 deg, height sd 0.3, position sd 0.15, luminance noise sd 25,
   plus a +/-8 luminance threshold shift. Other-world null excludes the twin of the true world.
Writes robustness.json.
"""
import glob, json, math, random
from pathlib import Path
import numpy as np
import scorer
from calibrate import rand_pos, REALISTIC

P = Path(__file__).parent
TWIN = {'G3a': 'G3b', 'G3b': 'G3a'}


def heavy_frames(world, pos, frames, rng, shift):
    out = []
    for f in frames:
        g = dict(f)
        g['yaw'] += math.radians(rng.gauss(0, 1.0)); g['pitch'] += math.radians(rng.gauss(0, 1.0)); g['h'] += rng.gauss(0, .3)
        px, pz = pos[f['which']]
        r = world.render(px + rng.gauss(0, .15), pz + rng.gauss(0, .15), g, plane=world.landing_plane(px, pz))
        if r is None:
            return None
        lum = r['lum'] + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 25, r['lum'].shape)
        g['obs'] = np.where(r['cls'] == 0, 0, np.where(lum < scorer.DARK_T + shift, 1, 2)).astype(np.uint8)
        out.append(g)
    return out


def main():
    rng = random.Random(7)
    frames = scorer.load_frames()
    worlds = {Path(f).stem: scorer.World(scorer.load_blocks(f)) for f in sorted(glob.glob(str(P / 'control-worlds/*.blocks')))}
    twin_agree = float((worlds['G3a'].b == worlds['G3b'].b).mean())
    trials = []
    while len(trials) < 30:
        name = rng.choice(REALISTIC); w = worlds[name]; pos = rand_pos(rng); shift = rng.choice([-8, 0, 8])
        pf = heavy_frames(w, pos, frames, rng, shift)
        if pf is None:
            continue
        dark = float(np.mean([(f['obs'] == 1).mean() for f in pf]))
        if not .01 <= dark <= .15:
            continue
        t = scorer.score(w, pos, pf)['T_dark']
        wp = max(scorer.score(w, rand_pos(rng), pf)['T_dark'] for _ in range(30))
        ow = max(scorer.score(v, pos, pf)['T_dark'] for n, v in worlds.items() if n not in (name, TWIN.get(name)))
        trials.append({'world': name, 'shift': shift, 'dark_frac': dark, 'true_T_dark': t, 'wrong_position_max': wp, 'other_world_max': ow})
        print(len(trials), name, round(t, 3), round(wp, 3), round(ow, 3), flush=True)
    out = {'G3a_G3b_voxel_agreement': twin_agree, 'heavy_error_trials': trials,
           'summary': {'true_min': min(x['true_T_dark'] for x in trials),
                       'true_median': float(np.median([x['true_T_dark'] for x in trials])),
                       'wrong_position_max': max(x['wrong_position_max'] for x in trials),
                       'other_world_max_excl_twin': max(x['other_world_max'] for x in trials)}}
    (P / 'robustness.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['summary']), 'twin agreement', twin_agree)


if __name__ == '__main__':
    main()
