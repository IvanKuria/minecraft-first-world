"""Per-hit null (part of the frozen decision rule). For a replay result that passed the floor:
  (a) the SAME regrown world at 200 random position pairs (seeded by the job key),
  (b) 20 worlds of the SAME variant from random start states, scored at the hit's positions.
DECISIVE iff all FROZEN.json conditions hold. Usage (under heavy-lock): PY null_for_hit.py RESULTS.jsonl KEY
"""
import json, random, subprocess, sys, tempfile, os
from pathlib import Path
import scorer
from calibrate import rand_pos
from replay_driver import gen_run

P = Path(__file__).parent


def main(results, key):
    frozen = json.loads((P / 'FROZEN.json').read_text())['thresholds']
    rec = [json.loads(l) for l in Path(results).read_text().splitlines() if json.loads(l)['key'] == key][-1]
    frames = scorer.load_frames()
    rng = random.Random(key)
    tmp = tempfile.mkdtemp(prefix='link-null-'); wp = os.path.join(tmp, 'w.blocks')
    gen_run(rec['variant'], rec['s_gen_start'], wp)
    world = scorer.World(scorer.load_blocks(wp))
    pos_null = [scorer.score(world, rand_pos(rng), frames)['T_dark'] for _ in range(frozen['null_positions'])]
    world_null = []
    for _ in range(frozen['null_worlds']):
        gen_run(rec['variant'], rng.randrange(1 << 48), wp)
        world_null.append(scorer.score(scorer.World(scorer.load_blocks(wp)), rec['positions'], frames)['T_dark'])
    os.remove(wp)
    s = rec['score']; nmax = max(pos_null + world_null)
    checks = {'T_dark_floor': s['T_dark'] >= frozen['T_dark_floor'],
              'T_dark_r2_floor': s['T_dark_r2'] >= frozen['T_dark_r2_floor'],
              'margin_over_null_max': s['T_dark'] >= nmax + frozen['margin_over_null_max'],
              'self_consistent_rng': rec.get('self_consistent', False)}
    out = {'key': key, 'score': {k: v for k, v in s.items() if k != 'per_frame'}, 'null_max': nmax,
           'pos_null': pos_null, 'world_null': world_null, 'checks': checks, 'DECISIVE': all(checks.values())}
    Path(results).with_suffix('.nulls.jsonl').open('a').write(json.dumps(out) + '\n')
    print(json.dumps({k: out[k] for k in ['key', 'null_max', 'checks', 'DECISIVE']}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
