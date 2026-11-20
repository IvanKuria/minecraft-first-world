"""Write FROZEN.json: thresholds + hashes. Run once, before any candidate world is generated."""
import hashlib, json, time
from pathlib import Path
P = Path(__file__).parent; R = P.parents[1]
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = ['scorer.py', 'raycast.c', 'lcg.py', 'call_order.py', 'states-by-hypothesis.json', 'null_for_hit.py',
         'calibrate.py', 'null-calibration.json', 'robustness.json'] + [f'frames/{n}.png' for n in
         ['o-2.017', 'o-2.121', 'o-2.276', 'r2-153.054', 'r2-153.158', 'r2-153.313', 'r2-153.520']]
ext = {'geometry/opening-edge/pose-snapshot.json': R / 'geometry/opening-edge/pose-snapshot.json',
       'geometry/edge-registration/four-texture-poses.json': R / 'geometry/edge-registration/four-texture-poses.json'}
cal = json.loads((P / 'null-calibration.json').read_text()); rob = json.loads((P / 'robustness.json').read_text())
out = {'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'statement': 'Frozen before any candidate-state world was generated or scored. Thresholds come only from control '
                    'worlds (random seeds) against real frames and synthetic positive controls.',
       'footage_sha256': '0fcefdaf79dfa1f80ea975a3504c1ddb8837f57dedd3b2d6336d01fa72be362a',
       'thresholds': {'T_dark_floor': 0.30, 'T_dark_r2_floor': 0.30, 'mIoU_floor': 0.0,
                      'margin_over_null_max': 0.15, 'null_positions': 200, 'null_worlds': 20},
       'rationale': {'real_footage_null_max_T_dark': cal['real_footage_null_summary']['max_T_dark'],
                     'real_footage_null_p99_T_dark': cal['real_footage_null_summary']['p99_T_dark'],
                     'synthetic_true_median_light_error': cal['synthetic_summary']['true_T_dark_median'],
                     'synthetic_true_min_light_error': cal['synthetic_summary']['true_T_dark_min'],
                     'synthetic_true_median_heavy_error': rob['summary']['true_median'],
                     'synthetic_wrong_position_max_heavy': rob['summary']['wrong_position_max'],
                     'synthetic_other_world_max_heavy_excl_twin': rob['summary']['other_world_max_excl_twin']},
       'decision_rule': 'DECISIVE iff self-consistent RNG replay (gen state_out == S_gen_end) AND T_dark>=0.30 AND '
                        'T_dark_r2>=0.30 AND T_dark >= max(per-hit null: same world at 200 random positions, 20 same-variant '
                        'random-state worlds at the hit positions) + 0.15. Anything else is NOT a recovery.',
       'sha256': {f: h(P / f) for f in files} | {k: h(v) for k, v in ext.items()}}
(P / 'FROZEN.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out['thresholds']))
