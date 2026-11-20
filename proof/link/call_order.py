"""Convert the footage-derived RNG candidates into (a) the state before the opening spawn's x draw and
(b) the state at the END of world generation under each call-order hypothesis.

Inputs (unchanged, hashed into the output):
  ../../runtime/conditional-two-reset/opening-holdout-results.json   175 baseline (state before FIRST OBSERVED reset)
  ../../runtime/conditional-two-reset/candidate-internal-states.json 63,074 (same convention; superset, not used for replay)
  ../../runtime/conditional-two-reset/hidden-reset-gap-results.json  25 scenarios, 4,385 pairs (state before OPENING x draw)

Conventions (verified against runtime/benchmark_spawn_recovery.py and search_hidden_resets.py):
  * a Math.random() call = nextDouble = 2 LCG steps; Player.resetPos = 2 calls (x then z) = 4 steps.
  * 'state before X' = internal seed such that the next next() call begins X.
Writes states-by-hypothesis.json.
"""
import hashlib, json
from pathlib import Path
from lcg import advance, reset_xz, unscramble

P = Path(__file__).parent
RT = P.parents[1] / 'runtime/conditional-two-reset'

# Hypotheses for the number of LCG steps between END of generation and the opening resetPos x draw.
# Each is (name, steps, justification). Generation-internal calls (incl. any grass pass) belong to GEN's N.
HYPOTHESES = [
    ('H0_direct', 0,
     'Film end-frame + rd-132211: Level ctor -> LevelRenderer ctor -> Player ctor -> resetPos. rd-132211 Level ctor after '
     'filling only calls calcLightDepths/load (no RNG); LevelRenderer/Chunk/Tesselator/Textures ctors make no Math.random '
     'calls (bytecode-verified in generator/chunk-order: renderer consumes no Math RNG). Player ctor calls resetPos first.'),
] + [
    (f'H1_prefilm_resets_{j}', 4 * j,
     f'{j} extra resetPos call(s) (R pressed/held) after init but before the opening view; each = 2 Math.random = 4 steps. '
     'Rendering between them consumes nothing in rd-132211.') for j in range(1, 9)
] + [
    ('H2_renderer_ctor_1perchunk', 2 * 1024,
     'Speculative: filmed LevelRenderer/Chunk ctor drew one Math.random per chunk (16x4x16=1024 chunks), e.g. a random '
     'initial dirty flag for the stress test. Not seen in any preserved code.'),
    ('H2_renderer_ctor_2perchunk', 4 * 1024, 'As above, two draws per chunk.'),
]
EXTRA_CALL_OFFSETS_NOTE = ('The driver also scans a generic small offset k = 0..K extra Math.random calls (2k steps) '
                           'to cover unknown single draws (e.g. a random Player yaw) without enumerating causes.')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def positions(s_open, indices):
    """Reset coordinates for nextDouble indices (each reset = indices i, i+1)."""
    out = []
    for i in indices[::2]:
        x, z, _ = reset_xz(advance(s_open, 2 * i))
        out.append([x, z])
    return out


def main():
    hold = json.loads((RT / 'opening-holdout-results.json').read_text())
    gaps = json.loads((RT / 'hidden-reset-gap-results.json').read_text())
    rows = []
    # Baseline 175: state before first observed reset -> back 4 steps to opening.
    for s in hold['survivors']:
        s1 = s['state_before_first_observed_reset']
        s_open = advance(s1, -4)
        x, z, _ = reset_xz(s_open)
        assert abs(x - s['opening_x']) < 1e-9 and abs(z - s['opening_z']) < 1e-9, 'opening prediction mismatch'
        rows.append({'set': 'baseline175', 'gap_before_first': 0, 'gap_between': 0, 's_open': s_open,
                     'coordinate_indices': [0, 1, 2, 3, 4, 5]})
    base = {r['s_open'] for r in rows}
    n_pairs = 0
    for t in gaps['tests']:
        for s_open in t['initial_internal_state_candidates']:
            n_pairs += 1
            rows.append({'set': 'hidden_gap_pairs', 'gap_before_first': t['hidden_resets_before_first_observed'],
                         'gap_between': t['hidden_resets_between_observed'], 's_open': s_open,
                         'coordinate_indices': t['coordinate_indices']})
    zero = {r['s_open'] for r in rows if r['set'] == 'hidden_gap_pairs' and r['gap_before_first'] == 0 and r['gap_between'] == 0}
    assert zero == base, 'gap(0,0) must reproduce the 175 baseline'
    for i, r in enumerate(rows):
        r['id'] = i
        r['positions_opening_r1_r2'] = positions(r['s_open'], r['coordinate_indices'])
        r['s_gen_end'] = {name: advance(r['s_open'], -steps) for name, steps, _ in HYPOTHESES}
        # Own-Random linkage (H3): only meaningful if resetPos is the first Math.random() in the JVM.
        r['H3_math_random_ctor_seed_low48'] = unscramble(r['s_open'])
    out = {
        'status': 'Conditional state conversions; no candidate authenticated.',
        'inputs_sha256': {f: sha(RT / f) for f in ['opening-holdout-results.json', 'hidden-reset-gap-results.json']},
        'counts': {'baseline175': len(base), 'hidden_gap_pairs': n_pairs, 'distinct_s_open': len({r['s_open'] for r in rows})},
        'hypotheses': [{'name': n, 'steps_from_gen_end_to_opening_x': s, 'why': w} for n, s, w in HYPOTHESES],
        'extra_offsets': EXTRA_CALL_OFFSETS_NOTE,
        'H3_own_random': ('If the generator used its own `new Random()` (rd-160052 style: Level field + PerlinNoiseFilter), '
                          'Math\'s lazy Random is created at the first Math.random() (Player.resetPos). JDK6 Random() seed = '
                          '++seedUniquifier + System.nanoTime(). Then s_open = (seedMath ^ 0x5DEECE66D) & mask, and the '
                          'generator seed (low 48 bits) = seedMath - k - (t_math - t_gen) mod 2^48, with k = 1 + number of '
                          'other no-arg Random() constructions in between (small), and t_math - t_gen ~ generation time '
                          '(~20 s ~ 2^34.2 ns). Search width per candidate ~ (window ns)/(nanoTime tick). Not enumerated.'),
        'gen_start_rule': 'S_gen_start = advance(s_gen_end[H], -N_steps(variant, params)); N_steps = 2 x Math.random calls.',
        'rows': rows,
    }
    (P / 'states-by-hypothesis.json').write_text(json.dumps(out) + '\n')
    print(json.dumps(out['counts']), len(rows))


if __name__ == '__main__':
    main()
