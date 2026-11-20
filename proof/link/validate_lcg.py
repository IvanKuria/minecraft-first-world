"""Validate lcg.py against Java 8 ground truth (java/random-fixtures.jsonl). Writes lcg-validation.json."""
import json, random, struct
from pathlib import Path
from lcg import JRandom, advance, back, affine, scramble, unscramble, M, MASK, to_float32

P = Path(__file__).parent
rows = [json.loads(l) for l in (P/'java/random-fixtures.jsonl').read_text().splitlines()]
checked = fails = 0
for row in rows:
    if 'seed' not in row:
        continue
    r = JRandom(seed=row['seed'])
    assert r.state == row['state0'], 'scramble'
    for op, v, before, after in row['ops']:
        assert r.state == before
        k0 = r.calls
        if op == 0: ok = r.next_double() == float(v)
        elif op == 1: ok = to_float32(r.next_float()) == to_float32(float(v))
        elif op == 2: ok = r.next_int() == int(v)
        elif op == 3:
            b, val = map(int, v.split(':')); ok = r.next_int(b) == val
        elif op == 4: ok = r.next_long() == int(v)
        elif op == 5: ok = r.next_boolean() == (v == 'true')
        else: ok = to_float32(to_float32(r.next_double()) * 256) == float(v)
        ok = ok and r.state == after and advance(before, r.calls - k0) == after and back(after, r.calls - k0) == before
        checked += 1; fails += (not ok)
    assert r.state == row['state_end']
jump = [r for r in rows if 'jump_seed' in r][0]
jump_ok = advance(jump['s0'], 2*jump['draws']) == jump['s_end'] and back(jump['s_end'], 2*jump['draws']) == jump['s0'] \
    and advance(jump['s_end'], -2*jump['draws']) == jump['s0']
rng = random.Random(1); prop = 0
for _ in range(2000):
    s = rng.randrange(M); n = rng.randrange(1, 1 << 60); m = rng.randrange(1 << 40)
    assert advance(advance(s, n), m) == advance(s, n + m)
    assert back(advance(s, n), n) == s == advance(advance(s, n), -n)
    assert unscramble(scramble(s)) == s
    prop += 1
out = {'java': 'Zulu 8.0.504 (proof/jdk), java.util.Random internal seed read by reflection',
       'random_ops_checked': checked, 'failures': fails,
       'op_kinds': ['nextDouble', 'nextFloat', 'nextInt()', 'nextInt(bound) incl. power-of-two and rejection bounds', 'nextLong', 'nextBoolean', '(float)nextDouble*256'],
       'per_op_checks': 'value, post-state, advance(before,k)==after, back(after,k)==before',
       'jump_2e6_steps_ok': jump_ok, 'algebraic_property_tests': prop,
       'note': 'Java 6 and 8 share the identical LCG and nextX algorithms; they differ only in the no-arg constructor seedUniquifier.'}
(P/'lcg-validation.json').write_text(json.dumps(out, indent=1) + '\n')
print(out)
