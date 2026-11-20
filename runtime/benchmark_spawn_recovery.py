"""Noisy-coordinate and missing-reset tests; never uses truth inside recover().

Inputs are actual java.util.Random outputs cast as the preserved Player does.
Nearest lattice point is one candidate, not exhaustive search or uniqueness proof.
"""
import json, math, random, struct, time
from pathlib import Path
from fpylll import IntegerMatrix, LLL, CVP

BASE = Path(__file__).resolve().parent
A, C, M = 0x5DEECE66D, 11, 1 << 48

def affine(steps):
    if not isinstance(steps,int) or steps<0:
        raise ValueError('steps must be a nonnegative integer')
    mul, add, a, c = 1, 0, A, C
    while steps:
        if steps & 1:
            mul, add = a * mul % M, (a * add + c) % M
        a, c = a * a % M, c * (a + 1) % M
        steps >>= 1
    return mul, add

def advance(state, steps):
    mul, add = affine(steps)
    return (mul * state + add) % M

def coordinate(first_state):
    second = (A * first_state + C) % M
    double = ((first_state >> 22) * 2**27 + (second >> 21)) / 2**53
    return struct.unpack('f', struct.pack('f', double))[0] * 256

def recover(observations, error, indices=None):
    """indices count nextDouble calls after the initial state, zero based."""
    if indices is None:
        indices = list(range(len(observations)))
    count = len(observations)
    B = IntegerMatrix(count, count)
    target, bounds, offsets = [], [], []
    for i, (value, index) in enumerate(zip(observations, indices)):
        # Float round-to-nearest may move coordinate by at most 2^-17.
        # Conservative 2^-15 allowance; final exact float check is tighter.
        allowance = error + 2**-15
        low = max(0, math.floor((value - allowance) * 2**18)) << 22
        high = min((1 << 26) - 1, math.ceil((value + allowance) * 2**18))
        high = ((high + 1) << 22) - 1
        mul, offset = affine(2 * (index - indices[0]))
        bounds.append((low, high)); offsets.append(offset)
        target.append((low + high) // 2 - offset)
        B[0, i] = mul
        if i:
            B[i, i] = M
    LLL.reduction(B)
    vector = CVP.closest_vector(B, tuple(target), method='proved')
    first = vector[0] % M
    # Return state before coordinate index zero, including skipped opening calls.
    mul, offset = affine(2 * indices[0] + 1)
    initial = (first - offset) * pow(mul, -1, M) % M
    predicted = [coordinate(advance(initial, 2*i + 1)) for i in indices]
    bucket_pass = all(lo <= (v + off) % M <= hi
                      for v, off, (lo, hi) in zip(vector, offsets, bounds))
    # Final exact Java-float outputs, independent of widened bit bucket bounds.
    # Fixtures serialize float values as doubles, retaining their exact values.
    exact_pass = all(abs(v - obs) <= error + 1e-12
                     for v, obs in zip(predicted, observations))
    return {'candidate_internal_state': initial, 'passes_buckets': bucket_pass,
            'passes_measured_coordinates': exact_pass, 'predicted': predicted}

def main():
    fixtures = [json.loads(line) for line in (BASE/'spawn-fixtures.jsonl').read_text().splitlines()]
    result = {'label': 'Synthetic noisy measurements only; no historical seed recovered',
              'limits': ['Nearest candidate only; not exhaustive, no uniqueness claim',
                         'Known exact coordinate axes and origin in this experiment',
                         'No physics, interpolation or camera errors beyond injected noise',
                         'Generator and RNG equivalence to filmed version remain unproven'],
              'benchmarks': [], 'gap_tests': []}
    for count, error in [(4,.01),(4,.1),(6,.01),(6,.1),(6,.5),(6,1.0)]:
        for mode in ('quantized', 'uniform_noise'):
            truth_matches, interval_passes, heldout_matches = 0, 0, 0
            started = time.monotonic()
            for sample, fixture in enumerate(fixtures):
                noise = random.Random(sample + 931)
                original = fixture['coordinates'][:count]
                if mode == 'quantized':
                    observations = [round(v/(2*error))*(2*error) for v in original]
                else:
                    observations = [v + noise.uniform(-error,error) for v in original]
                attempt = recover(observations,error)
                candidate = attempt['candidate_internal_state']
                truth_matches += candidate == ((fixture['seed'] ^ A) % M)
                interval_passes += attempt['passes_measured_coordinates']
                # Independent next coordinate, absent from fitting input.
                heldout_matches += abs(coordinate(advance(candidate,2*count+1))
                                       - fixture['coordinates'][count]) < 1e-12
            row = {'coordinate_count':count,'error_blocks':error,'mode':mode,
                   'trials':len(fixtures),'correct_state':truth_matches,
                   'passes_intervals':interval_passes,'heldout_coordinate_match':heldout_matches,
                   'seconds':time.monotonic()-started}
            result['benchmarks'].append(row); print(json.dumps(row),flush=True)
    # Two unobserved R resets in gap1, one in gap2. Each reset consumes two doubles.
    truth_indices = [0,1,6,7,10,11]
    error = .1
    for sample, fixture in enumerate(fixtures[:8]):
        observations = [round(fixture['coordinates'][i]/(2*error))*(2*error)
                        for i in truth_indices]
        candidates = []
        started = time.monotonic()
        for gap1 in range(5):
            for gap2 in range(5):
                indices = [0,1,2+2*gap1,3+2*gap1,4+2*(gap1+gap2),5+2*(gap1+gap2)]
                attempt = recover(observations,error,indices)
                if attempt['passes_measured_coordinates']:
                    candidates.append({'gap1':gap1,'gap2':gap2,**attempt})
        row = {'fixture':sample,'searched_gap_pairs':25,'candidates':candidates,
               'true_state_retained':any(c['candidate_internal_state']==((fixture['seed']^A)%M)
                                         and (c['gap1'],c['gap2'])==(2,1) for c in candidates),
               'seconds':time.monotonic()-started}
        result['gap_tests'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
    (BASE/'spawn-recovery-benchmark.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':
    main()
