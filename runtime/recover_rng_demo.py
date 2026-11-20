"""Synthetic feasibility test; NOT a seed recovered from the historical footage."""
import json, math, time
from pathlib import Path
from z3 import BitVec, BitVecVal, Extract, UGE, ULE, Solver, sat

BASE = Path(__file__).resolve().parent
ROWS = [json.loads(x) for x in (BASE.parent / 'generator/respawn-runtime.jsonl').read_text().splitlines()]
OBS = [row[key] for row in ROWS if 'raw_rng_x' in row for key in ('raw_rng_x', 'raw_rng_z')]
A, C, MASK = 0x5DEECE66D, 11, (1 << 48) - 1
EXPECTED_INTERNAL_STATE = (123456789 ^ A) & MASK

def recover(error, count):
    # Bound the first26bits of each actual nextDouble using an outward-rounded
    # coordinate interval. Float-rounding uncertainty is conservatively included.
    solver = Solver()
    solver.set(timeout=20000)
    initial = BitVec('initial_state', 48)
    state = initial
    intervals = []
    for value in OBS[:count]:
        state = state * BitVecVal(A, 48) + BitVecVal(C, 48)
        high26 = Extract(47, 22, state)
        allowance = error + 2**-15
        lo = max(0, math.floor((value - allowance) * 2**18))
        hi = min((1 << 26)-1, math.ceil((value + allowance) * 2**18))
        solver.add(UGE(high26, lo), ULE(high26, hi))
        intervals.append([lo, hi])
        state = state * BitVecVal(A, 48) + BitVecVal(C, 48)
    started = time.monotonic()
    result = solver.check()
    row = {'coordinate_error_blocks':error, 'observation_count':count,
           'intervals_high26':intervals, 'status':str(result),
           'solve_seconds':time.monotonic()-started}
    if result == sat:
        found = solver.model()[initial].as_long()
        row['first_candidate_internal_state'] = found
        row['matches_synthetic_expected_state'] = found == EXPECTED_INTERNAL_STATE
        solver.add(initial != found)
        t = time.monotonic()
        row['second_candidate_check'] = str(solver.check())
        row['second_check_seconds'] = time.monotonic()-t
    elif str(result) == 'unknown': row['reason'] = solver.reason_unknown()
    return row

if __name__ == '__main__':
    rows = []
    for error, count in [(0.00001,6),(0.01,6),(0.1,6),(0.01,4)]:
        row = recover(error,count)
        print(json.dumps(row),flush=True)
        rows.append(row)
        (BASE/'rng-demo-results.json').write_text(json.dumps({
            'purpose':'Synthetic state-recovery feasibility; no historical seed claim',
            'synthetic_constructor_seed':123456789,
            'expected_initial_internal_state':EXPECTED_INTERNAL_STATE,
            'source_coordinates':OBS, 'results':rows},indent=2))
