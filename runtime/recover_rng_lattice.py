"""Recover synthetic Java RNG state from bounded respawn coordinates using CVP.

This tests an attack under known call spacing, NOT the historical video seed.
Intervals stay centered on exact coordinates: NOT a noisy-measurement test.
For genuinely perturbed/quantized measurements see benchmark_spawn_recovery.py.
The candidate must pass interval checks; nearest-vector output alone is no proof.
"""
import json, math, time
from pathlib import Path
from fpylll import IntegerMatrix, LLL, CVP

BASE = Path(__file__).resolve().parent
ROWS = [json.loads(x) for x in (BASE.parent/'generator/respawn-runtime.jsonl').read_text().splitlines()]
OBS = [row[key] for row in ROWS if 'raw_rng_x' in row for key in ('raw_rng_x','raw_rng_z')]
A,C,M = 0x5DEECE66D,11,1<<48
MASK = M-1
EXPECTED = (123456789 ^ A) & MASK
AA,CC = A*A % M, C*(A+1) % M

def next_double(state):
    first = (A*state+C) & MASK
    second = (A*first+C) & MASK
    return second, ((first>>22)*2**27+(second>>21))/2**53

def recover(error,count):
    lo,hi,target,offsets = [],[],[],[]
    offset = 0
    B = IntegerMatrix(count,count)
    for i,value in enumerate(OBS[:count]):
        allowance=error+2**-15
        low=max(0,math.floor((value-allowance)*2**18))<<22
        high=min((1<<26)-1,math.ceil((value+allowance)*2**18))
        high=((high+1)<<22)-1
        lo.append(low);hi.append(high);offsets.append(offset)
        target.append((low+high)//2-offset)
        B[0,i]=pow(AA,i,M)
        if i:B[i,i]=M
        offset=(AA*offset+CC)%M
    started=time.monotonic()
    LLL.reduction(B)
    vector=CVP.closest_vector(B,tuple(target),method='proved')
    first_state=vector[0] % M
    initial=((first_state-C)*pow(A,-1,M))%M
    actual=[];state=initial
    for _ in range(count):
        state,value=next_double(state);actual.append(value*256)
    state,holdout=next_double(state)
    valid=all(lo[i]<=((vector[i]+offsets[i])%M)<=hi[i] for i in range(count))
    return {'coordinate_error_blocks':error,'observation_count':count,
        'solve_seconds':time.monotonic()-started,
        'candidate_internal_state':initial,
        'passes_all_coordinate_intervals':valid,
        'matches_synthetic_expected_state':initial==EXPECTED,
        'unrounded_predicted_coordinates':actual,
        'next_double_after_observations':holdout,
        'held_out_double_matches':holdout==ROWS[-1]['expected_next_double'] if count==6 else None,
        'uniqueness':'Not proven; CVP finds one nearest candidate, not all interval solutions'}

if __name__=='__main__':
    result={'label':'EXACT-CENTERED SYNTHETIC FEASIBILITY ONLY: no historical seed recovered',
       'warning':'Widened intervals centered on exact values do not test measurement-noise robustness',
       'expected_synthetic_internal_state':EXPECTED,'measured_java_coordinates':OBS,'results':[]}
    for error,count in [(1e-5,6),(.01,6),(.1,6),(.5,6),(.01,4),(.1,4)]:
        row=recover(error,count);result['results'].append(row);print(json.dumps(row),flush=True)
    (BASE/'rng-lattice-results.json').write_text(json.dumps(result,indent=2))
