"""Predictions excluded from fitting, under the no-intervening-RNG scenario.

The preceding two random outputs are the opening spawn ONLY if the earlier
spawn and first observed R were consecutive resets of the same RNG stream.
"""
import json,csv
from pathlib import Path
from benchmark_spawn_recovery import M,advance,coordinate

BASE=Path(__file__).resolve().parent
folder=BASE/'conditional-two-reset'
states=json.loads((folder/'candidate-internal-states.json').read_text())
with (folder/'predicted-opening.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['state_before_first_observed_reset','opening_x','opening_z','fraction_x','fraction_z'])
    for state in states:
        # JavaLCG has period2^48: forwardM-3/M-1 equals backward3/backward1.
        x=coordinate(advance(state,M-3));z=coordinate(advance(state,M-1))
        writer.writerow([state,x,z,x%1,z%1])
print(json.dumps({'predictions':len(states),'historical_validation':False,
    'condition':'Opening spawn then observedR1 then observedR2, with no extra RNG consumption'}))
