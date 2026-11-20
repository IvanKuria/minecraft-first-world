"""Extended hidden-reset search: Notch's tick() calls resetPos() EVERY tick while R is held (60 tps), so a human
press (~50-250 ms) makes ~3-15 resets, of which only the last is visible. The Sept search covered 0..4 hidden resets
per interval; this covers 0..G (default 15) in parallel. Same solver, same frozen measurement intervals.
Output: conditional-two-reset/hidden-reset-gap-results-ext.jsonl (one line per scenario). Not seed authentication."""
import json, sys, time
from pathlib import Path
from multiprocessing import Pool
from tile_mixed_coordinates import tiled_recover
from benchmark_spawn_recovery import advance

BASE = Path(__file__).resolve().parent; FOLDER = BASE / 'conditional-two-reset'
absolute_intervals = [[0,256],[0,256],[195,201],[188,250],[222,225],[113,135]]
fractional_intervals = [[.075,.130],[.785,.835],[.42,.50],[.82,.92],[.78,.85],[.79,.86]]
absolute = [(a+b)/2 for a,b in absolute_intervals]; ae = [(b-a)/2 for a,b in absolute_intervals]
fractional = [(a+b)/2 for a,b in fractional_intervals]; fe = [(b-a)/2 for a,b in fractional_intervals]

def run(g):
    gap1, gap2 = g
    indices = [0, 1, 2+2*gap1, 3+2*gap1, 4+2*(gap1+gap2), 5+2*(gap1+gap2)]
    t = time.monotonic()
    result = tiled_recover(absolute, ae, fractional, fe, indices, cap=250000, max_calls=indices[-1]+2, target_estimate=300)
    states = result.pop('states')
    return {'hidden_resets_before_first_observed': gap1, 'hidden_resets_between_observed': gap2, 'coordinate_indices': indices,
            'n_states': len(states), 'covered_all_boxes_without_output_cap': result.get('covered_all_boxes_without_output_cap'),
            'seconds': round(time.monotonic()-t, 1), 'initial_internal_state_candidates': states}

if __name__ == '__main__':
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 15; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out = FOLDER / 'hidden-reset-gap-results-ext.jsonl'
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            r = json.loads(l); done.add((r['hidden_resets_before_first_observed'], r['hidden_resets_between_observed']))
    # self-check: (0,0) must reproduce the frozen 175 baseline
    if (0,0) not in done:
        r = run((0,0)); existing = {x['state_before_first_observed_reset'] for x in json.loads((FOLDER/'opening-holdout-results.json').read_text())['survivors']}
        assert {advance(s,4) for s in r['initial_internal_state_candidates']} == existing, 'baseline mismatch'
        r['matches_independent_two_stage_baseline'] = True
        with out.open('a') as f: f.write(json.dumps(r)+'\n')
        done.add((0,0)); print('baseline ok', r['n_states'], r['seconds'], 's', flush=True)
    todo = [(a,b) for a in range(G+1) for b in range(G+1) if (a,b) not in done]
    # most likely presses first: 2..9 hidden resets each
    todo.sort(key=lambda g: (abs(g[0]-5)+abs(g[1]-5)))
    with Pool(workers) as p, out.open('a') as f:
        for r in p.imap_unordered(run, todo):
            f.write(json.dumps(r)+'\n'); f.flush()
            print(r['hidden_resets_before_first_observed'], r['hidden_resets_between_observed'], r['n_states'], r['covered_all_boxes_without_output_cap'], r['seconds'], flush=True)
