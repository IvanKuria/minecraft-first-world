"""Footage vs synthetic: near-field (d<10 of the player) sweep-label histograms, same thresholds."""
import json, math, sys, glob
from pathlib import Path
import numpy as np
from labhist import labels_from_scores
HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
NEAR = 10.0
THR = {'0.45': (.45, .15), '0.35': (.35, .12), '0.3': (.3, .1)}

def bins(h):
    """h: dict k->count  ->  fractions in bins 0, -1, -2, -3, <=-4, + (>=1)"""
    h = {int(k): v for k, v in h.items()}
    n = sum(h.values())
    if n == 0: return None
    f = lambda pred: sum(v for k, v in h.items() if pred(k)) / n
    return {'n': n, '0': f(lambda k: k == 0), '-1': f(lambda k: k == -1), '-2': f(lambda k: k == -2), '-3': f(lambda k: k == -3),
            '<=-4': f(lambda k: k <= -4), '+': f(lambda k: k >= 1), '+1': f(lambda k: k == 1)}

def footage(thr='0.35'):
    out = {}
    for which in ('r1', 'r2'):
        d = json.loads((HERE / f'sweep-{which}.json').read_text())
        lab = labels_from_scores(d['scores'], *THR[thr])
        ph = d['phase']
        lab = {c: k for c, k in lab.items() if math.hypot(c[0] + .5 - ph[0], c[1] + .5 - ph[1]) < NEAR}
        h = {}
        for k in lab.values(): h[k] = h.get(k, 0) + 1
        obs = json.loads((PROOF / ('r1map/footage-obs-r1.json' if which == 'r1' else 'c5/footage-obs.json')).read_text())
        near_obs = [(x, z, v) for x, z, v in obs['labels'] if math.hypot(x + .5 - ph[0], z + .5 - ph[1]) < NEAR]
        out[which] = {'hist': h, 'bins': bins(h), 'obs_open_near': sum(v for *_, v in near_obs), 'obs_near': len(near_obs)}
    return out

def synth(path, thr='0.35'):
    d = json.loads(Path(path).read_text())
    agg = {'r1': {}, 'r2': {}}; tru = {'r1': {}, 'r2': {}}; per = {'r1': [], 'r2': []}
    for sp in d['spawns']:
        for which in ('r1', 'r2'):
            h = sp[f'{which}_{thr}']['hist']
            per[which].append(bins(h))
            for k, v in h.items(): agg[which][int(k)] = agg[which].get(int(k), 0) + v
            for k, v in sp[f'{which}_truth_near'].items(): tru[which][int(k)] = tru[which].get(int(k), 0) + v
    return d, agg, tru, per

def fmt(b):
    if b is None: return '   (no labels)'
    return '%4d  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f' % (b['n'], b['0'], b['-1'], b['-2'], b['-3'], b['<=-4'], b['+'])

if __name__ == '__main__':
    thr = sys.argv[1] if len(sys.argv) > 1 else '0.35'
    F = footage(thr)
    print('near field d<%g, thresholds %s' % (NEAR, THR[thr]))
    print('%-22s %-4s %4s  %4s  %4s  %4s  %4s  %4s  %4s' % ('source', 'set', 'n', 'f0', 'f-1', 'f-2', 'f-3', 'f<=-4', 'f+'))
    for w in ('r1', 'r2'):
        print('%-22s %-4s %s   (obs map: %d open of %d near columns)' % ('FOOTAGE', w, fmt(F[w]['bins']), F[w]['obs_open_near'], F[w]['obs_near']))
    rows = []
    for p in sorted(glob.glob(str(HERE / 'synth/*.json'))):
        d, agg, tru, per = synth(p, thr)
        tag = Path(p).stem
        for w in ('r1', 'r2'):
            b = bins(agg[w]); bt = bins(tru[w])
            print('%-22s %-4s %s   truth: %s' % (tag, w, fmt(b), fmt(bt)))
