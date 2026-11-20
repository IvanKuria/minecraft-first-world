"""Real footage observations (R1 first reset, R2 second reset) in the scorer's class scheme, with the confusion tables
of a chosen rendered-fixture calibration (pipeline-matched: trender + obs_footage + sweep).
usage: real_maps.py <fixture-key> <outdir>"""
import json, sys
from pathlib import Path
import lobsio as O
HERE = Path(__file__).resolve().parent
key, od = sys.argv[1], Path(sys.argv[2]); od.mkdir(parents=True, exist_ok=True)
CF = json.loads((HERE / 'fixtures' / key / 'conf.json').read_text())
summ = {}
for w in ('r1', 'r2'):
    cls, Ol, Cl, sweep, unl, mraw = O.footage(w)
    O.write_map(od / f'real-{w}.map', 1 if w == 'r1' else 2, cls, CF['conf'][w], CF['lnull'][w], Ol, Cl)
    h = {}
    for c in cls.values(): h[c] = h.get(c, 0) + 1
    summ[w] = {'columns': len(cls), 'sweep_labels': len(sweep), 'unlabelled_scored': len(unl), 'map_labels': len(mraw), 'comps': len(Cl), 'class_hist': dict(sorted(h.items()))}
print(json.dumps(summ))
