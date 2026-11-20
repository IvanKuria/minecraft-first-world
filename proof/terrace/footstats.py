"""Footage near-field label-map statistics (d<10): level fractions, step edges, 1-wide slits/bumps among labelled
columns (a slit = labelled column lower than BOTH labelled neighbours along x or along z)."""
import json, math
from labhist import labels_from_scores
from pathlib import Path
HERE = Path(__file__).resolve().parent
for which in ('r1', 'r2'):
    d = json.loads((HERE / f'sweep-{which}.json').read_text()); ph = d['phase']
    lab = labels_from_scores(d['scores'], .35, .12)
    lab = {c: k for c, k in lab.items() if math.hypot(c[0] + .5 - ph[0], c[1] + .5 - ph[1]) < 10}
    pairs = diff = d1 = d2 = 0
    for (x, z), k in lab.items():
        for dx, dz in ((1, 0), (0, 1)):
            q = (x + dx, z + dz)
            if q in lab:
                pairs += 1; dd = abs(lab[q] - k); diff += dd > 0; d1 += dd == 1; d2 += dd == 2
    slit = bump = tri = 0
    for (x, z), k in lab.items():
        for (a, b) in (((x - 1, z), (x + 1, z)), ((x, z - 1), (x, z + 1))):
            if a in lab and b in lab:
                tri += 1
                if lab[a] > k and lab[b] > k: slit += 1
                if lab[a] < k and lab[b] < k: bump += 1
    n = len(lab)
    print(f'{which}: labelled {n}; adjacent labelled pairs {pairs}: step edges {diff} ({diff/pairs:.3f}), |dk|=1 {d1/pairs:.3f}, |dk|=2 {d2/pairs:.3f}; '
          f'triples {tri}: slit {slit} ({slit/tri:.3f}), bump {bump} ({bump/tri:.3f})')
