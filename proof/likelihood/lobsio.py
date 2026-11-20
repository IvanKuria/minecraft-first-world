"""Observation classes for the likelihood scorer, shared by synthetic (lobs) and footage observations."""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
LD = 8                      # model depth classes 0..7, 8 = deep
NCLS = 19
PH = {'r1': (0.46819817592972424, 0.8626385697595783), 'r2': (0.8118, 0.8185)}

def classify(sweep, unl, mraw, phase):
    """sweep: {(dx,dz): k}, unl: set of scored-unlabelled (d<10), mraw: {(dx,dz): 0/1} raw map labels -> {(dx,dz): cls}"""
    out = {}
    for c, k in sweep.items(): out[c] = k + 8
    for c in unl:
        if c in out: continue
        m = mraw.get(c); out[c] = 12 if m == 1 else 13 if m == 0 else 14
    for c, m in mraw.items():
        if c in out: continue
        d = math.hypot(c[0] + .5 - phase[0], c[1] + .5 - phase[1])
        out[c] = (15 if m == 1 else 16) if d < 16 else (17 if m == 1 else 18)
    return out

def parse_lobs(path):
    """-> list of spawn records {id, which, px, pz, O:[...], C:[...], M:{}, S:{}, U:set, T:{}}"""
    recs = []; cur = None
    for l in Path(path).read_text().splitlines():
        p = l.split()
        if p[0] == 'SPAWN':
            cur = {'id': p[1], 'which': p[2], 'px': float(p[3]), 'pz': float(p[4]), 'O': [], 'C': [], 'M': {}, 'S': {}, 'U': set(), 'T': {}}; recs.append(cur)
        elif p[0] == 'O': cur['O'].append(l)
        elif p[0] == 'C': cur['C'].append(l)
        elif p[0] == 'M': cur['M'][(int(p[1]), int(p[2]))] = int(p[3])
        elif p[0] == 'S': cur['S'][(int(p[1]), int(p[2]))] = int(p[3])
        elif p[0] == 'U': cur['U'].add((int(p[1]), int(p[2])))
        elif p[0] == 'T': cur['T'][(int(p[1]), int(p[2]))] = min(int(p[3]), LD) if int(p[3]) >= 0 else LD
    return recs

def footage(which):
    """real footage observation of reset r1/r2 in the same classes (sweep labels recomputed at TERRACE's 0.35/0.12)."""
    sw = json.loads((PROOF / 'terrace' / f'sweep-{which}.json').read_text())
    ph = PH[which]
    sweep, unl = {}, set()
    for x, z, s in sw['scores']:
        if math.hypot(x + .5 - ph[0], z + .5 - ph[1]) >= 10: continue
        it = sorted(((int(k), v) for k, v in s.items()), key=lambda kv: -kv[1][0])
        if it and it[0][1][0] >= .35 and (len(it) < 2 or it[0][1][0] - it[1][1][0] >= .12): sweep[(x, z)] = it[0][0]
        else: unl.add((x, z))
    mj = json.loads((PROOF / ('r1map/footage-obs-r1.json' if which == 'r1' else 'c5/footage-obs.json')).read_text())
    mraw = {(x, z): v for x, z, v in mj['labels']}
    obs = (PROOF / ('r1map/footage-r1.obs' if which == 'r1' else 'c5/footage.obs')).read_text().splitlines()
    O = [l for l in obs if l.startswith('O ')]; C = [l for l in obs if l.startswith('C ')]
    return classify(sweep, unl, mraw, ph), O, C, sweep, unl, mraw

def write_map(path, ref, cls, conf, lnull, O=(), C=()):
    with open(path, 'w') as f:
        f.write(f'R {ref}\nNCLS {NCLS}\n')
        for c in range(NCLS): f.write('CONF %d ' % c + ' '.join('%.6g' % x for x in conf[c]) + '\n')
        for c in range(NCLS): f.write('LNULL %d %.6f\n' % (c, lnull[c]))
        for (dx, dz), c in sorted(cls.items()): f.write(f'Q {dx} {dz} {c}\n')
        for l in O: f.write(l + '\n')
        for l in C: f.write(l + '\n')

def fixture_classes(m):
    """classes of one saved fixture map (pc_fixtures.py): sweep labels recomputed at 0.35/0.12, d < 10."""
    ph = m['phase']; sweep, unl = {}, set()
    for x, z, s in m['scores']:
        if math.hypot(x + .5 - ph[0], z + .5 - ph[1]) >= 10: continue
        it = sorted(((int(k), v) for k, v in s.items()), key=lambda kv: -kv[1][0])
        if it and it[0][1][0] >= .35 and (len(it) < 2 or it[0][1][0] - it[1][1][0] >= .12): sweep[(x, z)] = it[0][0]
        else: unl.add((x, z))
    mraw = {(x, z): v for x, z, v in m['labels']}
    return classify(sweep, unl, mraw, ph)
