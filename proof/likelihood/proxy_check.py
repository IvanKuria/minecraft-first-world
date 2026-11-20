"""Validate the lobs sweep proxy against TERRACE's rendered+swept synthetic labels (same world, same spawns)."""
import json, sys, subprocess, numpy as np
from pathlib import Path
import lworld as Lw
HERE = Path(__file__).resolve().parent
SPEC = {'shS1024.18874368': ("init=unset,ntypes=6,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,unsetfill=0,margin=1,cap=fixed,capy=42,shell=1", 18874368),
        'shS4096.14680064': ("init=unset,ntypes=6,rng=int,pick=xyz,nb=scan,rule=mono,K=4096,unsetfill=0,margin=1,cap=fixed,capy=42,shell=1", 14680064),
        'shR4096.62914560': ("init=unset,ntypes=6,rng=int,pick=xyz,nb=rand,rule=mono,K=4096,unsetfill=0,margin=1,cap=fixed,capy=42,shell=1", 62914560)}
PH = {'r1': (0.46819817592972424, 0.8626385697595783), 'r2': (0.8118, 0.8185)}
rhos = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.85]
tag = sys.argv[1]
spec, M = SPEC[tag]
syn = json.loads((HERE.parent / 'terrace/synth' / f'{tag}.json').read_text())
wpath = HERE / 'worlds' / f'{tag}.blocks'
if not wpath.exists():
    subprocess.run([str(Lw.LGEN), 'run', f'{spec},M={M}', '0x123456789AB', str(wpath)], check=True, capture_output=True)
b = np.fromfile(wpath, np.uint8).reshape(64, 256, 256)
import importlib.util
spec_s = importlib.util.spec_from_file_location('surf', HERE.parent / 'terrace/surf.py'); surf = importlib.util.module_from_spec(spec_s); spec_s.loader.exec_module(surf)
st, _ = surf.stats(b)
print('surface check: mine f0 %.5f  terrace f0 %.5f' % (st['f0'], syn['surface']['f0']))
spf = HERE / 'worlds' / f'{tag}.spawns'
with open(spf, 'w') as f:
    for i, s in enumerate(syn['spawns']):
        for w in ('r1', 'r2'):
            f.write(f"{i} {w} {s['cx'] + PH[w][0]:.6f} {s['cz'] + PH[w][1]:.6f}\n")
def bins(h):
    n = sum(h.values()); f = lambda p: sum(v for k, v in h.items() if p(int(k))) / n if n else float('nan')
    return [n, f(lambda k: k == 0), f(lambda k: k == -1), f(lambda k: k == -2), f(lambda k: k == -3), f(lambda k: k <= -4), f(lambda k: k >= 1)]
for rho in rhos:
    outp = HERE / 'worlds' / f'{tag}.obs.txt'
    subprocess.run([str(HERE / 'lobs'), str(wpath), '42', str(spf), str(outp), str(rho)], check=True)
    mine = {}; cur = None
    for l in outp.read_text().splitlines():
        p = l.split()
        if p[0] == 'SPAWN': cur = (int(p[1]), p[2]); mine[cur] = {}
        elif p[0] == 'S': mine[cur][int(p[3])] = mine[cur].get(int(p[3]), 0) + 1
    agg_m = {'r1': {}, 'r2': {}}; agg_t = {'r1': {}, 'r2': {}}
    for (i, w), h in mine.items():
        for k, v in h.items(): agg_m[w][k] = agg_m[w].get(k, 0) + v
        for k, v in syn['spawns'][i][f'{w}_0.35']['hist'].items(): agg_t[w][int(k)] = agg_t[w].get(int(k), 0) + v
    for w in ('r1', 'r2'):
        print('rho %.2f %s  proxy   n %5d f0 %.2f -1 %.2f -2 %.2f -3 %.2f <=-4 %.2f + %.2f' % (rho, w, *bins(agg_m[w])))
        print('          %s  TERRACE n %5d f0 %.2f -1 %.2f -2 %.2f -3 %.2f <=-4 %.2f + %.2f' % (w, *bins(agg_t[w])))
