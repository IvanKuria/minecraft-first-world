"""Class-frequency comparison: real footage maps vs rendered fixtures (model check, no seeds involved)."""
import json, glob, sys, numpy as np, lobsio as O
groups = {'sw0': [8], 'sw-1': [7], 'sw-2': [6], 'sw<=-3': [0, 1, 2, 3, 4, 5], 'sw+': [9, 10, 11], 'Uopen': [12], 'Uclosed': [13], 'Unone': [14],
          'open<16': [15], 'clos<16': [16], 'open>=16': [17], 'clos>=16': [18]}
def hist(cls):
    n = len(cls); h = np.bincount(list(cls.values()), minlength=O.NCLS)
    return n, {g: h[v].sum() / n for g, v in groups.items()}
rows = []
for w in ('r1', 'r2'):
    n, h = hist(O.footage(w)[0]); rows.append(('REAL ' + w, n, h, None))
for key in sys.argv[1:]:
    for w in ('r1', 'r2'):
        acc = []; ns_ = []
        for f in sorted(glob.glob(f'fixtures/{key}/*/fx*.json')):
            m = json.load(open(f))['maps'][w]; n, h = hist(O.fixture_classes(m)); acc.append(h); ns_.append(n)
        if acc: rows.append((f'{key[6:]} {w} ({len(acc)}fx)', np.mean(ns_), {g: np.mean([a[g] for a in acc]) for g in groups}, {g: np.std([a[g] for a in acc]) for g in groups}))
print('%-30s %5s ' % ('', 'n') + ' '.join('%8s' % g for g in groups))
for name, n, h, sd in rows:
    print('%-30s %5d ' % (name, n) + ' '.join('%8.3f' % h[g] for g in groups))
    if sd: print('%-30s %5s ' % ('   (sd over fixtures)', '') + ' '.join('%8.3f' % sd[g] for g in groups))
