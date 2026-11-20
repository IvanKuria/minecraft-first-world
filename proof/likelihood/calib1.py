"""First look: arrival time (sweeps) vs distance to the nearest source in the top layers of full-height worlds."""
import sys, json, numpy as np
from scipy.spatial import cKDTree
import lworld as L
tag = sys.argv[1]; ytop = int(sys.argv[2]) if len(sys.argv) > 2 else 63; margin = int(sys.argv[3]) if len(sys.argv) > 3 else 0
raw, arr, seeds, meta = L.load(tag)
A = arr.astype(np.float64) / L.V
ys = np.arange(ytop - 7, ytop + 1)
xs = np.arange(40, 216); zs = np.arange(40, 216)
Y, Z, X = np.meshgrid(ys, zs, xs, indexing='ij')
P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(float)
S = seeds[:, 1:4].astype(float); t = seeds[:, 4]
tr_all = cKDTree(S)
d_all, i_all = tr_all.query(P, k=4)
# shell distance (walls x/z in {0,255}, floor y=0), irrelevant in this interior box except floor (>=56): ignore
a = A[Y.ravel(), Z.ravel(), X.ravel()]
ok = a >= 0
d1 = d_all[:, 0]
print('cells', len(a), 'never set', int((~ok).sum()))
bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28]
print('d_min bin    n      meanA   sdA    q10    q50    q90   meanA/d')
for lo, hi in zip(bins[:-1], bins[1:]):
    m = ok & (d1 >= lo) & (d1 < hi)
    if m.sum() < 50: continue
    v = a[m]
    print('%4d-%-4d %7d  %6.3f %6.3f %6.3f %6.3f %6.3f  %6.3f' % (lo, hi, m.sum(), v.mean(), v.std(), *np.quantile(v, [.1, .5, .9]), v.mean() / ((lo + hi) / 2)))
# by layer
print('layer  meanA  frac_set_at_T for T=2..8')
for y in ys:
    m = ok & (P[:, 1] == y)
    print(y, '%.3f' % a[m].mean(), ' '.join('%.3f' % (a[m] < T).mean() for T in range(2, 9)))
