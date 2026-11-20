"""Fit model M1 by maximum likelihood of the TRUE column depths of synthetic full-height worlds at several T;
evaluate on held-out worlds (bits per column vs the marginal-depth null)."""
import sys, json, numpy as np
from scipy.optimize import minimize
import lworld as Lw, feats as F, model as M
fam = sys.argv[1]; Ts = [float(x) for x in sys.argv[2].split(',')]; ytop = int(sys.argv[3]) if len(sys.argv) > 3 else 63
train = [fam + 'a', fam + 'b']; test = [fam + 'c', fam + 'd']
rng = np.random.default_rng(1)
def data(tags, n):
    out = []
    for t in tags:
        raw, arr, seeds, meta = Lw.load(t)
        cols = rng.integers(1, 255, size=(n, 2))
        da, dr, dsh = F.column_feats(seeds, cols, ytop)
        for T in Ts:
            w = Lw.world_at(raw, arr, int(T * Lw.V))
            Dt = F.true_depth(w, cols, ytop)
            out.append((da, dr, dsh, T, Dt))
    return out
tr = data(train, 15000); te = data(test, 15000)
def nll(th, ds):
    P = M.unpack(th); s = 0; n = 0
    for da, dr, dsh, T, Dt in ds:
        p = M.probs(da, dr, dsh, T, P)
        s -= np.log(p[np.arange(len(Dt)), Dt]).sum(); n += len(Dt)
    return s / n
th0 = np.array([0.4, np.log(0.6), 0.75, np.log(0.6), np.log(1.0), 0.0, np.log(0.8)])
if '_r' in fam or fam[3] == 'r':   # random-neighbour rule: slower fronts, own wait ~ Exp(n/6)
    th0 = np.array([2.0, np.log(2.5), 0.85, np.log(1.5), np.log(2.0), np.log(0.3), np.log(0.8)])
r = minimize(nll, th0, args=(tr,), method='Nelder-Mead', options={'maxiter': 2500, 'xatol': 1e-3, 'fatol': 1e-5})
r2 = minimize(nll, r.x, args=(tr,), method='Powell', options={'maxiter': 4000, 'xtol': 1e-3, 'ftol': 1e-6})
if r2.fun < r.fun: r = r2
P = M.unpack(r.x)
res = {'family': fam, 'ytop': ytop, 'Ts': Ts, 'params': {k: float(v) for k, v in P.items()}, 'train_nll': float(r.fun)}
# held-out: per T, bits/column vs null (marginal depth distribution from the TRAIN worlds at that T), calibration table
per = []
for (da, dr, dsh, T, Dt) in te:
    p = M.probs(da, dr, dsh, T, P)
    trD = np.concatenate([d[4] for d in tr if d[3] == T])
    p0 = np.bincount(trD, minlength=F.L + 1) / len(trD)
    ll = np.log2(p[np.arange(len(Dt)), Dt]); ll0 = np.log2(np.maximum(p0[Dt], 1e-12))
    pred = p.mean(0); emp = np.bincount(Dt, minlength=F.L + 1) / len(Dt)
    per.append({'T': T, 'bits_per_col': float((ll - ll0).mean()), 'emp': [round(x, 3) for x in emp], 'pred': [round(x, 3) for x in pred]})
res['heldout'] = per
print(json.dumps(res, indent=1))
