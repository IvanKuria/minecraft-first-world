"""Probabilistic full-height growth model ("M1"): per column P(D = 0..L-1, >= L) from the seed distances only.
Cell i of the column (y = ytop - i):
   rock path arrival  m_r = g(dr_i)  with dr_i = min(d_rockseed, kappa * d_shell)
   air  path arrival  m_a = g(da_i)
   rock wins the cell iff m_r - m_a < v          (v: column-shared competition noise ~ N(0, sv^2))
   solid iff rock wins and m_r + u + tau_i < T   (u: column-shared slowness ~ N(0, su^2); tau_i ~ Exp(lam) own wait, independent)
   D = number of non-solid cells from the top.
g(d) = a + b * d**p.  (u, v) integrated with Gauss-Hermite nodes."""
import numpy as np
from feats import L
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(5)
GH_W = GH_W / GH_W.sum()

def unpack(th):
    a, lb, p, lsu, lsv, llam, lkap = th
    return dict(a=a, b=np.exp(lb), p=p, su=np.exp(lsu), sv=np.exp(lsv), lam=np.exp(llam), kap=np.exp(lkap))

def g(d, P):
    return P['a'] + P['b'] * np.power(np.maximum(d, 0.5), P['p'])

def probs(da, dr, dsh, T, P):
    """returns pD [n, L+1]"""
    drr = np.minimum(dr, P['kap'] * dsh[:, None])
    mr = g(drr, P); ma = g(da, P)
    n = len(da)
    out = np.zeros((n, L + 1))
    for xu, wu in zip(GH_X, GH_W):
        u = P['su'] * xu
        slack = T - u - mr                      # [n,L]
        pf = np.where(slack > 0, 1 - np.exp(-P['lam'] * np.maximum(slack, 0)), 0.0)
        for xv, wv in zip(GH_X, GH_W):
            v = P['sv'] * xv
            win = (mr - ma) < v
            ps = np.where(win, pf, 0.0)         # P(cell i solid | u, v)
            q = 1 - ps
            cum = np.cumprod(q, axis=1)         # P(top i+1 cells all non-solid)
            prev = np.concatenate([np.ones((n, 1)), cum[:, :-1]], 1)
            pk = prev * ps                       # P(D = i)
            out[:, :L] += wu * wv * pk
            out[:, L] += wu * wv * cum[:, -1]
    return np.clip(out, 1e-12, 1)
