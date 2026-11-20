"""Per-column features for the probabilistic growth model: nearest rock-source / air-seed distances of the top L cells.
Sources: kept seeds (type 0 air, 1 rock) and the rock shell (walls x/z in {0,255}; the floor is >= 55 blocks below the
top layers and never nearest there)."""
import numpy as np
from scipy.spatial import cKDTree
L = 8

def seed_trees(seeds, ytop):
    S = seeds[:, 1:4].astype(float); t = seeds[:, 4]
    keep = S[:, 1] <= ytop
    ta = cKDTree(S[keep & (t == 0)]); tr = cKDTree(S[keep & (t != 0)])
    return ta, tr

def column_feats(seeds, cols, ytop, W=256):
    """cols: (n,2) int x,z.  returns d_air[n,L], d_rock[n,L], d_shell[n] (depth i = cell y = ytop - i)."""
    ta, tr = seed_trees(seeds, ytop)
    n = len(cols)
    P = np.empty((n * L, 3))
    for i in range(L):
        P[i::L, 0] = cols[:, 0]; P[i::L, 1] = ytop - i; P[i::L, 2] = cols[:, 1]
    da, _ = ta.query(P); dr, _ = tr.query(P)
    x, z = cols[:, 0], cols[:, 1]
    dsh = np.minimum.reduce([x, W - 1 - x, z, W - 1 - z]).astype(float)
    return da.reshape(n, L), dr.reshape(n, L), dsh

def true_depth(world, cols, ytop):
    """column-top depth D = ytop - (highest solid y); clipped to L (=deep)."""
    col = world[:, cols[:, 1], cols[:, 0]]  # [D, n]
    s = col[:ytop + 1] != 0
    top = np.where(s.any(0), ytop - np.argmax(s[::-1], axis=0), -1)
    return np.minimum(ytop - top, L)
