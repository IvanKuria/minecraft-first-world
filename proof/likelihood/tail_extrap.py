"""Crude null-tail extrapolation from a rank run's top-N dump: fit an exponential tail to the top-k stage-2 scores among
the stage-1 top-N of a window of W seeds; predict the number of seeds above the planted seed's stage-2 score in a window
of 1e8 (W scaled by 1e8/W).  Assumes the stage-2 upper tail is captured by the stage-1 top-N (checked: the planted seed
sits far inside it) and an exponential tail beyond the k-th best -- an approximation, not a measurement."""
import json, sys, math, numpy as np
from pathlib import Path
res = Path(sys.argv[1]); k = int(sys.argv[2]) if len(sys.argv) > 2 else 50
for l in res.read_text().splitlines():
    d = json.loads(l); top = Path(sys.argv[3]) / f"{d['fixture']}.top" if len(sys.argv) > 3 else None
    if not top or not top.exists(): continue
    a = np.loadtxt(top); s2 = np.sort(a[:, 2])[::-1]; W = d['window_others']
    # exponential tail: P(S > s) ~ (k/W) exp(-(s - s_k)/beta), beta = mean excess over s_k of the top k
    sk = s2[k - 1]; beta = float(np.mean(s2[:k - 1] - sk))
    n_above_W = int((s2 >= d['s2_true']).sum())
    exp_above_1e8 = (k * 1e8 / W) * math.exp(-(d['s2_true'] - sk) / beta)
    print(json.dumps({'fixture': d['fixture'], 'window': W, 's2_true': d['s2_true'], 's2_best_other': float(s2[0]), 's2_kth': float(sk), 'beta': round(beta, 2),
                      'above_true_in_window': n_above_W, 'expected_above_true_at_1e8': round(exp_above_1e8, 3)}))
