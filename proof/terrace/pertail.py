"""Per-spawn tail test: how often does a synthetic near field look as terraced as the footage?
statistic s = fraction of near-field labels at k in {-1,-2,-3} (shallow depressions); footage r1, r2 values."""
import json, glob, sys
from pathlib import Path
import compare as CP
thr = sys.argv[1] if len(sys.argv) > 1 else '0.35'
F = CP.footage(thr)
fs = {w: F[w]['bins']['-1'] + F[w]['bins']['-2'] + F[w]['bins']['-3'] for w in ('r1', 'r2')}
f2 = {w: F[w]['bins']['-2'] for w in ('r1', 'r2')}
print('footage shallow-depression label fraction: r1 %.3f  r2 %.3f   (-2 only: r1 %.3f r2 %.3f)' % (fs['r1'], fs['r2'], f2['r1'], f2['r2']))
print('%-22s %6s %6s %6s %6s %6s  %s' % ('world', 'mean', 'p50', 'p90', 'max', '#>=fp', 'P(r1-like & r2-like) est'))
for p in sorted(glob.glob(str(CP.HERE / 'synth/*.json'))):
    d, agg, tru, per = CP.synth(p, thr)
    vals = {w: [(b['-1'] + b['-2'] + b['-3']) if b else 0.0 for b in per[w]] for w in ('r1', 'r2')}
    allv = sorted(vals['r1'] + vals['r2'])
    n = len(allv)
    lo = min(fs.values())
    ge = sum(1 for v in allv if v >= lo)
    p1 = sum(1 for v in vals['r1'] if v >= fs['r1']) / len(vals['r1']); p2 = sum(1 for v in vals['r2'] if v >= fs['r2']) / len(vals['r2'])
    print('%-22s %6.3f %6.3f %6.3f %6.3f %3d/%-3d  %.3f' % (Path(p).stem, sum(allv) / n, allv[n // 2], allv[int(.9 * n)], allv[-1], ge, n, p1 * p2))
