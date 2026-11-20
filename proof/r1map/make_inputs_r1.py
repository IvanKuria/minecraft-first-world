"""R1MAP task 4: candidate rows for the two-map search.  Same rows as c5/cands-<timer>-<window>.txt, with the
candidate's own first-observed-reset position (x1, z1) appended (reset_xz of the next reset after the opening one;
zero-gap baseline assumption, identical to how c5/make_inputs.py derives x2, z2).
Output: r1map/cands-<timer>-<window>-r12.txt, lines: s_open calls k_m q_m x2 z2 x1 z1."""
import glob, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'link'))
from lcg import reset_xz  # noqa: E402
summ = {}
for f in sorted(glob.glob(str(HERE.parent / 'c5/cands-*.txt'))):
    rows = []
    fx, fz = [], []
    for l in open(f):
        p = l.split()
        if len(p) < 6: continue
        s = int(p[0]); _, _, s1 = reset_xz(s); x1, z1, s2 = reset_xz(s1); x2, z2, _ = reset_xz(s2)
        assert x2 == float(p[4]) and z2 == float(p[5]), 'x2/z2 mismatch with c5 inputs'
        rows.append(' '.join(p[:6]) + f' {x1!r} {z1!r}'); fx.append(x1 % 1); fz.append(z1 % 1)
    name = Path(f).stem + '-r12.txt'
    (HERE / name).write_text('\n'.join(rows) + '\n')
    summ[name] = {'rows': len(rows), 'x1_frac': [min(fx), max(fx)] if fx else None, 'z1_frac': [min(fz), max(fz)] if fz else None}
(HERE / 'inputs-r12-summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print(json.dumps(summ, indent=1))
