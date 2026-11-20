"""Joint null tail of (centroid hits, IoU) for random generator seeds against the REAL second-reset map."""
import glob, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
res = {}
for f in sorted(glob.glob(str(HERE / 'runs/realnull-*.txt'))):
    lines = open(f).read().splitlines()
    if not lines or not lines[-1].startswith('SUMMARY'): continue
    tested = int(lines[-1].split('tested=')[1].split()[0])
    s1 = int(lines[-1].split('pass_stage1=')[1].split()[0])
    P = [(int(l.split()[5]), float(l.split()[6])) for l in lines if l.startswith('PASS')]
    tab = {}
    for c in (3, 4, 5, 6):
        for t in (0.3, 0.4, 0.5, 0.6):
            n = sum(1 for h, i in P if h >= c and i >= t)
            tab[f'cent>={c},iou>={t}'] = {'n': n, 'rate': n / tested, 'upper95_if_0': 3 / tested if n == 0 else None}
    res[Path(f).stem] = {'tested': tested, 'stage1_cent>=3_rate': s1 / tested, 'max_iou': max((i for _, i in P), default=None), 'joint': tab}
(HERE / 'realnull-summary.json').write_text(json.dumps(res, indent=1) + '\n')
for k, v in res.items():
    print(k, 'tested', v['tested'], 'cent>=3', f"{v['stage1_cent>=3_rate']:.2e}", 'maxIoU', v['max_iou'])
    for c in (4, 5, 6):
        print('   ', '  '.join(f"c{c} i{t}: {v['joint'][f'cent>={c},iou>={t}']['rate']:.1e}" for t in (0.3, 0.4, 0.5, 0.6)))
