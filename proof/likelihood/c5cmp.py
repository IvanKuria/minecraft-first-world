"""C5 statistics vs the likelihood on the SAME rendered fixtures: planted seed's C5 (cent hits, Voronoi IoU at the
corrected surface layer y = 63.5) and pass/fail at the PC's operating points, plus null pass rates and the likelihood
z/max over <nnull> random 48-bit states (lscore synth, full scorer S=1).
usage: c5cmp.py <variant-key> <model-family> <T> <nnull> <threads> [tag=test]"""
import json, math, sys, subprocess
from pathlib import Path
import lobsio as O
HERE = Path(__file__).resolve().parent
key, fam, T, nnull, nth = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); tag = sys.argv[6] if len(sys.argv) > 6 else 'test'
FX = HERE / 'fixtures' / key; CF = json.loads((FX / 'conf.json').read_text())
work = HERE / 'work' / f'fx-{key}-{tag}'; prm = work / 'params.txt'
out = (HERE / 'results' / f'c5cmp-{key}-{tag}.jsonl').open('w')
for f in sorted((FX / tag).glob('fx*.json')):
    rec = json.loads(f.read_text()); maps = []
    for w in ('r1', 'r2'):
        m = rec['maps'][w]; mp = work / f'{f.stem}-{w}.map'
        if not mp.exists(): O.write_map(mp, 1 if w == 'r1' else 2, O.fixture_classes(m), CF['conf'][w], CF['lnull'][w], m['O'], m['C'])
        maps.append(str(mp))
    p1, p2 = rec['maps']['r1']['pos'], rec['maps']['r2']['pos']
    r = subprocess.run([str(HERE / 'lscore'), 'synth', str(prm), ','.join(maps), str(rec['G0']), *[repr(x) for x in (p1[0], p1[1], p2[0], p2[1])], str(nnull), '77', '1', '1.0', str(nth)],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    tc = d['true_c5']   # [hits, iou, ncomp] per map (r1, r2)
    pts = {'c3,3|i.5,.4': (3, 3, .5, .4), 'c3,3|i.3,.3': (3, 3, .3, .3), 'c2,2|i.3,.3': (2, 2, .3, .3)}
    d['true_c5_pass'] = {k: tc[0][0] >= a and tc[1][0] >= b and tc[0][1] >= i1 and tc[1][1] >= i2 for k, (a, b, i1, i2) in pts.items()}
    d['fixture'] = f.stem; out.write(json.dumps(d) + '\n'); out.flush()
    print(f.stem, 'LLR true %.1f null max %.1f (n=%d, ge=%d)' % (d['true_llr'], d['null_max'], d['nnull'], d['null_ge_true']),
          '| C5 true r1 %d/%.2f r2 %d/%.2f' % (tc[0][0], tc[0][1], tc[1][0], tc[1][1]), d['true_c5_pass'], 'C5 null pass', d['c5_null_pass'], flush=True)
