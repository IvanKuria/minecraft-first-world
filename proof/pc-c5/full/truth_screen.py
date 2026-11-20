"""Cheap screen of every generated full-height world by its TRUE surface profile (terrace/surf.stats: fractions at the
modal top level / -1 / -2 / -3 / <=-4), compared with what the terrace lane's synthetic observation of its own fitted shell
windows looked like in truth (scan K4096 3.5V: .44/.16/.10/.06/.24; rand K4096 15V: .38/.13/.12/.09/.28; NOTES.txt).
Ranks (form, rule, K, nt, T) by L1 distance of the truth profile to the mean of those two fitted profiles. Provisional
until the synthetic-observation fit (fit_full.py) completes. Output: truth-screen.json"""
import json, glob, sys, re
from pathlib import Path
from multiprocessing import Pool
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'terrace'))
import surf
REF = np.array([(.44 + .38) / 2, (.16 + .13) / 2, (.10 + .12) / 2, (.06 + .09) / 2, (.24 + .28) / 2])
V = 256 * 256 * 64


def one(p):
    try: st, _ = surf.stats(surf.load(p))
    except (FileNotFoundError, ValueError): return None   # already synthesised and deleted -> taken from synth/*.json below
    prof = np.array([st['f0'], st['f-1'], st['f-2'], st['f-3'], st['f_deep(<=-4)']])
    m = re.match(r'(\w+?)_(scan|rand)_K(\d+)_nt(\d+)_s(\d+)\.(\d+)\.blocks', Path(p).name)
    form, rule, K, nt, s, it = m.groups()
    return {'form': form, 'rule': rule, 'K': int(K), 'nt': int(nt), 's': int(s), 'T': round(int(it) / V, 2), 'mode': st['mode'],
            'profile': [round(x, 3) for x in prof], 'L1': round(float(np.abs(prof - REF).sum()), 3)}


if __name__ == '__main__':
    files = sorted(glob.glob(str(HERE / 'worlds' / '*.blocks')))
    with Pool(8) as p: rows = [r for r in p.map(one, files, chunksize=8) if r]
    for j in glob.glob(str(HERE / 'synth' / '*.json')):
        d = json.loads(Path(j).read_text()); st = d['surface']
        base, T = Path(j).stem.rsplit('.T', 1); form, rule, K, nt, s_ = base.split('_')
        prof = np.array([st['f0'], st['f-1'], st['f-2'], st['f-3'], st['f_deep(<=-4)']])
        rows.append({'form': form, 'rule': rule, 'K': int(K[1:]), 'nt': int(nt[2:]), 's': int(s_[1:]), 'T': round(float(T), 2), 'mode': st['mode'],
                     'profile': [round(x, 3) for x in prof], 'L1': round(float(np.abs(prof - REF).sum()), 3)})
    agg = {}
    for r in rows: agg.setdefault((r['form'], r['rule'], r['K'], r['nt'], r['T']), []).append(r)
    out = sorted(({'cfg': k, 'L1': float(np.mean([r['L1'] for r in v])), 'profile': np.mean([r['profile'] for r in v], axis=0).round(3).tolist(),
                   'mode': [r['mode'] for r in v], 'n': len(v)} for k, v in agg.items()), key=lambda a: a['L1'])
    (HERE / 'truth-screen.json').write_text(json.dumps({'ref': REF.tolist(), 'rows': out}, indent=1))
    best = {}
    for a in out:
        k = a['cfg'][:4]
        if k not in best: best[k] = a
    for k, a in sorted(best.items(), key=lambda kv: kv[1]['L1']):
        print(k, 'T=%sV' % a['cfg'][4], 'L1=%.3f' % a['L1'], a['profile'], a['mode'])
