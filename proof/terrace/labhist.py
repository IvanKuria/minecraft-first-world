import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
def labels_from_scores(scores, rmin, margin):
    lab = {}
    for x, z, s in scores:
        it = sorted(((int(k), v) for k, v in s.items()), key=lambda kv: -kv[1][0])
        if it and it[0][1][0] >= rmin and (len(it) < 2 or it[0][1][0] - it[1][1][0] >= margin): lab[(x, z)] = it[0][0]
    return lab
def hist(lab, lo=-8, hi=3):
    return {k: sum(1 for v in lab.values() if v == k) for k in range(lo, hi + 1)}
if __name__ == '__main__':
    for which in sys.argv[1:]:
        d = json.loads((HERE / f'sweep-{which}.json').read_text())
        for rm, mg in ((.45, .15), (.35, .12), (.3, .1)):
            lab = labels_from_scores(d['scores'], rm, mg)
            print(which, rm, mg, len(lab), hist(lab))
