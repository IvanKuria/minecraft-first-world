"""Truth-level T scan of the full-height surface: global column-depth histogram (interior, 8-block border excluded) and
wall/edge checks, per family and T.  -> tscan.jsonl"""
import sys, json, numpy as np
import lworld as Lw
fams = sys.argv[1].split(','); Ts = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
DEF = {'fh0s1024': np.arange(3, 8.01, .25), 'fh0s4096': np.arange(2, 6.01, .25), 'fh0s256': np.arange(4, 11.01, .25),
       'fh0r4096': np.arange(10, 30.01, 1), 'fh0r1024': np.arange(16, 50.01, 2), 'fh1s1024': np.arange(3, 8.01, .25),
       'fh1r4096': np.arange(10, 30.01, 1)}
out = open('tscan.jsonl', 'a')
for fam in fams:
    ytop = 62 if fam.startswith('fh1') else 63
    for s in 'abcd':
        raw, arr, seeds, meta = Lw.load(fam + s)
        for T in (Ts or DEF[fam]):
            w = Lw.world_at(raw, arr, int(T * Lw.V))
            t = Lw.tops(w)
            dep = ytop - t[8:-8, 8:-8]
            h = np.bincount(np.minimum(dep, 10).ravel(), minlength=11) / dep.size
            # world edge: wall top must equal the plain (rows x=0 / x=255 / z=0 / z=255 columns), floor solid
            edge = np.concatenate([t[:, 0], t[:, 255], t[0, :], t[255, :]])
            floor = (w[0] != 0).mean()
            # near-edge plain (2..6 blocks from the wall) solid-top fraction
            ne = np.concatenate([dep[:, :0], ]) if False else None
            t2 = ytop - t
            band = np.concatenate([t2[2:-2, 2:7].ravel(), t2[2:-2, -7:-2].ravel(), t2[2:7, 2:-2].ravel(), t2[-7:-2, 2:-2].ravel()])
            rec = {'fam': fam, 'world': fam + s, 'T': float(T), 'hist': [round(float(x), 4) for x in h],
                   'wall_top_eq_plain': float((edge == ytop).mean()), 'floor_solid': float(floor), 'nearedge_f0': float((band == 0).mean())}
            out.write(json.dumps(rec) + '\n')
    print(fam, 'done', flush=True)
