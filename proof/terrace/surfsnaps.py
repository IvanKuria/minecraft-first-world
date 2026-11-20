import json, sys, os
sys.path.insert(0, '.')
import surf
tag, spec, S = sys.argv[1], sys.argv[2], sys.argv[3]
V = 256 * 256 * 64
with open(os.environ.get('OUT', 'grid-surf.jsonl'), 'a') as out:
    for line in open(f'worlds/{tag}.snaps.jsonl'):
        d = json.loads(line)
        b = surf.load(d['file'])
        st, t = surf.stats(b)
        rec = {'tag': tag, 'spec': spec, 'state': S, 'T_over_V': round(d['snap_iterations'] / V, 4), 'iters': d['snap_iterations'],
               'steps_so_far': d['steps_so_far'], 'rejections': d['rejections_so_far'], 'unset_left': d['unset_left'],
               'bottom_solid': round(float((b[0] != 0).mean()), 4),
               'edge_col_top_eq_mode': round(float(((surf.tops(b)[0, :] == st['mode']).mean() + (surf.tops(b)[:, 0] == st['mode']).mean()) / 2), 4),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}}
        out.write(json.dumps(rec) + '\n')
        os.remove(d['file'])
