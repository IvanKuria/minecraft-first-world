#!/bin/zsh
# H-front grid: one run per (structural variant, RNG state) with T snapshots; true surface stats per snapshot -> grid-surf.jsonl
# usage: grid_surf.sh <tag> <spec-without-M/snap> <state> [keep]
cd $(git rev-parse --show-toplevel)/proof/terrace
tag=$1; spec=$2; S=$3; keep=$4
SNAP="1v/1.5v/2v/2.5v/3v/4v/5v/6v/8v/10v/12v/15v/20v/25v/30v/40v/50v"
TGEN_SNAP=worlds/$tag ./tgen run "$spec,M=209715200,snap=$SNAP" $S - > worlds/$tag.json 2> worlds/$tag.snaps.jsonl
~/.venvs/arg/bin/python - "$tag" "$spec" "$S" "$keep" <<'PY'
import json, sys, os
sys.path.insert(0, '.')
import surf
tag, spec, S, keep = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
V = 256 * 256 * 64
with open('grid-surf.jsonl', 'a') as out:
    for line in open(f'worlds/{tag}.snaps.jsonl'):
        d = json.loads(line)
        st, _ = surf.stats(surf.load(d['file']))
        b = surf.load(d['file'])
        bottom_solid = float((b[0] != 0).mean())
        rec = {'tag': tag, 'spec': spec, 'state': S, 'T_over_V': round(d['snap_iterations'] / V, 3), 'iters': d['snap_iterations'],
               'steps_so_far': d['steps_so_far'], 'rejections': d['rejections_so_far'], 'unset_left': d['unset_left'],
               'bottom_solid': round(bottom_solid, 4), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}}
        out.write(json.dumps(rec) + '\n')
        if keep != 'keep': os.remove(d['file'])
PY
