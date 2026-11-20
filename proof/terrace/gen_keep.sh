#!/bin/zsh
# gen_keep.sh <tag> <spec> <state> <snaps in units of V, e.g. 4/5/6>  -> worlds/<tag>.<iters>.blocks
cd $(git rev-parse --show-toplevel)/proof/terrace
spec=$(~/.venvs/arg/bin/python -c "
import sys; V=4194304; s=[int(float(x)*V+.5) for x in sys.argv[1].split('/')]; print(max(s), '/'.join(map(str,s)))" $4)
max=${spec%% *}; snaps=${spec#* }
TGEN_SNAP=worlds/$1 ./tgen run "$2,M=$max,snap=$snaps" $3 - > worlds/$1.json 2> worlds/$1.snaps.jsonl
