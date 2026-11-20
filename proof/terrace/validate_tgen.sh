#!/bin/zsh
# tgen must reproduce GEN's gen exactly (fnv64, state_out, steps) when the TERRACE keys are unset.
cd $(git rev-parse --show-toplevel)/proof/terrace
for V in U6s U4s "U6s:K=256" "U6s:pick=xzy,sweeps=8" "U6r:sweeps=8" "U6sm:sweeps=4"; do
  for S in 0x123456789AB 0x5DEECE66D; do
    a=$(../gen/gen run "$V" $S - | ~/.venvs/arg/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['fnv64'], d['state_out'], d['steps'])")
    b=$(./tgen run "$V" $S - | ~/.venvs/arg/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['fnv64'], d['state_out'], d['steps'])")
    [[ "$a" == "$b" ]] && echo "OK   $V $S $a" || echo "DIFF $V $S gen=$a tgen=$b"
  done
done
