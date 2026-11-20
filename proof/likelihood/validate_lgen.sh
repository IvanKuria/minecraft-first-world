#!/bin/zsh
# lgen must reproduce tgen exactly (fnv64, state_out, steps), with and without the arrival/raw dumps.
cd $(git rev-parse --show-toplevel)/proof/likelihood
PY=~/.venvs/arg/bin/python
j() { $PY -c "import json,sys; d=json.load(sys.stdin); print(d['fnv64'], d['state_out'], d['steps'])"; }
for V in U6s "U6r:sweeps=8" "U6sm:sweeps=4" \
         "init=unset,ntypes=6,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,unsetfill=0,margin=1,cap=fixed,capy=42,shell=1,M=18874368" \
         "init=unset,ntypes=8,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,M=20971520,unsetfill=0,margin=0,cap=none,capy=63,shell=1" \
         "init=unset,ntypes=8,rng=int,pick=xyz,nb=rand,rule=mono,K=4096,M=31457280,unsetfill=0,margin=1,cap=fixed,capy=62,shell=1"; do
  for S in 0x123456789AB 0x5DEECE66D; do
    a=$(../terrace/tgen run "$V" $S - | j)
    b=$(./lgen run "$V" $S - | j)
    c=$(LGEN_ARRIVAL=/tmp/lgen_val.arr LGEN_RAW=/tmp/lgen_val.raw ./lgen run "$V" $S - | j)
    [[ "$a" == "$b" && "$a" == "$c" ]] && echo "OK   $V $S $a" || echo "DIFF $V $S tgen=$a lgen=$b lgen+dump=$c"
  done
done
rm -f /tmp/lgen_val.arr /tmp/lgen_val.raw
