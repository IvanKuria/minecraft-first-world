#!/bin/zsh
# Batch 4: "slab shell" model: rd-132211 slab border (walls + floor rock, y<=42) kept, interior unset, C4 growth with a
# fixed iteration count, leftover interior -> AIR.  -> grid4-surf.jsonl
cd $(git rev-parse --show-toplevel)/proof/terrace
SNAP="1v/1.5v/2v/2.5v/3v/3.5v/4v/4.5v/5v/6v/7v/8v/10v/12v/15v/20v/25v/30v/40v/50v/64v"
run() { TGEN_SNAP=worlds/$1 ./tgen run "$2,M=268435456,snap=$SNAP" $3 - > worlds/$1.json 2> worlds/$1.snaps.jsonl; OUT=grid4-surf.jsonl ~/.venvs/arg/bin/python surfsnaps.py "$1" "$2" "$3"; }
for S in 0x123456789AB 0x5DEECE66D; do
  for K in 256 1024 4096; do
    run shell_s_K${K}_${S} "init=unset,ntypes=6,rng=int,pick=xyz,nb=scan,rule=mono,K=$K,unsetfill=0,margin=1,cap=fixed,capy=42,shell=1" $S
  done
  for K in 1024 4096; do
    run shell_r_K${K}_${S} "init=unset,ntypes=6,rng=int,pick=xyz,nb=rand,rule=mono,K=$K,unsetfill=0,margin=1,cap=fixed,capy=42,shell=1" $S
  done
done
