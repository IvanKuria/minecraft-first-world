#!/bin/zsh
# Batch 3: GEN flat-cap family at high seed density (does K alone make shallow depressions?) -> grid3-surf.jsonl
cd $(git rev-parse --show-toplevel)/proof/terrace
SNAP="8v/16v/32v"
run() { TGEN_SNAP=worlds/$1 ./tgen run "$2,M=134217728,snap=$SNAP" $3 - > worlds/$1.json 2> worlds/$1.snaps.jsonl; OUT=grid3-surf.jsonl ~/.venvs/arg/bin/python surfsnaps.py "$1" "$2" "$3"; }
for S in 0x123456789AB; do
  for K in 16384 65536 262144; do
    for NT in 3 4 6; do
      run u_nt${NT}_K${K}_${S} "init=unset,ntypes=$NT,rng=int,pick=xyz,nb=scan,rule=mono,K=$K,unsetfill=1,margin=1,cap=fixed,capy=42" $S
    done
  done
done
