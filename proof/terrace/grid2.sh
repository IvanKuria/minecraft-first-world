#!/bin/zsh
# Batch 2: true-surface statistics on a fine T grid (no files kept).  Writes grid2-surf.jsonl.
cd $(git rev-parse --show-toplevel)/proof/terrace
SNAP="1v/1.5v/2v/2.5v/3v/3.25v/3.5v/3.75v/4v/4.25v/4.5v/4.75v/5v/5.5v/6v/7v/8v/10v/12v/15v/20v/30v/40v/50v/64v"
run() {  # tag spec state
  TGEN_SNAP=worlds/$1 ./tgen run "$2,M=268435456,snap=$SNAP" $3 - > worlds/$1.json 2> worlds/$1.snaps.jsonl
  OUT=grid2-surf.jsonl ~/.venvs/arg/bin/python surfsnaps.py "$1" "$2" "$3"
}
for S in 0x123456789AB 0x5DEECE66D; do
  for K in 256 1024 4096; do
    for P in 44 45 46; do
      # bottom-safe H-front: seeds y in [1,41], picks y in [1,P-1], leftover unset -> rock iff y<=42
      run hfs_K${K}_P${P}_${S} "init=unset,ntypes=6,rng=int,pick=xyz,nb=scan,rule=mono,K=$K,margin=1,sy=41,sy0=1,py=$((P-1)),py0=1,fill=42" $S
    done
  done
  for K in 1024 4096; do run hfr_K${K}_P45_${S} "init=unset,ntypes=6,rng=int,pick=xyz,nb=rand,rule=mono,K=$K,margin=1,sy=41,sy0=1,py=44,py0=1,fill=42" $S; done
  for NT in 4 8; do run hfs_nt${NT}_K1024_P45_${S} "init=unset,ntypes=$NT,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,margin=1,sy=41,sy0=1,py=44,py0=1,fill=42" $S; done
  # literal lead H-front (seeds nextInt(42), picks nextInt(P), no y offset)
  for P in 43 44 45 48 64; do run hfl_K1024_P${P}_${S} "init=unset,ntypes=6,rng=int,pick=xyz,nb=scan,rule=mono,K=1024,margin=1,sy=42,py=$P,fill=42" $S; done
  # sky erosion: GEN U-family (cap 42) but cells above the cap act as air neighbours
  for K in 1024 4096 16384; do run sky_K${K}_${S} "init=unset,ntypes=6,rng=int,pick=xyz,nb=scan,rule=mono,K=$K,unsetfill=1,margin=1,cap=fixed,capy=42,sky=1" $S; done
  run skyr_K4096_${S} "init=unset,ntypes=6,rng=int,pick=xyz,nb=rand,rule=mono,K=4096,unsetfill=1,margin=1,cap=fixed,capy=42,sky=1" $S
done
