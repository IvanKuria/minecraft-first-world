#!/bin/zsh
# R1MAP: null (random-seed) pass rates of the C5 early test on BOTH real maps (second reset footage.obs, first reset
# footage-r1.obs), same 4e6-seed slice of the first PM-24h row's clock window as c5/runs/realnull.sh (source of random
# G0 only; not a search).  C5_ANY=1 logs any seed passing (cent>=3, IoU>=0.30) on EITHER map, with both maps' values.
cd $(git rev-parse --show-toplevel)/proof/r1map
C5=../c5
read s c k q x z x1 z1 < runs/row0-r12.txt
lo=$((q - 40*3579545))
for V in U6s "U6s:K=256" "U6s:ntypes=12" U4s; do
  C5_ANY=1 $C5/c5search "$V" runs/row0-r12.txt $C5/footage.obs,footage-r1.obs 3579545 10 40 1 3 0.30 6 $lo $((lo+4000000)) > "runs/realnull2-${V//[:=]/_}.txt"
done
C5_ANY=1 $C5/c5search "U6s:K=4096" runs/row0-r12.txt $C5/footage.obs,footage-r1.obs 3579545 10 40 1 3 0.30 6 $lo $((lo+1000000)) > "runs/realnull2-U6s_K_4096.txt"
C5_ANY=1 C5_RANDNULL=1 $C5/c5search U6s runs/row0-r12.txt $C5/footage.obs,footage-r1.obs 3579545 10 40 1 3 0.30 6 $lo $((lo+4000000)) > "runs/realnull2-randomG0-U6s.txt"
