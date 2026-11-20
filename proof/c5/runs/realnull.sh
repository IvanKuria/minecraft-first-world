#!/bin/zsh
# Null (random-seed) pass rates of the C5 early test on the REAL second-reset map: a 4e6-seed slice of the first
# PM-24h candidate row's clock window (q range is arbitrary; the slice is used only as a source of random G0).
# Logs every seed with >= 3 centroid hits and IoU >= 0.30 (hits, IoU).  Not a search: no regrowth, no scoring.
cd $(git rev-parse --show-toplevel)/proof/c5
read s c k q x z < <(head -1 cands-PM-24h.txt)
lo=$((q - 40*3579545))
head -1 cands-PM-24h.txt > runs/row0.txt
for V in U6s "U6s:K=256" "U6s:ntypes=12" U4s; do
  ./c5search "$V" runs/row0.txt footage.obs 3579545 10 40 1 3 0.30 6 $lo $((lo+4000000)) > "runs/realnull-${V//[:=]/_}.txt"
done
./c5search "U6s:K=4096" runs/row0.txt footage.obs 3579545 10 40 1 3 0.30 6 $lo $((lo+1000000)) > "runs/realnull-U6s_K_4096.txt"
