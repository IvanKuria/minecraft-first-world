#!/bin/zsh
# C5 synthetic power study (c5sim2: exact bucketed Voronoi, centroid statistic). 4 processes in parallel, <50 MB RAM each.
cd $(git rev-parse --show-toplevel)/proof/c5
run() { ./c5sim2 sim "$1" "$2" "$3" "$4" 4 32 "$5" 2 > "runs/$6.jsonl" 2> "runs/$6.err"; }
run U6s 16 200000 0 201 A_U6s_s0 &
run U6s 16 200000 1 202 B_U6s_s1 &
run U6s 16 200000 2 203 C_U6s_s2 &
run "U6s:K=256" 16 200000 1 204 D_U6sK256_s1 &
wait
run "U6s:ntypes=12" 16 200000 1 205 E_U6s_nt12_s1 &
run U4s 16 200000 1 206 H_U4s_s1 &
run "U6s:K=4096" 12 60000 1 207 G_U6sK4096_s1 &
run U6r 12 60000 1 208 F_U6r_s1 &
wait
