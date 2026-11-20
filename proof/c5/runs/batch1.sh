#!/bin/zsh
# C5 synthetic power study batch 1 (run under heavy-lock; 4 processes in parallel, <50 MB RAM each)
cd $(git rev-parse --show-toplevel)/proof/c5
run() { ./c5sim sim "$1" "$2" "$3" "$4" 4 32 "$5" 2 > "runs/$6.jsonl" 2> "runs/$6.err"; }
run U6s 12 100000 0 101 A_U6s_s0 &
run U6s 12 100000 1 102 B_U6s_s1 &
run "U6s:K=256" 12 100000 0 103 D_U6sK256_s0 &
run "U6s:ntypes=12" 12 100000 0 104 E_U6s_nt12_s0 &
wait
run U6s 12 100000 2 105 F_U6s_s2 &
run U6r 8 30000 0 106 C_U6r_s0 &
run "U6s:K=4096" 8 30000 0 107 G_U6sK4096_s0 &
run "U6s:K=4096" 8 30000 1 108 H_U6sK4096_s1 &
wait
