#!/bin/zsh
# C5 synthetic power study, continuation (batch2 stopped after A6/B6/C7/D11 worlds to fit the time box).
cd $(git rev-parse --show-toplevel)/proof/c5
run() { ./c5sim2 sim "$1" "$2" "$3" "$4" 4 32 "$5" 2 > "runs/$6.jsonl" 2> "runs/$6.err"; }
run U6s 8 100000 0 301 A2_U6s_s0 &
run U6s 8 100000 1 302 B2_U6s_s1 &
run "U6s:ntypes=12" 10 100000 1 305 E_U6s_nt12_s1 &
run U4s 10 100000 1 306 H_U4s_s1 &
wait
run "U6s:K=4096" 8 30000 1 307 G_U6sK4096_s1 &
run U6r 8 30000 1 308 F_U6r_s1 &
run U6s 8 100000 2 303 C2_U6s_s2 &
run "U6s:K=256" 6 100000 1 304 D2_U6sK256_s1 &
wait
