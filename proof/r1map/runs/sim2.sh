#!/bin/zsh
# R1MAP two-view synthetic power study (r1sim sim): 40 worlds x 200 spawn pairs per run; 4 processes in parallel,
# <60 MB RAM each.  Independent rigid registration shifts per map ~ N(0, sigma2), N(0, sigma1).
cd $(git rev-parse --show-toplevel)/proof/r1map
run() { ./r1sim sim "$1" 40 2000 "$2" "$3" 4 "$4" 200 > "runs/sim2-$5.jsonl" 2> "runs/sim2-$5.err"; }
run U6s 1 1 401 U6s_s11 &
run U6s 0 0 402 U6s_s00 &
run U6s 2 2 403 U6s_s22 &
run "U6s:K=256" 1 1 404 U6sK256_s11 &
wait
run "U6s:K=4096" 1 1 405 U6sK4096_s11 &
run U6r 1 1 406 U6r_s11 &
run "U6s:ntypes=12" 1 1 407 U6snt12_s11 &
run U4s 1 1 408 U4s_s11 &
wait
