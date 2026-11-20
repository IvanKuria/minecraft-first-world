#!/bin/zsh
# trimmed queue (time box): replaces run_fx3.sh after the K4096 batch.
cd $(git rev-parse --show-toplevel)/proof/likelihood
PY=~/.venvs/arg/bin/python; export LS_STRIDE=4
R=results/fxrank-Vfull-K4096-nt8-scan3.5-test.jsonl
while [[ $(wc -l < $R) -lt 8 ]]; do sleep 5; done
pkill -f run_fx3.sh; sleep 1; pkill -f "fx_rank.py"; pkill -f "lscore rank"; sleep 1
FX_ONLY=fx000 $PY fx_rank.py Vfull-K1024-nt8-scan5 fh0s1024 5.0 2000 3 test 500000 > fx-rank-k1024.log 2>&1
FX_ONLY=fx001,fx002,fx003,fx004,fx005 $PY fx_rank.py Vfull-K1024-nt8-scan5 fh0s1024 5.0 2000 3 test 50000 >> fx-rank-k1024.log 2>&1
$PY fx_rank.py Vfull-K4096-nt8-rand16 fh0r4096 16.0 2000 3 test 50000 > fx-rank-r16.log 2>&1
FX_SUFFIX=-shift1 FX_ONLY=fx002,fx003 $PY fx_rank.py Vfull-K4096-nt8-scan3.5 fh0s4096 3.5 2000 3 test 50000 1.0 1 > fx-rank-k4096-shift.log 2>&1
echo Q4_DONE >> fx-rank-k4096-shift.log
