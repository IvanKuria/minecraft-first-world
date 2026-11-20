#!/bin/zsh
# queue 2: after two 1e6-window K4096 fixtures, switch to 1e5 windows for the rest + other variants + shift robustness.
cd $(git rev-parse --show-toplevel)/proof/likelihood
PY=~/.venvs/arg/bin/python; export LS_STRIDE=4
R=results/fxrank-Vfull-K4096-nt8-scan3.5-test.jsonl
while [[ ! -f $R || $(wc -l < $R) -lt 2 ]]; do sleep 10; done
pkill -f run_fx.sh; pkill -f "fx_rank.py Vfull-K4096"; pkill -f "lscore rank"; sleep 2
FX_ONLY=fx002,fx003,fx004,fx005,fx006,fx007 $PY fx_rank.py Vfull-K4096-nt8-scan3.5 fh0s4096 3.5 2000 3 test 50000 >> fx-rank-k4096.log 2>&1
while [[ $(ls fixtures/Vfull-K1024-nt8-scan5/test/fx*.json 2>/dev/null | wc -l) -lt 6 ]]; do sleep 10; done
FX_ONLY=fx000,fx001 $PY fx_rank.py Vfull-K1024-nt8-scan5 fh0s1024 5.0 2000 3 test 500000 > fx-rank-k1024.log 2>&1
FX_ONLY=fx002,fx003,fx004,fx005 $PY fx_rank.py Vfull-K1024-nt8-scan5 fh0s1024 5.0 2000 3 test 50000 >> fx-rank-k1024.log 2>&1
FX_SUFFIX=-shift1 FX_ONLY=fx002,fx003,fx004,fx005 $PY fx_rank.py Vfull-K4096-nt8-scan3.5 fh0s4096 3.5 2000 3 test 50000 1.0 1 > fx-rank-k4096-shift.log 2>&1
while [[ $(ls fixtures/Vfull-K4096-nt8-rand16/test/fx*.json 2>/dev/null | wc -l) -lt 4 ]]; do sleep 10; done
$PY fx_rank.py Vfull-K4096-nt8-rand16 fh0r4096 16.0 2000 3 test 50000 > fx-rank-r16.log 2>&1
echo Q3_DONE >> fx-rank-r16.log
