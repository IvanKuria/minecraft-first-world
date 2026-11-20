#!/bin/zsh
# Light CPU jobs (<100 MB RSS each): fixture rendering (1 process) and lscore ranking (3 threads).
cd $(git rev-parse --show-toplevel)/proof/likelihood
PY=~/.venvs/arg/bin/python
export LS_STRIDE=4
# wait for the K4096 test fixtures
while [[ $(ls fixtures/Vfull-K4096-nt8-scan3.5/test/fx*.json 2>/dev/null | wc -l) -lt 8 ]]; do sleep 10; done
$PY fx_rank.py Vfull-K4096-nt8-scan3.5 fh0s4096 3.5 2000 3 test 500000 > fx-rank-k4096.log 2>&1
echo RANK_K4096_DONE >> fx-rank-k4096.log
