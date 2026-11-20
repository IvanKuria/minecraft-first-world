#!/bin/zsh
cd $(git rev-parse --show-toplevel)/proof/likelihood
PY=~/.venvs/arg/bin/python
while [[ $(ls fixtures/Vfull-K4096-nt8-scan3.5/test/fx*.json 2>/dev/null | wc -l) -lt 8 ]]; do sleep 10; done
$PY pc_fixtures.py Vfull-K1024-nt8-scan5 10 calib > fx-k1024-calib.log 2>&1
$PY pc_fixtures.py Vfull-K1024-nt8-scan5 6 test > fx-k1024-test.log 2>&1
$PY pc_fixtures.py Vfull-K4096-nt8-rand16 8 calib > fx-r16-calib.log 2>&1
$PY pc_fixtures.py Vfull-K4096-nt8-rand16 4 test > fx-r16-test.log 2>&1
echo FIX2_DONE >> fx-r16-test.log
