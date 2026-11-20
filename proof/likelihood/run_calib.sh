#!/bin/zsh
cd $(git rev-parse --show-toplevel)/proof/likelihood
PY=~/.venvs/arg/bin/python
$PY calib2.py fh0r4096 13,15,17,19 > calib2-fh0r4096.json 2>&1
$PY calib2.py fh0r1024 20,23,26,29 > calib2-fh0r1024.json 2>&1
$PY calib2.py fh1s1024 4,4.5,5,5.5,6 62 > calib2-fh1s1024.json 2>&1
$PY calib2.py fh1r4096 13,15,17,19 62 > calib2-fh1r4096.json 2>&1
echo CALIB_DONE
