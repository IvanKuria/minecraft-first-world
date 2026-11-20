#!/bin/zsh
# Synthetic observation batch (one process at a time, ~60 MB RAM).  usage: run_synth.sh <nspawn> <world...>
cd $(git rev-parse --show-toplevel)/proof/terrace
n=$1; shift
for w in "$@"; do
  tag=$(basename $w .blocks)
  [[ -s synth/$tag.json ]] && continue
  ~/.venvs/arg/bin/python synth.py $w $n 7 synth/$tag.json top > synth/$tag.log 2>&1
done
