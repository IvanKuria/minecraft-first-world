#!/bin/zsh
# run_one.sh <tag> <variant> <state> [snapspec]  -> worlds/<tag>.<iters>.blocks (+ final worlds/<tag>.final.blocks), json to worlds/<tag>.json
cd $(git rev-parse --show-toplevel)/proof/terrace
tag=$1; V=$2; S=$3
TGEN_SNAP=worlds/$tag ./tgen run "$V" $S worlds/$tag.final.blocks > worlds/$tag.json 2> worlds/$tag.snaps.jsonl
