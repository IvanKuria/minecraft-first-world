"""Run the frozen link/null_for_hit.py on Windows: only preloads raycast.dll into scorer._lib and points
replay_driver.GEN at gen.exe. Usage: python null_pc.py RESULTS.jsonl KEY"""
import sys
from pathlib import Path
import pc_replay  # sets sys.path to link/, imports frozen scorer
pc_replay.load_dll()
import replay_driver
replay_driver.GEN = pc_replay.GEN
import null_for_hit
null_for_hit.main(sys.argv[1], sys.argv[2])
