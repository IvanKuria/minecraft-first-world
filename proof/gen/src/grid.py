"""Run a variant grid: gen -> metrics; append JSON lines.  usage: grid.py out.jsonl spec1 spec2 ..."""
import sys, json, subprocess, os
sys.path.insert(0, os.path.dirname(__file__))
from metrics import metrics, load
out = sys.argv[1]
seed = os.environ.get("GRID_STATE", "0x123456789AB")
for spec in sys.argv[2:]:
    r = json.loads(subprocess.check_output(["./gen", "run", spec, seed, "tmp/grid.blocks"]))
    m = metrics(load("tmp/grid.blocks"))
    row = {"spec": spec, "state": seed, "gen": r, "metrics": m}
    with open(out, "a") as f: f.write(json.dumps(row) + "\n")
    print(spec, "PASS" if m["pass_all"] else "fail", m["T"], "air%.2f mode%d plain%.2f wall%.2f" % (m["interior_air_frac"], m["top_mode_y"], m["frac_cols_top_eq_mode"], m["wall_min_face_solid"]), flush=True)
