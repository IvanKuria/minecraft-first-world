"""Bit-for-bit check of the CUDA port (build/c5gpu.exe) against the CPU reference (build/c5search.exe).
(1) the 4 planted-seed fixtures written by validate_search_pc.py (same args the CPU run used);
(2) real footage map, real candidate rows, a 1e6-seed q block per row, at the operating point and at a loose point.
PASS sets must be identical (same cand, j, q_g, G0, centHit, iou string) and SUMMARY lines identical."""
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
B = HERE / 'build'; FX = HERE / 'fixtures'; C5 = HERE.parent / 'c5'


def run(exe, args):
    t = time.time()
    r = subprocess.run([str(B / exe), *args], capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    return sorted(l for l in lines if l.startswith('PASS')), [l for l in lines if l.startswith('SUMMARY')][0], time.time() - t, r.stderr.strip()


def compare(name, args):
    a = run('c5search.exe', args); b = run('c5gpu.exe', args)
    ok = a[0] == b[0] and a[1] == b[1]
    rec = {'case': name, 'identical': ok, 'cpu_pass': len(a[0]), 'gpu_pass': len(b[0]), 'cpu_summary': a[1], 'gpu_summary': b[1],
           'cpu_s': round(a[2], 2), 'gpu_s': round(b[2], 2), 'gpu_stderr': b[3]}
    print(json.dumps(rec), flush=True)
    return ok


allok = True
for t in range(4):
    v, F, cmin, imin, lo, hi = json.loads((FX / f'w{t}.args').read_text())
    allok &= compare(f'planted{t}', [v, str(FX / f'w{t}.cands'), str(FX / f'w{t}.obs'), F, '10', '40', '1', cmin, imin, '32', lo, hi])
rows = [l.split() for l in (C5 / 'cands-PIT-24h.txt').read_text().split('\n') if l.strip()]
for i, r in enumerate(rows[:3]):
    one = HERE / f'fixtures/real-row{i}.cands'; one.write_text(' '.join(r) + '\n')
    qm = int(r[3]); lo = qm - int(25 * 1193182); hi = lo + 999_999
    for var, cmin, imin in [('U6s:K=1024,ntypes=8', '4', '0.5'), ('U6s:K=4096,ntypes=8', '2', '0.2')]:
        allok &= compare(f'real-row{i}-{var}-c{cmin}-i{imin}', [var, str(one), str(C5 / 'footage.obs'), '1193182', '10', '40', '2', cmin, imin, '32', str(lo), str(hi)])
print(json.dumps({'all_identical': allok}))
