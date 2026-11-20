"""GPU stage-1 (lkernel) throughput on a planted fixture's real registration: K4096 nt8 params and an all-rock K=300 copy."""
import json, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; L = HERE.parent / 'likelihood'
key = 'Vfull-K4096-nt8-scan3.5'; W = L / 'work' / f'fx-{key}-test'
r = json.loads((L / 'fixtures' / key / 'test' / 'fx000.json').read_text())
pos = [*r['maps']['r1']['pos'], *r['maps']['r2']['pos']]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
res = {}
for name, prm in [('K4096-nt8', W / 'params.txt'), ('allrock-K300', HERE / 'lk-validate' / 'params-rock300.txt')] + \
        [(Path(p).stem, Path(p)) for p in sys.argv[2:]]:
    t = time.time()
    out = subprocess.run([str(HERE / 'build' / 'lkernel.exe'), str(prm), f"{W / 'fx000-r1.map'},{W / 'fx000-r2.map'}", 'lit',
                          str(r['L'] - n // 2), str(r['L'] + n // 2 - 1), *map(repr, pos), '1e30'], capture_output=True, text=True,
                         env={**os.environ, 'LS_STRIDE': '3'}).stdout.strip().splitlines()
    dt = time.time() - t
    res[name] = {'seeds': n, 'seconds': round(dt, 2), 'seeds_per_s': n / dt, 'summary': out[-1] if out else None}
    print(name, res[name], flush=True)
(HERE / 'lk-validate' / 'bench.json').write_text(json.dumps(res, indent=1))
