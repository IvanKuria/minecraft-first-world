"""All-rock calibration of the likelihood lane's model (calib2.py, unchanged) on the PC.
Generates lgen worlds (arrival/raw/seed dumps, 4 states a-d) for the best all-rock families of the K-ladder fit, then runs
likelihood/calib2.py <family> <Ts> <ytop> and saves likelihood/calib2-<family>.json. lworld.LGEN is pointed at lgen.exe
(MSVC build of the same lgen.c; bit-identical to tgen on the all-rock spec)."""
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; LK = HERE.parent / 'likelihood'
sys.path.insert(0, str(LK))
import lworld as L
L.LGEN = LK / 'lgen.exe'
V = L.V
STATES = {'a': '0x123456789AB', 'b': '0x5DEECE66D', 'c': '0x2545F4914F6C', 'd': '0x1F123BB5159A'}
BASE = 'init=unset,ntypes=1,rng=int,pick=xyz,nb={nb},rule=mono,K={K},unsetfill=0,cap=fixed,capy=63,shell={sh},{form}'
FORM = {'V': 'margin=0', '3': 'margin=1,sy=63,sy0=1,py=63,py0=1', '2': 'margin=1'}
# family tag (calib2 reads fam[3] == 'r' as random-neighbour): rk<form><rule><K>[n = no shell]
FAMS = {'rkVs300': ('V', 'scan', 300, 1, 10, '5,5.5,6,6.5,7', 63), 'rkVs512': ('V', 'scan', 512, 1, 10, '5,5.5,6,6.5,7', 63),
        'rk3s256': ('3', 'scan', 256, 1, 10, '5,5.5,6,6.5,7', 63), 'rk2s256': ('2', 'scan', 256, 1, 10, '5,5.5,6,6.5,7', 62),
        'rkVr600n': ('V', 'rand', 600, 0, 36, '20,22.5,25,27.5,30', 63)}
for fam in (sys.argv[1].split(',') if len(sys.argv) > 1 else FAMS):
    form, nb, K, sh, sw, Ts, ytop = FAMS[fam]
    for s in 'abcd':
        if (LK / 'worlds' / f'{fam}{s}.json').exists(): continue
        t0 = time.time()
        m = L.run(f'{fam}{s}', BASE.format(nb=nb, K=K, sh=sh, form=FORM[form]), STATES[s], int(sw * V))
        print(json.dumps({'tag': f'{fam}{s}', 'sec': round(time.time() - t0, 1), 'unset_left': m['unset_left']}), flush=True)
    out = LK / f'calib2-{fam}.json'
    if not out.exists():
        r = subprocess.run([sys.executable, str(LK / 'calib2.py'), fam, Ts, str(ytop)], capture_output=True, text=True, cwd=str(LK))
        if r.returncode: print(r.stderr[-1500:], flush=True); continue
        out.write_text(r.stdout)
    d = json.loads(out.read_text())
    print(json.dumps({'family': fam, 'params': d['params'], 'heldout_bits': [(h['T'], round(h['bits_per_col'], 3)) for h in d['heldout']]}), flush=True)
