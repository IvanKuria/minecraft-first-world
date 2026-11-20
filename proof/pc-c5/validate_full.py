"""Full-height model: GPU (c5gpu4, X specs with capy=63, surf=63) vs CPU reference (c5search.c with the surface
constant 42.5f -> 63.5f, GEN specs with capy=63 that draw the same seeds) on real rows, two maps and R2 only,
2e6-seed q blocks, at the operating point and a loose point. PASS sets and the counters must be identical
(the SUMMARY's variant string differs by construction and is excluded)."""
import json, subprocess
from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
GPUEXE = os.environ.get('C5_GPUEXE', 'c5gpu4.exe')
B = HERE / 'build'; C5 = HERE.parent / 'c5'; R1 = HERE.parent / 'r1map'
PAIRS = [('X:K=1024,nt=8,m=0,yb=64,tf=0,capy=63,surf=63', 'U6s:K=1024,ntypes=8,margin=0,capy=63'),     # V-full
         ('X:K=4096,nt=8,m=1,yb=62,tf=0,capy=63,surf=63', 'U6s:K=4096,ntypes=8,capy=63')]               # E62
MAPS = [f"{C5 / 'footage.obs'},{R1 / 'footage-r1.obs'}", str(C5 / 'footage.obs')]
rows = [l for l in (R1 / 'cands-PM-78h-r12.txt').read_text().splitlines() if l.startswith('99100327271809 0 2')][:1] + \
       [l for l in (R1 / 'cands-HPET-24h-r12.txt').read_text().splitlines() if l.startswith('266841333641652 0 2')][:1]
ok_all = True
for i, row in enumerate(rows):
    cf = HERE / 'fixtures' / f'full-row{i}.cands'; cf.write_text(row + '\n')
    p = row.split(); f = 3579545 if i == 0 else 14318180
    lo = int(p[3]) - int(25 * f); hi = lo + 1_999_999
    for gx, gc in PAIRS:
        for maps in MAPS:
            for cmin, imin in ([('3,3', '0.5,0.4'), ('2,2', '0.2,0.2')] if ',' in maps else [('4', '0.5'), ('2', '0.2')]):
                args = [str(cf), maps, str(f), '10', '40', '2', cmin, imin, '16', str(lo), str(hi)]
                c = subprocess.run([str(B / 'c5search_s63.exe'), gc, *args], capture_output=True, text=True, check=True).stdout.splitlines()
                g = subprocess.run([str(B / GPUEXE), gx, *args], capture_output=True, text=True, check=True).stdout.splitlines()
                cp = sorted(l for l in c if l.startswith('PASS')); gp = [l for l in g if l.startswith('PASS')]
                strip = lambda s: ' '.join(t for t in s.split() if not t.startswith('variant='))
                same = cp == gp and strip(c[-1]) == strip(g[-1])
                ok_all &= same
                print(json.dumps({'row': i, 'form': gx, 'maps': maps.count(',') + 1, 'cmin': cmin, 'imin': imin, 'identical': same,
                                  'passes': len(cp), 'summary': strip(c[-1])[:160]}), flush=True)
print(json.dumps({'all_identical': ok_all}))
