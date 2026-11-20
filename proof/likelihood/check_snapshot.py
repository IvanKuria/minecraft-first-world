"""World-at-T rebuilt from ONE long dump run must equal a direct lgen run with M=T (byte-identical blocks)."""
import subprocess, sys, json, numpy as np
import lworld as L
spec, state, T, Mlong = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
L.run('chk', spec, state, Mlong)
raw, arr, seeds, meta = L.load('chk')
w = L.world_at(raw, arr, T)
subprocess.run([str(L.LGEN), 'run', f'{spec},M={T}', state, str(L.HERE / 'worlds/chk_direct.blocks')], check=True, capture_output=True)
d = np.fromfile(L.HERE / 'worlds/chk_direct.blocks', np.uint8).reshape(L.D, L.H, L.W)
print(json.dumps({'spec': spec, 'state': state, 'T': T, 'Mlong': Mlong, 'identical': bool((w == d).all()), 'ndiff': int((w != d).sum()),
                  'seeds': int(len(seeds))}))
