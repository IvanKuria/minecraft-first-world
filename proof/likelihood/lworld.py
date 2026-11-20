"""LIKELIHOOD lane helpers: load lgen dumps, rebuild the fixed-T world from one long run, column tops."""
import json, os, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
W = H = 256; D = 64; V = W * H * D
LGEN = HERE / 'lgen'

def run(tag, spec, state, M, outdir=None):
    """one long lgen run with arrival/raw/seed dumps -> worlds/<tag>.{arr,raw,seeds,json}"""
    od = Path(outdir or HERE / 'worlds'); od.mkdir(exist_ok=True)
    env = {**os.environ, 'LGEN_ARRIVAL': str(od / f'{tag}.arr'), 'LGEN_RAW': str(od / f'{tag}.raw'), 'GEN_SEEDLOG': str(od / f'{tag}.seeds')}
    out = subprocess.run([str(LGEN), 'run', f'{spec},M={M}', str(state), '-'], capture_output=True, text=True, env=env, check=True).stdout
    meta = json.loads(out); meta.update({'tag': tag, 'spec': spec, 'state_in_arg': str(state), 'M_run': M})
    (od / f'{tag}.json').write_text(json.dumps(meta) + '\n')
    return meta

def load(tag, d=None):
    od = Path(d or HERE / 'worlds')
    raw = np.fromfile(od / f'{tag}.raw', np.uint8).reshape(D, H, W)
    arr = np.fromfile(od / f'{tag}.arr', np.uint32).reshape(D, H, W)
    seeds = np.loadtxt(od / f'{tag}.seeds', dtype=np.int64, ndmin=2)  # k x y z t
    meta = json.loads((od / f'{tag}.json').read_text())
    return raw, arr, seeds, meta

def world_at(raw, arr, T_iter):
    """fixed-T world (T in ITERATIONS): cells set before iteration T keep their type; everything else (leftover) -> air."""
    w = np.where(arr < np.uint32(T_iter), raw, 0).astype(np.uint8)
    w[w == 255] = 0
    return w

def tops(w):
    s = w != 0
    return np.where(s.any(0), D - 1 - np.argmax(s[::-1], axis=0), -1)  # [z, x]
