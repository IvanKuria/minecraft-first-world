"""Synthetic observation of a generated world from the footage poses, scored with the SAME texture plane sweep.
usage: synth.py <world.blocks> <nspawn> <seed> <out.json> [grassrule]
For each random spawn (integer column in [24,232]^2) the 4 first-reset and 4 second-reset poses are rendered with the
authentic textures (trender), blurred (sigma .7 px), swept exactly like the footage (sweep.frame_scores/combine), and
labelled with the frozen thresholds.  The reference plane is the world's modal column-top level (like the footage, where
the pose fits lock onto the dominant flat level).  Ground truth top heights are recorded for every labelled column.
"""
import json, math, sys, subprocess, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import sweep as S
import surf
HERE = S.HERE
W = H = 256; D = 64
THR = [(.45, .15), (.35, .12), (.3, .1)]
BLUR = .9; NOISE = 7.0; NEAR = 10.0
NOISE_RNG = np.random.default_rng(12345)

def render(world, px, py, pz, yaw, pitch, rule, gy, out):
    subprocess.run([str(HERE / 'trender'), str(world), '256', '256', '64', '%.6f' % px, '%.6f' % py, '%.6f' % pz,
                    '%.8f' % yaw, '%.8f' % pitch, '640', '360', rule, str(gy), str(HERE / 'tex/terrain.ppm'), str(out)], check=True)
    a = np.asarray(Image.open(out).convert('RGB')).astype(float)
    from scipy.ndimage import gaussian_filter
    a = np.stack([gaussian_filter(a[..., c], BLUR) for c in range(3)], -1)
    a = a + NOISE_RNG.normal(0, NOISE, a.shape)
    return Image.fromarray(np.clip(a + .5, 0, 255).astype(np.uint8))

def main():
    world, nsp, seed, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rule = sys.argv[5] if len(sys.argv) > 5 else 'top'
    b = surf.load(world)
    t = surf.tops(b)  # [z, x]
    st, _ = surf.stats(b)
    mode = st['mode']
    rng = np.random.default_rng(seed)
    tmp = HERE / f'tmp_{os.getpid()}.ppm'
    res = {'world': world, 'surface': st, 'mode': mode, 'rule': rule, 'spawns': []}
    sets = {'r1': S.r1_poses(), 'r2': S.r2_poses()}
    for i in range(nsp):
        cx, cz = (int(v) for v in rng.integers(24, 232, 2))
        rec = {'cx': cx, 'cz': cz, 'top_at_spawn': int(t[cz, cx]) - mode}
        for which, (phase, frames) in sets.items():
            per = []
            for name, path, yaw, pitch, h in frames:
                px = cx + phase[0]; pz = cz + phase[1]
                py = mode + 1 + h - .3 * math.sin(pitch)
                im = render(world, px, py, pz, yaw, pitch, rule, mode, tmp)
                per.append(S.frame_scores(im, yaw, pitch, h, phase))
            res_c = S.combine(per)
            # near-field truth (all columns within NEAR of the player): true top relative to mode
            tru_near = {}
            for x in range(-11, 12):
                for z in range(-11, 12):
                    if math.hypot(x + .5 - phase[0], z + .5 - phase[1]) < NEAR:
                        k = int(t[cz + z, cx + x]) - mode; tru_near[k] = tru_near.get(k, 0) + 1
            rec[f'{which}_truth_near'] = tru_near
            for rm, mg in THR:
                lab = S.assign(res_c, rm, mg)
                lab = {c: v for c, v in lab.items() if math.hypot(c[0] + .5 - phase[0], c[1] + .5 - phase[1]) < NEAR}
                hist = {}
                conf = {}
                for (x, z), v in lab.items():
                    k = v[0]; hist[k] = hist.get(k, 0) + 1
                    X, Z = cx + x, cz + z
                    tru = int(t[Z, X]) - mode if 0 <= X < W and 0 <= Z < H else None
                    key = f'{k}|{tru}'; conf[key] = conf.get(key, 0) + 1
                rec[f'{which}_{rm}'] = {'n': len(lab), 'hist': hist, 'conf': conf}
        res['spawns'].append(rec)
        print(json.dumps({'i': i, 'cx': cx, 'cz': cz, 'r1': rec['r1_0.35']['hist'], 'r2': rec['r2_0.35']['hist']}), flush=True)
    tmp.unlink(missing_ok=True)
    Path(outp).write_text(json.dumps(res) + '\n')

if __name__ == '__main__':
    main()
