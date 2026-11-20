import sys, math
import numpy as np, cv2
import popin
from popin_fit import us, vs, R0, R1, obs_masks, FR
def colmap(ci):
    out = np.zeros(ci.shape + (3,), np.uint8); out[:] = (240, 200, 117)
    ok = ci >= 0
    cx = (ci // 16) % 16; cz = ci % 16
    rng = np.random.RandomState(1)
    pal = rng.randint(60, 255, (16, 16, 3))
    out[ok] = pal[cx[ok], cz[ok]]
    return out
def show(X, Z, yaw, pitch, eye, fname, frames=FR):
    m = obs_masks(frames)
    ci, hx, hz = popin.chunk_index_map(X, Z, eye, math.radians(yaw), math.radians(pitch), us, vs)
    rows = [cv2.resize(colmap(ci), (1280, (R1 - R0) * 2), interpolation=cv2.INTER_NEAREST)]
    for n in frames:
        im = np.where(m[n][..., None], 80, 255).astype(np.uint8).repeat(3, 2)
        # overlay chunk boundaries
        bx = (np.diff(np.floor(hx / 16), axis=1, prepend=np.nan) != 0) | (np.diff(np.floor(hz / 16), axis=1, prepend=np.nan) != 0)
        by = (np.diff(np.floor(hx / 16), axis=0, prepend=np.nan) != 0) | (np.diff(np.floor(hz / 16), axis=0, prepend=np.nan) != 0)
        b = (bx | by) & np.isfinite(hx)
        im[b] = (0, 0, 255)
        rows.append(cv2.resize(im, (1280, (R1 - R0) * 2), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(fname, np.vstack(rows))
if __name__ == '__main__':
    X, Z, yaw, pitch, eye = map(float, sys.argv[1:6])
    show(X, Z, yaw, pitch, eye, sys.argv[6])
