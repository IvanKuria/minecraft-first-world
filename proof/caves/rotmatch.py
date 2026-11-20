"""Test pure rotation between a pop-in frame and a later frame with known pose (frame 41, opening-texture fit)."""
import sys, math
import numpy as np, cv2
from scipy.optimize import minimize
from popin_obs import frame
F = 180 / math.tan(math.radians(35))
def dirs(yaw, pitch, u, v):
    rx = (u + .5 - 320) / F; ry = (v + .5 - 180) / F
    s, c, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    return np.stack([c * rx - s * sp * ry + s * cp, -cp * ry - sp, s * rx + c * sp * ry - c * cp], -1)
def project(yaw, pitch, d):
    # inverse of dirs: camera basis
    s, c, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    right = np.array([c, 0, s]); up = np.array([s * sp, cp, -c * sp]) * -1; fwd = np.array([s * cp, -sp, -c * cp])
    # dirs = right*rx + (-up_img)*ry + fwd  where ry increases downward: d = rx*right + ry*down + fwd, down=[-s*sp,-cp,c*sp]
    down = np.array([-s * sp, -cp, c * sp])
    z = d @ fwd; x = d @ right / z; y = d @ down / z
    return x * F + 320 - .5, y * F + 180 - .5, z
def sample(im, u, v):
    return cv2.remap(im, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderValue=(255, 0, 255))
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2])
    yb, pb = math.radians(float(sys.argv[3])), math.radians(float(sys.argv[4]))
    A = frame(a).astype(np.float32); B = frame(b).astype(np.float32)
    v0, v1 = int(sys.argv[5]), int(sys.argv[6])
    us, vs = np.meshgrid(np.arange(0, 640, 1.0), np.arange(v0, v1, 1.0))
    obs = A[v0:v1, :, 1]
    mask = obs < 200   # terrain only (sky g ~200)
    def loss(p):
        d = dirs(math.radians(p[0]), math.radians(p[1]), us, vs)
        u2, v2, z = project(yb, pb, d)
        s = sample(B, u2, v2)[..., 1]
        m = mask & (z > 0) & (u2 > 2) & (u2 < 637) & (v2 > 2) & (v2 < 357)
        if m.sum() < 500: return 1.0
        x = obs[m] - obs[m].mean(); y = s[m] - s[m].mean()
        return -float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    best = []
    for y0 in np.arange(float(sys.argv[7]), float(sys.argv[8]), 1.0):
        for p0 in np.arange(float(sys.argv[9]), float(sys.argv[10]), .5):
            best.append((loss([y0, p0]), y0, p0))
    best.sort()
    print('grid best', best[:5])
    r = minimize(loss, [best[0][1], best[0][2]], method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-5})
    print('refined', r.x, -r.fun)
