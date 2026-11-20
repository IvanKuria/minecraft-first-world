"""Startup pop-in model (rd-132211 LevelRenderer/Chunk/Frustum, unchanged-since-17:32 classes per DEEP-REVIEW;
LevelRenderer/Chunk themselves are later builds -> model, not proof).
Camera model = generator/opening-texture/model.py (pixel centres, f=180/tan35, 0.3 setback, yaw=yRot, pitch=xRot).
Per render pass the FIRST dirty frustum chunk in index order (x + y*16)*16 + z is rebuilt (both lists); a chunk rebuilt
in pass 1 shows only its dark list until the next frame."""
import numpy as np, math
F = 180 / math.tan(math.radians(35))
W = H = 256; D = 64

def ray_dirs(yaw, pitch, us, vs):
    rx = (us + .5 - 320) / F; ry = (vs + .5 - 180) / F
    s, c, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    dx = c * rx - s * sp * ry + s * cp
    dy = -cp * ry - sp
    dz = s * rx + c * sp * ry - c * cp
    return dx, dy, dz

def cam_pos(px, py, pz, yaw, pitch):
    """camera centre from player position (0.3 setback along view direction)"""
    s, c, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    return px - .3 * s * cp, py + .3 * sp, pz + .3 * c * cp

def plane_hits(yaw, pitch, cy, us, vs, plane=64.0):
    """relative hit offsets (from camera centre) of pixel rays with the horizontal plane y=plane; nan if no hit"""
    dx, dy, dz = ray_dirs(yaw, pitch, us, vs)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (plane - cy) / dy
    ok = (dy < 0) & (t > 0)
    rx = np.where(ok, t * dx, np.nan); rz = np.where(ok, t * dz, np.nan)
    return rx, rz, np.where(ok, t, np.nan)

def chunk_index_map(px, pz, cyeye, yaw, pitch, us, vs, plane=64.0):
    """chunk array index of the layer-3 chunk under each pixel (top face of the plain at y=plane); -1 sky/out of world"""
    cx_, cy_, cz_ = cam_pos(px, cyeye, pz, yaw, pitch)
    rx, rz, t = plane_hits(yaw, pitch, cy_, us, vs, plane)
    hx = cx_ + rx; hz = cz_ + rz
    ok = np.isfinite(hx) & (hx >= 0) & (hx < W) & (hz >= 0) & (hz < H)
    ci = np.where(ok, (np.floor(np.where(ok, hx, 0) / 16).astype(int) + 3 * 16) * 16 + np.floor(np.where(ok, hz, 0) / 16).astype(int), -1)
    return ci, hx, hz

# ---- exact GL frustum (gluPerspective(70, aspect, .05, 1000); modelview T(0,0,-.3) Rx(xRot) Ry(yRot) T(-p)) ----
def gl_matrices(px, py, pz, yaw_deg, pitch_deg, aspect=16 / 9):
    f = 1 / math.tan(math.radians(35)); n, fa = .05, 1000.
    P = np.zeros((4, 4)); P[0, 0] = f / aspect; P[1, 1] = f; P[2, 2] = (fa + n) / (n - fa); P[2, 3] = 2 * fa * n / (n - fa); P[3, 2] = -1
    def Rx(a):
        a = math.radians(a); c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.]])
    def Ry(a):
        a = math.radians(a); c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.]])
    def T(x, y, z):
        m = np.eye(4); m[:3, 3] = [x, y, z]; return m
    M = T(0, 0, -.3) @ Rx(pitch_deg) @ Ry(yaw_deg) @ T(-px, -py, -pz)
    return P, M

def frustum_planes(P, M):
    # GL stores column-major: proj[i] with i = col*4 + row. The Java code computes clip = modl x proj in that layout,
    # i.e. clip(col-major) = P @ M. Planes: right = clip[3]-clip[0] etc. using the column-major flat array.
    C = P @ M
    clip = C.T.reshape(-1)          # column-major flatten
    pl = np.zeros((6, 4))
    pl[0] = [clip[3] - clip[0], clip[7] - clip[4], clip[11] - clip[8], clip[15] - clip[12]]
    pl[1] = [clip[3] + clip[0], clip[7] + clip[4], clip[11] + clip[8], clip[15] + clip[12]]
    pl[2] = [clip[3] + clip[1], clip[7] + clip[5], clip[11] + clip[9], clip[15] + clip[13]]
    pl[3] = [clip[3] - clip[1], clip[7] - clip[5], clip[11] - clip[9], clip[15] - clip[13]]
    pl[4] = [clip[3] - clip[2], clip[7] - clip[6], clip[11] - clip[10], clip[15] - clip[14]]
    pl[5] = [clip[3] + clip[2], clip[7] + clip[6], clip[11] + clip[10], clip[15] + clip[14]]
    pl /= np.linalg.norm(pl[:, :3], axis=1)[:, None]
    return pl

_CORN = None
def chunk_corners():
    global _CORN
    if _CORN is None:
        idx = np.arange(1024); x = idx // 64; y = (idx // 16) % 4; z = idx % 16
        # index = (x + y*16)*16 + z  ->  x = idx // ... careful: (x + y*16)*16 + z, x<16,y<4,z<16
        x = (idx // 16) % 16; y = idx // 256; z = idx % 16
        lo = np.stack([x * 16, y * 16, z * 16], 1).astype(float); hi = lo + 16
        corners = np.stack([np.stack([np.where(b & 1, hi[:, 0], lo[:, 0]), np.where(b & 2, hi[:, 1], lo[:, 1]),
                                      np.where(b & 4, hi[:, 2], lo[:, 2])], 1) for b in range(8)], 1)  # (1024,8,3)
        _CORN = corners
    return _CORN

def in_frustum(px, py, pz, yaw_deg, pitch_deg, aspect=16 / 9):
    P, M = gl_matrices(px, py, pz, yaw_deg, pitch_deg, aspect)
    pl = frustum_planes(P, M)
    c = chunk_corners()
    d = np.einsum('kcj,pj->kpc', c, pl[:, :3]) + pl[None, :, 3, None]   # (1024,6,8)
    out = (d <= 0).all(axis=2).any(axis=1)
    return ~out

def index_check():
    # (x + y*16)*16 + z for x=3,y=2,z=5
    i = (3 + 2 * 16) * 16 + 5
    assert ((i // 16) % 16, i // 256, i % 16) == (3, 2, 5)
index_check()
