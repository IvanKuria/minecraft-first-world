"""Structural metrics of a 256x256x64 block file (layout (y*H+z)*W+x).  usage: metrics.py file.blocks [...]"""
import sys, json
import numpy as np
from scipy import ndimage

W, H, D = 256, 256, 64

def load(p):
    a = np.fromfile(p, dtype=np.uint8).reshape(D, H, W)  # [y, z, x]
    return a != 0

def metrics(s):
    out = {}
    # column top (light depth): highest solid y, -1 if none
    has = s.any(axis=0)
    top = np.where(has, D - 1 - np.argmax(s[::-1], axis=0), -1)
    vals, cnt = np.unique(top, return_counts=True)
    mode = int(vals[np.argmax(cnt)])
    out["top_mode_y"] = mode
    out["frac_cols_top_eq_mode"] = float((top == mode).mean())
    out["frac_cols_top_above_mode"] = float((top > mode).mean())
    out["frac_cols_top_mode_minus1_or_2"] = float(((top < mode) & (top >= mode - 2)).mean())
    out["frac_cols_top_deeper"] = float((top < mode - 2).mean())
    out["top_std"] = float(top.std())
    out["max_top"] = int(top.max())
    # walls: x=0,x=W-1,z=0,z=H-1 for y<=mode ; bottom y=0
    walls = np.concatenate([s[:mode + 1, :, 0].ravel(), s[:mode + 1, :, W - 1].ravel(), s[:mode + 1, 0, :].ravel(), s[:mode + 1, H - 1, :].ravel()])
    out["wall_solid_frac_below_mode"] = float(walls.mean())
    out["bottom_solid_frac"] = float(s[0].mean())
    faces = [s[:mode + 1, :, 0], s[:mode + 1, :, W - 1], s[:mode + 1, 0, :], s[:mode + 1, H - 1, :]]
    out["wall_min_face_solid"] = float(min(f.mean() for f in faces))
    btop = np.concatenate([top[:, 0], top[:, W - 1], top[0, :], top[H - 1, :]])
    out["boundary_cols_top_eq_mode"] = float((btop == mode).mean())
    inner = s[1:max(mode - 1, 2), 1:-1, 1:-1]
    out["interior_air_frac"] = float(1 - inner.mean())
    lab, n = ndimage.label(~inner)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        out["interior_air_components"] = int(n)
        out["largest_air_comp_share"] = float(sizes.max() / sizes.sum())
        out["air_comps_ge_1000"] = int((sizes >= 1000).sum())
        out["air_single_cells"] = int((sizes == 1).sum())
    # isolated solids (no 6-neighbour solid), anywhere
    p = np.pad(s, 1)
    nb = p[:-2, 1:-1, 1:-1].astype(np.int8) + p[2:, 1:-1, 1:-1] + p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] + p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:]
    out["isolated_solids"] = int((s & (nb == 0)).sum())
    # interface roughness: solid/air face transitions per cell in interior
    out["face_interfaces_per_cell"] = float(((s[1:] != s[:-1]).sum() + (s[:, 1:] != s[:, :-1]).sum() + (s[:, :, 1:] != s[:, :, :-1]).sum()) / s.size)
    # plain-level hole fraction (cells at mode level that are air while column has something)
    out["holes_at_mode_level"] = float(1 - s[mode].mean())
    out["solid_frac"] = float(s.mean())
    t1 = out["wall_min_face_solid"] >= 0.98 and out["bottom_solid_frac"] >= 0.98
    t2 = out["boundary_cols_top_eq_mode"] >= 0.90
    t3 = out["frac_cols_top_eq_mode"] >= 0.50
    t4 = out["interior_air_frac"] >= 0.10 and out.get("largest_air_comp_share", 0) * out["interior_air_frac"] * inner.size >= 1000
    out["T"] = {"T1": bool(t1), "T2": bool(t2), "T3": bool(t3), "T4": bool(t4)}
    out["pass_all"] = bool(t1 and t2 and t3 and t4)
    return out

if __name__ == "__main__":
    res = {p: metrics(load(p)) for p in sys.argv[1:]}
    print(json.dumps(res, indent=1))
