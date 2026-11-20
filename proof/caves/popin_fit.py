"""Fit the threshold pop-in model to the observed sky/terrain masks of frames 29-34."""
import sys, json, math
from pathlib import Path
import numpy as np, cv2
import popin
from popin_obs import frame, skymask
P = Path(__file__).parent
R0, R1 = 336, 360
us, vs = np.meshgrid(np.arange(640, dtype=float), np.arange(R0, R1, dtype=float))
FR = [29, 30, 31, 32, 33, 34]

def obs_masks(frames=FR):
    return {n: ~skymask(frame(n))[R0:R1] for n in frames}

def best_threshold(ci, ob):
    """best I such that terrain pred = (ci>=0)&(ci<I). returns (iou, I, agree)"""
    valid = ci >= 0
    # sky pixels predicted sky always
    v = ci[valid]; o = ob[valid]
    order = np.argsort(v, kind='stable'); v = v[order]; o = o[order]
    # candidate thresholds at unique values: pred terrain = first j pixels
    uniq, first = np.unique(v, return_index=True)
    cum_o = np.concatenate([[0], np.cumsum(o)])
    tot_o_all = ob.sum()
    best = (-1, None, None)
    for j, I in list(zip(first, uniq)) + [(len(v), uniq[-1] + 1 if len(uniq) else 0)]:
        inter = cum_o[j]; pred = j
        union = pred + tot_o_all - inter
        iou = inter / union if union else 1.0
        if iou > best[0]: best = (iou, int(I), int(j))
    return best

def evaluate(X, Z, yaw_deg, pitches, eye=64.62, masks=None, plane=64.0):
    masks = masks or obs_masks()
    res = []
    for n, p in zip(FR, pitches):
        ci, hx, hz = popin.chunk_index_map(X, Z, eye, math.radians(yaw_deg), math.radians(p), us, vs, plane)
        res.append(best_threshold(ci, masks[n]))
    return res

if __name__ == '__main__':
    m = obs_masks()
    tops = {n: np.argmax(m[n].any(axis=1)) + R0 for n in FR}
    print('top rows', tops)
