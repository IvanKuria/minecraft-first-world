"""Forward-simulate the rd-132211 build order at a static pose and render which strip pixels are built after n builds."""
import sys, math
import numpy as np, cv2
import popin
from popin_fit import us, vs, R0, R1
def sim(X, Z, eye, yaw, pitch, ns):
    fr = popin.in_frustum(X, eye, Z, yaw, pitch)
    order = np.nonzero(fr)[0]          # ascending index = build order (static camera)
    ci, hx, hz = popin.chunk_index_map(X, Z, eye, math.radians(yaw), math.radians(pitch), us, vs)
    rank = np.full(1024, 10**6); rank[order] = np.arange(len(order))
    rk = np.where(ci >= 0, rank[np.maximum(ci, 0)], 10**6)
    return order, [(rk < n) for n in ns]
if __name__ == '__main__':
    X, Z, eye, yaw, pitch = map(float, sys.argv[1:6])
    ns = list(range(0, 120, 10))
    order, ms = sim(X, Z, eye, yaw, pitch, ns)
    print('frustum chunks', len(order), 'layers', np.bincount(order // 256, minlength=4))
    rows = []
    for n, m in zip(ns, ms):
        im = np.where(m, 60, 230).astype(np.uint8)
        im = cv2.resize(im, (1280, (R1 - R0) * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(im, str(n), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1)
        rows.append(im); rows.append(np.full((2, 1280), 255, np.uint8))
    cv2.imwrite(sys.argv[6], np.vstack(rows))
