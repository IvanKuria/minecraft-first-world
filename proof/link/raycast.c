/* Voxel ray caster for the LINK scorer.
 * World layout follows rd-132211 Level: index (y*H + z)*W + x, W=width(x), H=height(z), D=depth(y).
 * A cell is solid when blocks[i] != 0. Rays start at origin o and travel along d (need not be unit).
 * Output per ray: hit (0/1), t (distance along unit direction), face (0..5 = -y,+y,-z,+z,-x,+x;
 * i.e. the face of the hit cell that the ray entered, matching Tile.render face order), and cell xyz.
 * Pure geometry; lighting is applied by the caller using Notch's 2D depth-map rule.
 * Build: cc -O2 -shared -fPIC -o libraycast.dylib raycast.c
 */
#include <math.h>
#include <stdint.h>

static int solid(const uint8_t *b, int W, int H, int D, int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H) return 0;
    return b[((long)y * H + z) * W + x] != 0;
}

void cast_rays(const uint8_t *blocks, int W, int H, int D,
               double ox, double oy, double oz,
               const double *dx, const double *dy, const double *dz, int n,
               double maxdist,
               uint8_t *hit, double *tout, uint8_t *face, int32_t *cell) {
    for (int r = 0; r < n; r++) {
        double vx = dx[r], vy = dy[r], vz = dz[r];
        double len = sqrt(vx * vx + vy * vy + vz * vz);
        vx /= len; vy /= len; vz /= len;
        hit[r] = 0; tout[r] = -1; face[r] = 255; cell[3*r] = cell[3*r+1] = cell[3*r+2] = -1;
        /* clip to world box */
        double t0 = 0, t1 = maxdist;
        double lo[3] = {0, 0, 0}, hi[3] = {W, D, H}, o[3] = {ox, oy, oz}, v[3] = {vx, vy, vz};
        int ok = 1;
        for (int a = 0; a < 3 && ok; a++) {
            if (fabs(v[a]) < 1e-12) { if (o[a] < lo[a] || o[a] > hi[a]) ok = 0; continue; }
            double ta = (lo[a] - o[a]) / v[a], tb = (hi[a] - o[a]) / v[a];
            if (ta > tb) { double s = ta; ta = tb; tb = s; }
            if (ta > t0) t0 = ta;
            if (tb < t1) t1 = tb;
            if (t0 > t1) ok = 0;
        }
        if (!ok) continue;
        double eps = 1e-9;
        double px = ox + vx * (t0 + eps), py = oy + vy * (t0 + eps), pz = oz + vz * (t0 + eps);
        int x = (int)floor(px), y = (int)floor(py), z = (int)floor(pz);
        if (x >= W) x = W - 1; if (y >= D) y = D - 1; if (z >= H) z = H - 1;
        if (x < 0) x = 0; if (y < 0) y = 0; if (z < 0) z = 0;
        int sx = vx > 0 ? 1 : -1, sy = vy > 0 ? 1 : -1, sz = vz > 0 ? 1 : -1;
        double tdx = fabs(vx) < 1e-12 ? 1e30 : fabs(1.0 / vx);
        double tdy = fabs(vy) < 1e-12 ? 1e30 : fabs(1.0 / vy);
        double tdz = fabs(vz) < 1e-12 ? 1e30 : fabs(1.0 / vz);
        double tmx = fabs(vx) < 1e-12 ? 1e30 : ((sx > 0 ? (x + 1 - ox) : (ox - x)) * tdx);
        double tmy = fabs(vy) < 1e-12 ? 1e30 : ((sy > 0 ? (y + 1 - oy) : (oy - y)) * tdy);
        double tmz = fabs(vz) < 1e-12 ? 1e30 : ((sz > 0 ? (z + 1 - oz) : (oz - z)) * tdz);
        /* entry face when starting at box boundary */
        int f = 255; double tcur = t0;
        if (t0 > 0) {
            /* which axis bounded t0 */
            double ex = fabs(vx) < 1e-12 ? -1 : ((sx > 0 ? 0 : W) - ox) / vx;
            double ey = fabs(vy) < 1e-12 ? -1 : ((sy > 0 ? 0 : D) - oy) / vy;
            double ez = fabs(vz) < 1e-12 ? -1 : ((sz > 0 ? 0 : H) - oz) / vz;
            if (fabs(ey - t0) < 1e-7) f = sy > 0 ? 0 : 1;
            else if (fabs(ez - t0) < 1e-7) f = sz > 0 ? 2 : 3;
            else if (fabs(ex - t0) < 1e-7) f = sx > 0 ? 4 : 5;
        }
        for (int it = 0; it < 4096; it++) {
            if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H || tcur > t1 + 1e-9) break;
            if (solid(blocks, W, H, D, x, y, z)) {
                if (f == 255) break; /* origin inside a solid cell: treat as no hit */
                hit[r] = 1; tout[r] = tcur; face[r] = (uint8_t)f;
                cell[3*r] = x; cell[3*r+1] = y; cell[3*r+2] = z;
                break;
            }
            if (tmx < tmy && tmx < tmz) { tcur = tmx; x += sx; tmx += tdx; f = sx > 0 ? 4 : 5; }
            else if (tmy < tmz) { tcur = tmy; y += sy; tmy += tdy; f = sy > 0 ? 0 : 1; }
            else { tcur = tmz; z += sz; tmz += tdz; f = sz > 0 ? 2 : 3; }
        }
    }
}
