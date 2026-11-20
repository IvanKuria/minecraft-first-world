"""Observed sky/terrain masks for the startup pop-in frames of the 2011 640x360 original-channel copy.
Frames are decoded with ffmpeg select=eq(n,N) (same indexing as LINK/R1MAP; n2959 reproduces LINK's r2 frame)."""
import subprocess, sys, json
from pathlib import Path
import numpy as np, cv2
P = Path(__file__).parent
V = P.parent / 'link/footage/wayback-fakeurl.mp4'
FR = P / 'frames'

def frame(n):
    f = FR / f'n{n:04d}.png'
    if not f.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(V), '-vf', f'select=eq(n\\,{n})', '-vsync', '0',
                        '-frames:v', '1', str(f)], check=True)
    return cv2.imread(str(f))

def skymask(im):
    b, g, r = [im[..., i].astype(int) for i in range(3)]
    return (b > 200) & (b - r > 60)

if __name__ == '__main__':
    out = {}
    for n in range(24, 46):
        im = frame(n)
        sky = skymask(im)
        terr = ~sky
        cols = []
        for u in range(640):
            rows = np.nonzero(terr[:, u])[0]
            cols.append(int(rows.min()) if rows.size else -1)
        cols = np.array(cols)
        tops = cols[cols >= 0]
        # rightmost terrain column in the bottom 14 rows
        bot = terr[346:360]
        anyc = np.nonzero(bot.any(axis=0))[0]
        out[n] = {'terrain_px': int(terr.sum()), 'top_row_median': float(np.median(tops)) if tops.size else None,
                  'top_row_p10': float(np.percentile(tops, 10)) if tops.size else None,
                  'cols_with_terrain': int((cols >= 0).sum()),
                  'max_col_bottom': int(anyc.max()) if anyc.size else None}
        print(n, out[n])
    (P / 'popin-obs-summary.json').write_text(json.dumps(out, indent=1))
