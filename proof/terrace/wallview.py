"""Edge test: render a world from just outside its -z wall (or -x wall) at plain height, pitch ~0, like 163.25 s, to
see the wall's top profile.  Footage 163.25-163.35 s: ONE grass band (top block side face) then stone, straight top edge."""
import math, sys, subprocess
import numpy as np
from PIL import Image
import surf
HERE = surf.__file__.rsplit('/', 1)[0]
def render(world, px, py, pz, yaw, pitch, out, W=852, H=480):
    subprocess.run([HERE + '/trender', world, '256', '256', '64', '%.4f' % px, '%.4f' % py, '%.4f' % pz, '%.6f' % yaw, '%.6f' % pitch,
                    str(W), str(H), 'top', '42', HERE + '/tex/terrain.ppm', out], check=True)
    return Image.open(out).convert('RGB')
ims = []
for w in sys.argv[2:]:
    b = surf.load(w); t = surf.tops(b)
    mode = surf.stats(b)[0]['mode']
    # camera 1.2 blocks outside the z=0 wall (z=-1.2), at x=128.5, eye 0.4 above the wall top (plain mode+1), looking +z (yaw=pi)
    for (xx, zz, yaw) in ((128.5, -1.2, math.pi), (64.5, -1.2, math.pi)):
        ims.append(render(w, xx, mode + 1 - .3, zz, yaw, 0.12, HERE + '/tmp_wall.ppm'))
sheet = Image.new('RGB', (852 * 2, 480 * (len(ims) // 2)))
for i, im in enumerate(ims): sheet.paste(im, ((i % 2) * 852, (i // 2) * 480))
sheet.resize((852, 240 * (len(ims) // 2))).save(HERE + f'/img/{sys.argv[1]}.png')
