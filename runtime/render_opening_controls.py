"""Known-pose real-renderer controls at the independently fitted opening view."""
import json,subprocess,shutil,math
from pathlib import Path

BASE=Path(__file__).resolve().parent
fit=max(json.loads((BASE.parent/'generator/opening-texture/fit-2.10.json').read_text()),key=lambda r:r['train'])
yaw,pitch,height,cx,cz=fit['parameters']
player=[128+(cx+.3*math.sin(yaw)*math.cos(pitch))%1,
        43+height-.3*math.sin(pitch),
        128+(cz-.3*math.cos(yaw)*math.cos(pitch))%1]
command=json.loads((BASE/'render-texture-control/command.json').read_text())
for width,height,suffix in [(640,360,'native'),(1280,720,'hd')]:
    folder=BASE/f'render-opening-control-{suffix}'
    folder.mkdir(exist_ok=True)
    shutil.copyfile(BASE/'render-texture-control/level.dat',folder/'level.dat')
    args=[f'-Dprobe.width={width}' if a.startswith('-Dprobe.width=') else
          f'-Dprobe.height={height}' if a.startswith('-Dprobe.height=') else a for a in command]
    args[-5:]=list(map(str,[*player,math.degrees(pitch),math.degrees(yaw)]))
    (folder/'command.json').write_text(json.dumps(args,indent=2)+'\n')
    (folder/'truth.json').write_text(json.dumps({'player':player,'pitch_deg':math.degrees(pitch),'yaw_deg':math.degrees(yaw),'grass_plane_y':43,'fov':70,'synthetic':True},indent=2)+'\n')
    with (folder/'stdout.txt').open('w') as out,(folder/'stderr.txt').open('w') as err:
        result=subprocess.run(args,cwd=folder,stdout=out,stderr=err,timeout=180)
    print(suffix,result.returncode,flush=True);result.check_returncode()
