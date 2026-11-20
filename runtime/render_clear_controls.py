"""Actual preserved-renderer calibration at the clear-grass film pose.

Synthetic flat plane, not a recreation of the historical terrain. Pose is
known by construction; matched film pose only motivates the test geometry.
"""
import json,subprocess,shutil
from pathlib import Path

BASE=Path(__file__).resolve().parent
command=json.loads((BASE/'render-texture-control/command.json').read_text())
for width,height,suffix in [(640,360,'native'),(1280,720,'hd')]:
    folder=BASE/f'render-clear-control-{suffix}'
    folder.mkdir(exist_ok=True)
    shutil.copyfile(BASE/'render-texture-control/level.dat',folder/'level.dat')
    args=[f'-Dprobe.width={width}' if a.startswith('-Dprobe.width=') else
          f'-Dprobe.height={height}' if a.startswith('-Dprobe.height=') else a for a in command]
    args[-5:]=['223.796530','49.935439','123.795970','41.973230','129.096507']
    (folder/'command.json').write_text(json.dumps(args,indent=2)+'\n')
    with (folder/'stdout.txt').open('w') as out,(folder/'stderr.txt').open('w') as err:
        result=subprocess.run(args,cwd=folder,stdout=out,stderr=err,timeout=180)
    print(suffix,result.returncode,flush=True)
    result.check_returncode()
