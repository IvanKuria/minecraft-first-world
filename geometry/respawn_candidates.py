from pathlib import Path
import cv2,numpy as np,json
P=Path(__file__).parent
cap=cv2.VideoCapture(str(P.parent/'footage-sources/nizzotch-original-channel-copy.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);rows=[];i=0
while True:
 ok,im=cap.read()
 if not ok:break
 if i%3==0:
  small=cv2.resize(im,(213,120));b,g,r=cv2.split(small.astype(np.int16));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);rows.append([i/fps,float(np.mean(sky))])
 i+=1
runs=[];active=[]
for t,p in rows:
 if p>.85:active.append(t)
 elif active:runs.append([active[0],active[-1]]);active=[]
if active:runs.append([active[0],active[-1]])
print(runs);(P/'sky-fraction.json').write_text(json.dumps({'fps':fps,'fraction':rows,'sky_runs_above_85_percent':runs},indent=2))
