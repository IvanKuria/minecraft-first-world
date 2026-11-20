from pathlib import Path
import cv2,numpy as np,json
p=Path(__file__).parent
cap=cv2.VideoCapture(str(p.parent/'footage-sources/nizzotch-original-channel-copy.mp4'));print(cap.get(cv2.CAP_PROP_FPS))
results={}
for n in [11,12,13]:
 target=cv2.imread(str(p/'frames'/f'f{n:03}.png')); t0=(n-.5)/2
 best=(1e10,None,None)
 for t in np.arange(t0-.25,t0+.26,.025):
  cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,im=cap.read()
  if not ok:continue
  small=cv2.resize(im,(640,360));err=np.mean((small.astype(float)-target.astype(float))**2)
  if err<best[0]:best=(err,t,im.copy())
 cv2.imwrite(str(p/'frames'/f'hd{n:03}.png'),best[2]);results[n]={'time':best[1],'mse':best[0]};print(n,results[n])
(p/'hd-match.json').write_text(json.dumps(results,indent=2))
