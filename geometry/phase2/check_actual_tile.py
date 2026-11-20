import cv2,numpy as np,json
from pathlib import Path
P=Path(__file__).parent;data=json.loads((P.parent/'patch-fit.json').read_text());out=[];imgs=[];tile=cv2.imread(str(P/'terrain.png'))[:16,:16]
for n in [11,12,13]:
 im=cv2.imread(str(P.parent/'frames'/f'hd{n:03}.png'));pts=np.float32(data['observations'][str(n)]['cube'][:4]);target=np.float32([[0,0],[128,0],[128,128],[0,128]]);H=cv2.getPerspectiveTransform(pts,target);rect=cv2.warpPerspective(im,H,(128,128));imgs.append(rect);small=cv2.resize(rect,(16,16),interpolation=cv2.INTER_AREA);best=[]
 for flip in [False,True]:
  for rot in range(4):
   t=np.rot90(tile,rot);t=np.fliplr(t) if flip else t;corr=np.corrcoef(t[:,:,1].ravel(),small[:,:,1].ravel())[0,1];best.append((float(corr),flip,rot))
 best.sort(reverse=True);out.append({'frame':n,'D4correlations':best});print(n,best)
imgs.append(cv2.resize(tile,(128,128),interpolation=cv2.INTER_NEAREST));cv2.imwrite(str(P/'actual-cube-texture-check.png'),np.hstack(imgs));(P/'actual-cube-texture-check.json').write_text(json.dumps(out,indent=2))
