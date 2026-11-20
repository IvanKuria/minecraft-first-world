import cv2,numpy as np,json
from pathlib import Path
P=Path(__file__).parent;data=json.loads((P/'patch-fit.json').read_text());f=data['calibration']['focal_px'];K=np.float64([[f,0,426],[0,f,240],[0,0,1]])
cliff=np.float64([[-4,1,1],[-2,1,1],[-2,1,3],[-1,1,3]])
cube=np.float64([[0,1,0],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[0,0,1],[0,0,0]])
results={}
for n in [11,12]:
 ob=data['observations'][str(n)];uv=np.float64(ob['cliff']);r=np.float64(data['views'][str(n)]['rotation_vector']);t=np.float64(data['views'][str(n)]['translation']);ok,r,t=cv2.solvePnP(cliff,uv,K,None,r,t,True,cv2.SOLVEPNP_ITERATIVE)
 pred=cv2.projectPoints(cube,r,t,K,None)[0].reshape(-1,2);err=np.linalg.norm(pred-np.float64(ob['cube']),axis=1)
 print(n,'cubeheldout RMS',np.sqrt(np.mean(err**2)),err);results[n]={'cube_rms_px':float(np.sqrt(np.mean(err**2))),'cube_errors':err.tolist(),'prediction':pred.tolist()}
 im=cv2.imread(str(P/'frames'/f'hd{n:03}.png'))
 for a,b in zip(pred,np.float64(ob['cube'])):cv2.drawMarker(im,tuple(a.astype(int)),(0,0,255),cv2.MARKER_CROSS,8,1);cv2.circle(im,tuple(b.astype(int)),3,(0,255,0),1)
 cv2.imwrite(str(P/f'heldout-cube-{n}.png'),im)
(P/'heldout-cube.json').write_text(json.dumps(results,indent=2))
