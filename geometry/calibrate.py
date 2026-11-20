import cv2,numpy as np,json
from pathlib import Path
from scipy.optimize import least_squares
P=Path(__file__).parent
# x right, y up, z toward near edge; one observed cube.
xyz=np.float64([[0,1,0],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[0,0,1],[0,0,0]])
obs={11:[[409,148],[440,150],[446,166],[412,162],[438,191],[409,189],[404,177]],12:[[432,160],[470,163],[481,183],[435,180],[469,215],[433,211],[425,186]],13:[[452.5,175.8],[503.3,179.2],[524.2,211.2],[461.7,206.2],[503.5,248.7],[448,241],[440.8,211.7]]}
f=180/np.tan(np.deg2rad(35))
for no,uv in obs.items():
 uv=np.float64(uv); xyzfull=xyz.copy(); uvfull=uv.copy(); keep=[0,1,2,4,5,6]; xyz=xyzfull[keep]; uv=uv[keep]; k=np.array([[f,0,320],[0,f,180],[0,0,1.]])
 ok,r,t=cv2.solvePnP(xyz,uv,k,None,flags=cv2.SOLVEPNP_SQPNP)
 pp=cv2.projectPoints(xyz,r,t,k,None)[0].reshape(-1,2)
 print(no,'fixedf rms',np.sqrt(np.mean(np.sum((pp-uv)**2,axis=1))),'r',r.ravel(),'t',t.ravel())
 im=cv2.imread(str(P/'frames'/f'f{no:03}.png'))
 for i,(a,b) in enumerate(zip(uv,pp)):
  cv2.circle(im,tuple(np.round(a).astype(int)),3,(0,255,0),1);cv2.drawMarker(im,tuple(np.round(b).astype(int)),(0,0,255),cv2.MARKER_CROSS,5,1)
 cv2.imwrite(str(P/f'fit-{no}.png'),im)
 # Joint per-view unconstrained focal diagnostic.
 def resid(v):
  kk=np.array([[v[6],0,320],[0,v[6],180],[0,0,1.]])
  return (cv2.projectPoints(xyz,v[:3],v[3:6],kk,None)[0].reshape(-1,2)-uv).ravel()
 sol=least_squares(resid,np.r_[r.ravel(),t.ravel(),f],max_nfev=3000)
 print('freef rms',np.sqrt(np.mean(sol.fun**2)*2),'f',sol.x[-1]); xyz=xyzfull
