from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent; f=180/np.tan(np.deg2rad(35));K=np.float64([[f,0,320],[0,f,180],[0,0,1]])
# Four visible corners of the world's bottom plane. Convention arbitrary up to rotation/reflection.
X=np.float64([[0,0,256],[0,0,0],[256,0,0],[256,0,256]])
obs={18:{'bottom':[[43,143],[245,357],[504,334],[630,11]],'top_back':[[109,103],[520,5]]},19:{'bottom':[[70,139],[245,354],[497,329],[586,23]],'top_back':[[132,104],[500,20]]},20:{'bottom':[[90,128],[248,339],[489,314],[559,21]],'top_back':[[151,98],[489,18]]}}
def R(y,p):
 s,c=np.sin(y),np.cos(y);sp,cp=np.sin(p),np.cos(p)
 return np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]])
def proj(X,v):
 cam=X@R(v[0],v[1]).T+v[2:5];return cam[:,:2]/cam[:,2,None]*f+[320,180]
results={}
for n,o in obs.items():
 uv=np.float64(o['bottom']);ok,r,t=cv2.solvePnP(X,uv,K,None,flags=cv2.SOLVEPNP_SQPNP);rot=cv2.Rodrigues(r)[0]
 v0=np.r_[np.arctan2(rot[0,2],rot[0,0]),np.arcsin(np.clip(-rot[2,1],-1,1)),t.ravel()]
 fit=least_squares(lambda v:(proj(X,v)-uv).ravel(),v0,max_nfev=2000);v=fit.x;C=-R(*v[:2]).T@v[2:5];err=np.linalg.norm(proj(X,v)-uv,axis=1)
 topuv=np.float64(o['top_back'])
 ht=least_squares(lambda h:(proj(np.float64([[0,h[0],256],[256,h[0],256]]),v)-topuv).ravel(),[43.])
 h=ht.x[0];print(n,'cam',C,'angles',np.rad2deg(v[:2]),'RMSbottom',np.sqrt(np.mean(err**2)),'height',h,'toperr',np.sqrt(np.mean(ht.fun**2)*2))
 results[n]={'camera_position':C.tolist(),'yaw_pitch_degrees':np.rad2deg(v[:2]).tolist(),'bottom_error_px':err.tolist(),'bottom_rms_px':float(np.sqrt(np.mean(err**2))),'height_fitted_from_top_pixels':float(h),'top_height_fit_rms_px':float(np.sqrt(np.mean(ht.fun**2)*2)),'heldout_height64_top_rms_px':float(np.sqrt(np.mean((proj(np.float64([[0,64,256],[256,64,256]]),v)-topuv)**2)*2)),'v':v.tolist(),'bottom_observed':o['bottom'],'top_observed':o['top_back']}
 im=cv2.imread(str(P/'boundary'/f'f{n:03}.png'));pp=proj(X,v).astype(int);cv2.polylines(im,[pp],True,(0,255,255),1,cv2.LINE_AA)
 for j,p in enumerate(pp):cv2.putText(im,'ABCD'[j],tuple(p+[-10,-5]),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
 for height,color in [(43,(255,0,0)),(64,(0,0,255)),(h,(0,255,0))]:
  tp=proj(np.float64([[0,height,256],[256,height,256]]),v).astype(int);cv2.line(im,tuple(tp[0]),tuple(tp[1]),color,1,cv2.LINE_AA)
 cv2.imwrite(str(P/f'boundary-fit-{n}.png'),im)
(P/'boundary-fit.json').write_text(json.dumps(results,indent=2))
