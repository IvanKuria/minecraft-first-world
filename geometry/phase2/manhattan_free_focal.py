from pathlib import Path
import numpy as np,cv2,json
from scipy.optimize import least_squares,minimize
P=Path(__file__).parent;im=cv2.imread(str(P/'f-153.300.png'));lines=cv2.createLineSegmentDetector().detect(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))[0].reshape(-1,4);lengths=np.linalg.norm(lines[:,2:]-lines[:,:2],axis=1);lines=lines[lengths>12];lengths=lengths[lengths>12];l=np.cross(np.c_[lines[:,:2],np.ones(len(lines))],np.c_[lines[:,2:],np.ones(len(lines))]);v0=np.r_[np.deg2rad([38.2915,42.2794]),70.]
def normals(fov):f=180/np.tan(np.deg2rad(fov/2));K=np.array([[f,0,320],[0,f,180],[0,0,1]]);n=l@K;return n/np.linalg.norm(n,axis=1)[:,None]
def R(v):y,p=v[:2];s,c=np.sin(y),np.cos(y);sp,cp=np.sin(p),np.cos(p);return np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]])
mat=normals(70)@R(v0);groups=np.argmin(np.abs(mat),axis=1);err=np.min(np.abs(mat),axis=1);out=[];rng=np.random.default_rng(518)
for threshold in [.008,.012,.015]:
 ix=np.flatnonzero(err<threshold)
 def residual(v,indices=ix):return (normals(v[2])@R(v))[indices,groups[indices]]*np.sqrt(np.minimum(lengths[indices],50)/25)
 sol=least_squares(residual,v0,bounds=([0,0,40],[np.pi/2,np.pi/2,110]),loss='soft_l1',f_scale=.003,max_nfev=500);boots=[]
 for _ in range(100):
  ii=np.concatenate([rng.choice(ix[groups[ix]==g],size=np.sum(groups[ix]==g),replace=True)for g in range(3)]);ss=least_squares(lambda v:residual(v,ii),sol.x,bounds=([0,0,40],[np.pi/2,np.pi/2,110]),loss='soft_l1',f_scale=.003,max_nfev=200);boots.append(float(ss.x[2]))
 rec={'selection_threshold':threshold,'axis_counts':[int(np.sum(groups[ix]==g))for g in range(3)],'yaw_pitch_degrees':np.rad2deg(sol.x[:2]).tolist(),'free_vertical_FOV':float(sol.x[2]),'bootstrap_FOV_2_50_98_percentile':np.percentile(boots,[2,50,98]).tolist(),'warning':'selection initialized fromfixed70orientation; correlated line edges and non-geometricfalseinliers not coveredbybootstrap'};out.append(rec);print(rec,flush=True)
(P/'manhattan-free-focal.json').write_text(json.dumps(out,indent=2))
