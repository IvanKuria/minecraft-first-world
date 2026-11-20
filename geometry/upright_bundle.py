import cv2,numpy as np,json
from scipy.optimize import least_squares
from pathlib import Path
P=Path(__file__).parent;data=json.loads((P/'patch-fit.json').read_text());f=data['calibration']['focal_px']
cliff=np.float64([[-4,1,1],[-2,1,1],[-2,1,3],[-1,1,3]])
cube=np.float64([[0,1,0],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[0,0,1],[0,0,0]])
X=np.vstack([cube,cliff]);obs={n:np.float64(data['observations'][str(n)]['cube']+data['observations'][str(n)]['cliff']) for n in [11,12,13]}
def R(y,p):
 s,c=np.sin(y),np.cos(y);sp,cp=np.sin(p),np.cos(p)
 return np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]])
def project(X,v,fx,fy):
 cam=X@R(v[0],v[1]).T+v[2:5]
 return cam[:,:2]/cam[:,2,None]*[fx,fy]+[426,240]
x0=[f,f]
for n in [11,12,13]:
 a=data['views'][str(n)];r=cv2.Rodrigues(np.float64(a['rotation_vector']))[0];x0.extend([np.arctan2(r[0,2],r[0,0]),np.arcsin(-r[2,1])]+a['translation'])
def residual(v):return np.concatenate([(project(X,v[2+i*5:7+i*5],v[0],v[1])-obs[n]).ravel() for i,n in enumerate([11,12,13])])
sol=least_squares(residual,x0,max_nfev=5000)
print('fit',sol.x[:2],'rms',np.sqrt(np.mean(sol.fun**2)*2))
for i,n in enumerate([11,12,13]):
 v=sol.x[2+i*5:7+i*5];err=np.linalg.norm(project(X,v,*sol.x[:2])-obs[n],axis=1);print(n,'all err',err)
 # Hold out cliff, estimate 5DOF camera with frozen shared intrinsics.
 fit=least_squares(lambda v:(project(cube,v,*sol.x[:2])-obs[n][:7]).ravel(),v)
 er=np.linalg.norm(project(cliff,fit.x,*sol.x[:2])-obs[n][7:],axis=1)
 print(n,'heldoutcliff',er,'rms',np.sqrt(np.mean(er**2)))
print(sol.x)
