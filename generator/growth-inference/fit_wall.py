from pathlib import Path
import cv2,numpy as np,json,time
from scipy.optimize import minimize
P=Path(__file__).parent;tile=cv2.imread(str(P.parents[1]/'geometry/phase2/terrain.png'))[:16,16:32,1].astype(float);im=cv2.imread(str(P/'boundary-164.png'))[:,:,1].astype(float)
Y,X=np.mgrid[60:290:7,30:270:7];YT,XT=np.mgrid[80:290:2,285:410:2];f=180/np.tan(np.deg2rad(35));rx=(X.ravel()+.5-320)/f;ry=(Y.ravel()+.5-180)/f;obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs);phases=np.array([(i/16,j/16)for i in range(16)for j in range(16)])
def lookup(u,v):
 u=((u%1)*15.984-.5)%16;v=((v%1)*15.984-.5)%16;ui=np.floor(u).astype(int);vi=np.floor(v).astype(int);a=u-ui;b=v-vi
 return tile[vi,ui]*(1-a)*(1-b)+tile[vi,(ui+1)%16]*a*(1-b)+tile[(vi+1)%16,ui]*(1-a)*b+tile[(vi+1)%16,(ui+1)%16]*a*b
def coords(v,x,y):
 yaw,pitch,d,u,w=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);rx=(x+.5-320)/f;ry=(y+.5-180)/f;dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
 return u-d*dx/dz,w+d*dy/dz
def predict(v,x,y):return lookup(*coords(v,x,y))
angles=json.loads((P/'manhattan-164.json').read_text())['angles_deg'];out=[]
for rot in [0,1,2,3]:
 ya,pi=np.deg2rad([angles[0]+90*rot,angles[1]]);s,c=np.sin(ya),np.cos(ya);sp,cp=np.sin(pi),np.cos(pi);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
 if np.any(dz>-.05):continue
 best=[]
 for d in np.arange(.3,15,.025):
  a=lookup(phases[:,0,None]-d*dx/dz,phases[:,1,None]+d*dy/dz);a-=a.mean(axis=1)[:,None];sc=a@obs/(np.linalg.norm(a,axis=1)+1e-9);i=np.argmax(sc);best.append((sc[i],d,*phases[i]))
 best.sort(reverse=True)
 def loss(v):
  if not .05<v[2]<30:return 1
  a=predict(v,X.ravel(),Y.ravel());a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
 fits=[minimize(loss,[ya,pi,d,u,w],method='Nelder-Mead',options={'maxiter':4000})for _,d,u,w in best[:12]];fits.sort(key=lambda r:r.fun)
 for r in fits[:6]:
  rec={'rotation':rot,'parameters':r.x.tolist(),'train':float(-r.fun),'heldout':float(np.corrcoef(predict(r.x,XT.ravel(),YT.ravel()),im[YT,XT].ravel())[0,1])};out.append(rec);print(rec,flush=True)
 (P/'wall-fit.json').write_text(json.dumps(out,indent=2))
