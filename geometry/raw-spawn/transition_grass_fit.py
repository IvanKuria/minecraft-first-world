from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent;tile=cv2.imread(str(P.parent/'phase2/terrain.png'))[:16,:16,1].astype(float);im=cv2.imread(str(P.parents[1]/'timing/original-reset2-2952.png'))[:,:,1].astype(float);Y,X=np.mgrid[270:335:3,50:200:3];YT,XT=np.mgrid[338:358,50:200];f=180/np.tan(np.deg2rad(35));rx=(X.ravel()+.5-320)/f;ry=(Y.ravel()+.5-180)/f;rxt=(XT.ravel()+.5-320)/f;ryt=(YT.ravel()+.5-180)/f;obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs);phase=np.array([(x/16,z/16)for x in range(16)for z in range(16)]);out=[]
def lookup(wx,wz):
 u=((wx%1)*15.984-.5)%16;v=((wz%1)*15.984-.5)%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
 return tile[v0,u0]*(1-a)*(1-b)+tile[v0,(u0+1)%16]*a*(1-b)+tile[(v0+1)%16,u0]*(1-a)*b+tile[(v0+1)%16,(u0+1)%16]*a*b
def values(v,x=rx,y=ry):
 yaw,pitch,h,px,pz=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*x-s*sp*y+s*cp;dy=-cp*y-sp;dz=s*x+c*sp*y-c*cp;return lookup(px-h*dx/dy,pz-h*dz/dy)
for rot in range(4):
 yaw,pitch=np.deg2rad([32.212183+90*rot,14.711907]);s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp;best=[]
 for h in np.arange(4,16.001,.05):
  val=lookup(phase[:,0,None]-h*dx/dy,phase[:,1,None]-h*dz/dy);val-=val.mean(axis=1)[:,None];scores=val@obs/(np.linalg.norm(val,axis=1)+1e-9);i=np.argmax(scores);best.append((float(scores[i]),h,*phase[i]))
 best.sort(reverse=True)
 def loss(v):a=values(v);a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
 fits=[minimize(loss,[yaw,pitch,h,px,pz],method='Nelder-Mead',options={'maxiter':2000})for _,h,px,pz in best[:8]];fits.sort(key=lambda s:s.fun);sol=fits[0];held=np.corrcoef(values(sol.x,rxt,ryt),im[YT,XT].ravel())[0,1];rec={'rotation':rot,'parameters':sol.x.tolist(),'train':float(-sol.fun),'heldout':float(held),'other_fits':[{'score':float(-s.fun),'height':float(s.x[2])}for s in fits]};out.append(rec);print(rec,flush=True);(P/'transition-grass-fit.json').write_text(json.dumps(out,indent=2))
