"""Adapted with credit from geometry/phase2/fit_late_grass.py; separate first-reset ROI."""
from pathlib import Path
import cv2,numpy as np,json,sys,time
from scipy.optimize import minimize
P=Path(__file__).parent;name=sys.argv[1]
tile=cv2.imread(str(P.parents[1]/'geometry/phase2/terrain.png'))[:16,:16,1].astype(float)
im=cv2.imread(str(P/f'f-{name}.png'))[:,:,1].astype(float)
Y,X=np.mgrid[200:290:4,360:620:4];YT,XT=np.mgrid[300:340,480:620]
if name=="2.00":Y,X=np.mgrid[240:300:3,360:620:3];YT,XT=np.mgrid[310:350,480:620]
if name=="1.90":Y,X=np.mgrid[310:338:2,380:620:2];YT,XT=np.mgrid[342:355,440:600]
if name=="1.80":Y,X=np.mgrid[335:347:2,380:620:2];YT,XT=np.mgrid[350:359,440:600]
f=180/np.tan(np.deg2rad(35));rx=(X.ravel()+.5-320)/f;ry=(Y.ravel()+.5-180)/f;rxt=(XT.ravel()+.5-320)/f;ryt=(YT.ravel()+.5-180)/f
obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs);phase=np.array([(x/16,z/16)for x in range(16)for z in range(16)]);out=[]
angles=json.loads((P/f'manhattan-{name}.json').read_text())['angles_deg'];started=time.monotonic()
def lookup(wx,wz):
 u=((wx%1)*15.984-.5)%16;v=((wz%1)*15.984-.5)%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
 return tile[v0,u0]*(1-a)*(1-b)+tile[v0,(u0+1)%16]*a*(1-b)+tile[(v0+1)%16,u0]*(1-a)*b+tile[(v0+1)%16,(u0+1)%16]*a*b
def values(v,x=rx,y=ry):
 yaw,pitch,h,px,pz=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*x-s*sp*y+s*cp;dy=-cp*y-sp;dz=s*x+c*sp*y-c*cp;return lookup(px-h*dx/dy,pz-h*dz/dy)
for rot in range(4):
 yaw,pitch=np.deg2rad([angles[0]+90*rot,angles[1]]);s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp;best=[]
 for h in np.arange(.2,4.001,.02):
  val=lookup(phase[:,0,None]-h*dx/dy,phase[:,1,None]-h*dz/dy);val-=val.mean(axis=1)[:,None];scores=val@obs/(np.linalg.norm(val,axis=1)+1e-9);i=np.argmax(scores);best.append((float(scores[i]),h,*phase[i]))
 best.sort(reverse=True)
 def loss(v):
  if not (0<v[2]<30):return 1
  a=values(v);a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
 fits=[minimize(loss,[yaw,pitch,h,px,pz],method='Nelder-Mead',options={'maxiter':2000})for _,h,px,pz in best[:8]];fits.sort(key=lambda s:s.fun);sol=fits[0];held=np.corrcoef(values(sol.x,rxt,ryt),im[YT,XT].ravel())[0,1]
 rec={'rotation':rot,'parameters':sol.x.tolist(),'parameters_degrees_height_phase':[float(np.degrees(sol.x[0])),float(np.degrees(sol.x[1])),*sol.x[2:].tolist()],'train':float(-sol.fun),'heldout':float(held),'other_fits':[{'score':float(-s.fun),'height':float(s.x[2])}for s in fits],'elapsed_seconds':time.monotonic()-started};out.append(rec);print(rec,flush=True);(P/f'fit-{name}.json').write_text(json.dumps(out,indent=2))
