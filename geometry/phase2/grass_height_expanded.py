from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares,minimize
P=Path(__file__).parent;tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float);f=180/np.tan(np.deg2rad(35));phase=np.array([(a/16,b/16)for a in range(16)for b in range(16)])
Y,X=np.mgrid[278:355:3,300:405:3];rx=(X.ravel()-320)/f;ry=(Y.ravel()-180)/f
YT,XT=np.mgrid[312:350,220:295];rxt=(XT.ravel()-320)/f;ryt=(YT.ravel()-180)/f

def tex(v,x,y):
 yaw,pitch,h,px,pz=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*x-s*sp*y+s*cp;dy=-cp*y-sp;dz=s*x+c*sp*y-c*cp
 u=((px-h*dx/dy)*16)%16;v=((pz-h*dz/dy)*16)%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
 return tile[v0,u0]*(1-a)*(1-b)+tile[v0,(u0+1)%16]*a*(1-b)+tile[(v0+1)%16,u0]*(1-a)*b+tile[(v0+1)%16,(u0+1)%16]*a*b

out=[]
for name in ['152.899','152.951']:
 frame=cv2.imread(str(P/f'f-{name}.png'));im=frame[:,:,1].astype(float);ob=im[Y,X].ravel();ob-=ob.mean();ob/=np.linalg.norm(ob)
 # Independent skylineorientation estimate for this specificframe.
 bb,gg,rr=cv2.split(frame.astype(float));sky=(bb>160)&(gg>120)&(bb-gg>15)&(gg-rr>25);ys=np.argmax(~sky,axis=0);xs=np.arange(640)
 def skyline(v):xc,yc,ml,mr=v;return yc+np.where(xs<xc,ml,mr)*(xs-xc)
 sf=least_squares(lambda v:skyline(v)-ys,[240,137,-.05,.25],loss='soft_l1',f_scale=1.5,bounds=([30,0,-2,0],[600,300,0,2]))
 xc,yc,ml,mr=map(float,sf.x);q=np.poly1d([1,-yc]);vp1=xc+q/ml-320;vp2=xc+q/mr-320;hy=min(np.roots(vp1*vp2+np.poly1d([1,-180])**2+f*f));vx=xc+(hy-yc)/ml;pitch=np.arctan((180-hy)/f);yaw=np.arctan(f/((vx-320)*np.cos(pitch)))
 s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
 best=[]
 for h in np.arange(4,74.001,.1):
  u=((phase[:,0,None]-h*dx/dy)*16)%16;v=((phase[:,1,None]-h*dz/dy)*16)%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
  val=tile[v0,u0]*(1-a)*(1-b)+tile[v0,(u0+1)%16]*a*(1-b)+tile[(v0+1)%16,u0]*(1-a)*b+tile[(v0+1)%16,(u0+1)%16]*a*b;val-=val.mean(axis=1)[:,None];sc=val@ob/(np.linalg.norm(val,axis=1)+1e-9);i=np.argmax(sc);best.append((float(sc[i]),h,*phase[i]))
 best.sort(reverse=True)
 def loss(v):a=tex(v,rx,ry);a-=a.mean();return -np.dot(a,ob)/(np.linalg.norm(a)+1e-9)
 fits=[]
 for sc,h,px,pz in best[:12]:
  sol=minimize(loss,[yaw,pitch,h,px,pz],method='Nelder-Mead',options={'maxiter':1600});fits.append(sol)
 fits.sort(key=lambda a:a.fun);sol=fits[0];corr=np.corrcoef(tex(sol.x,rxt,ryt),im[YT,XT].ravel())[0,1]
 rec={'file':f'f-{name}.png','t_requested':float(name),'height':float(sol.x[2]),'parameters':sol.x.tolist(),'training_correlation':float(-sol.fun),'heldout_correlation':float(corr),'skyline_yaw_pitch':np.rad2deg([yaw,pitch]).tolist(),'coarse_top':best[:6],'other_local_fits':[{'correlation':float(-s.fun),'height':float(s.x[2])}for s in fits]};out.append(rec);print(rec,flush=True)
(P/'grass-height-expanded.json').write_text(json.dumps(out,indent=2))
