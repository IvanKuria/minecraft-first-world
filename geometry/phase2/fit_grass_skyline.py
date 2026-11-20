from pathlib import Path
import cv2,numpy as np,json,time
from scipy.optimize import minimize
P=Path(__file__).parent;frame=cv2.imread(str(P/'f-152.899.png'));tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float)
yy,xx=np.mgrid[278:355:2,300:405:2];f=180/np.tan(np.deg2rad(35));rx=(xx.ravel()-320)/f;ry=(yy.ravel()-180)/f
im=frame[:,:,1].astype(float);obs=im[yy,xx].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)

def values(v,x=rx,y=ry,tiles=tile):
 yaw,pitch,h,phase_x,phase_z=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch)
 dx=c*x-s*sp*y+s*cp;dy=-cp*y-sp;dz=s*x+c*sp*y-c*cp
 u=((phase_x-h*dx/dy)*16)%16;v=((phase_z-h*dz/dy)*16)%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
 return tiles[v0,u0]*(1-a)*(1-b)+tiles[v0,(u0+1)%16]*a*(1-b)+tiles[(v0+1)%16,u0]*(1-a)*b+tiles[(v0+1)%16,(u0+1)%16]*a*b

def score(v,tx=tile):
 val=values(v,tiles=tx);val-=val.mean();return np.dot(val,obs)/(np.linalg.norm(val)+1e-9)
# Sweep scale and allpixelphases with orientation measured fromskyline.
phase=np.array([(a/16,b/16)for a in range(16)for b in range(16)]);best=[];start=time.time();yaw,pitch=np.deg2rad([31.56,14.63])
s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
for h in np.arange(4,20.001,.04):
 u=((phase[:,0,None]-h*dx/dy)*16)%16;v=((phase[:,1,None]-h*dz/dy)*16)%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
 val=tile[v0,u0]*(1-a)*(1-b)+tile[v0,(u0+1)%16]*a*(1-b)+tile[(v0+1)%16,u0]*(1-a)*b+tile[(v0+1)%16,(u0+1)%16]*a*b
 val-=val.mean(axis=1)[:,None];scores=val@obs/(np.linalg.norm(val,axis=1)+1e-9);i=np.argmax(scores);best.append([float(scores[i]),h,*phase[i]])
best.sort(reverse=True);print('coarse top',best[:10],flush=True)
res=[]
for sc,h,px,pz in best[:12]:
 sol=minimize(lambda v:-score(v),[yaw,pitch,h,px,pz],method='Nelder-Mead',options={'maxiter':1200,'xatol':1e-7});res.append((float(-sol.fun),sol.x))
res.sort(key=lambda x:-x[0]);print('FINAL',[(a,b.tolist())for a,b in res[:5]],'seconds',time.time()-start)
sc,v=res[0];out={'score':sc,'params':v.tolist(),'coarse_top':best[:20],'method':'skylineorientation fixedcoarseheightphase then5parameterlocalrefine','training_rect':[300,278,405,355]};(P/'grass-skyline-fit.json').write_text(json.dumps(out,indent=2))
Y,X=np.mgrid[:360,:640];pred=values(v,(X.ravel()-320)/f,(Y.ravel()-180)/f).reshape(360,640);cv2.imwrite(str(P/'grass-skyline-prediction.png'),np.uint8(np.clip(pred,0,255)))
for box in [(300,278,405,355),(300,240,385,272),(220,312,295,350)]:
 x0,y0,x1,y1=box;print('heldoutcorr',box,np.corrcoef(pred[y0:y1,x0:x1].ravel(),im[y0:y1,x0:x1].ravel())[0,1])
