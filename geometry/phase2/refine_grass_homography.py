from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float);focal=180/np.tan(np.deg2rad(35));rec=[r for r in json.loads((P/'grass-height-sequence.json').read_text()) if r['t_requested']==152.951][0];yaw,pitch,H,px,pz=rec['parameters'];s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch)
# Normalized image coordinates for stable unconstrained plane homography.
corners=np.float32([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]);screen=corners*100+[350,315];rx=(screen[:,0]-320)/focal;ry=(screen[:,1]-180)/focal;dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp;world=np.array([px-H*dx/dy,pz-H*dz/dy]).T;offset=np.floor(world.mean(axis=0));world-=offset;h=cv2.getPerspectiveTransform(corners,np.float32(world));v0=h.ravel()[:8]
im=cv2.imread(str(P/'f-152.951.png'))[:,:,1].astype(float);ys,xs=np.mgrid[278:355,300:405];coords=np.array([(xs.ravel()-350)/100,(ys.ravel()-315)/100]);YT,XT=np.mgrid[312:350,220:295];testcoords=np.array([(XT.ravel()-350)/100,(YT.ravel()-315)/100])
def texture(v,xy):
 a,b=xy;den=v[6]*a+v[7]*b+1;u=((v[0]*a+v[1]*b+v[2])/den*16)%16;w=((v[3]*a+v[4]*b+v[5])/den*16)%16;u0=np.floor(u).astype(int);w0=np.floor(w).astype(int);du=u-u0;dw=w-w0
 return tile[w0,u0]*(1-du)*(1-dw)+tile[w0,(u0+1)%16]*du*(1-dw)+tile[(w0+1)%16,u0]*(1-du)*dw+tile[(w0+1)%16,(u0+1)%16]*du*dw
def norm(a):return(a-a.mean())/(a.std()+1e-9)
results=[]
for sigma in [0.,.6,1.]:
 obs=im[ys,xs];obs=cv2.GaussianBlur(obs,(0,0),sigma)if sigma else obs;obs=norm(obs).ravel()
 def prediction(v):
  val=texture(v,coords).reshape(xs.shape);return cv2.GaussianBlur(val,(0,0),sigma).ravel()if sigma else val.ravel()
 sol=least_squares(lambda v:norm(prediction(v))-obs,v0,max_nfev=600,diff_step=1e-6,ftol=1e-10,xtol=1e-10);pred=prediction(sol.x);corr=np.corrcoef(pred,obs)[0,1];held=np.corrcoef(texture(sol.x,testcoords),im[YT,XT].ravel())[0,1];print(sigma,corr,held,sol.nfev,flush=True);results.append({'sigma':sigma,'train':float(corr),'heldout':float(held),'homography':np.r_[sol.x,1].reshape(3,3).tolist(),'world_offset':offset.tolist(),'nfev':sol.nfev})
(P/'grass-homography-fit.json').write_text(json.dumps(results,indent=2))
