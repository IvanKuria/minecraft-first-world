from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent;tile=cv2.imread(str(P.parents[1]/'geometry/phase2/terrain.png'))[:16,:16,1].astype(float)
Y,X=np.mgrid[220:330:5,380:620:5];YT,XT=np.mgrid[335:355,400:600]
def lookup(wx,wz,nearest=False):
 if nearest:return tile[np.floor((wz%1)*15.984).astype(int)%16,np.floor((wx%1)*15.984).astype(int)%16]
 u=((wx%1)*15.984-.5)%16;v=((wz%1)*15.984-.5)%16;ui=np.floor(u).astype(int);vi=np.floor(v).astype(int);a=u-ui;b=v-vi
 return tile[vi,ui]*(1-a)*(1-b)+tile[vi,(ui+1)%16]*a*(1-b)+tile[(vi+1)%16,ui]*(1-a)*b+tile[(vi+1)%16,(ui+1)%16]*a*b

def predict(v,fov,x,y,nearest=False):
 f=180/np.tan(np.deg2rad(fov/2));rx=(x+.5-320)/f;ry=(y+.5-180)/f;yaw,pitch,h,px,pz=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
 return lookup(px-h*dx/dy,pz-h*dz/dy,nearest)
def player(v):
 ya,pi,h,cx,cz=v
 return [(cx+.3*np.sin(ya)*np.cos(pi))%1,(cz-.3*np.cos(ya)*np.cos(pi))%1]
if __name__=='__main__':
 rows=[]
 for name in ['86.00','86.15','86.30']:
  im=cv2.imread(str(P/f'f-{name}.png'))[:,:,1].astype(float);obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)
  baseline=max(json.loads((P/f'fit-{name}.json').read_text()),key=lambda r:r['train'])['parameters'];records={}
  for path in [np.arange(70,67.99,-.25),np.arange(70,72.01,.25)]:
   start=np.array(baseline)
   for fov in path:
    def loss(v):
     a=predict(v,fov,X.ravel(),Y.ravel());a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
    fit=minimize(loss,start,method='Nelder-Mead',options={'maxiter':1600,'xatol':1e-7});start=fit.x
    record={'frame_name':name,'fov':float(fov),'parameters':fit.x.tolist(),'train':float(-fit.fun),'heldout':float(np.corrcoef(predict(fit.x,fov,XT.ravel(),YT.ravel()),im[YT,XT].ravel())[0,1]),'nearest_heldout':float(np.corrcoef(predict(fit.x,fov,XT.ravel(),YT.ravel(),True),im[YT,XT].ravel())[0,1]),'player_phase':player(fit.x),'player_height_above_plane':float(fit.x[2]-.3*np.sin(fit.x[1]))}
    if fov not in records or record['train']>records[fov]['train']:records[fov]=record
  out=[records[k] for k in sorted(records)];rows+=out;print(name,[(r['fov'],round(r['train'],3),round(r['heldout'],3),[round(v,3) for v in r['player_phase']])for r in out],flush=True)
  (P/'fov-profile.json').write_text(json.dumps(rows,indent=2))
