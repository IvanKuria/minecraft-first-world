from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from model import predict
P=Path(__file__).parent;best=max([r for r in json.loads((P/'joint-opening.json').read_text())if r['mode']=='fixed70'],key=lambda r:r['mean_train']);px,pz=best['player_phase'];out=[]
for name,angles in [('1.90',(318,-20)),('1.80',(319,-29))]:
 im=cv2.imread(str(P/f'f-{name}.png'))[:,:,1].astype(float)
 if name=='1.90':Y,X=np.mgrid[310:338:2,380:620:2];YT,XT=np.mgrid[342:355,440:600]
 else:Y,X=np.mgrid[335:347:2,380:620:2];YT,XT=np.mgrid[350:359,440:600]
 obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)
 def camera(v):
  ya,pi=v;return [ya,pi,.62+.3*np.sin(pi),px-.3*np.sin(ya)*np.cos(pi),pz+.3*np.cos(ya)*np.cos(pi)]
 def loss(v):
  a=predict(camera(v),70,X.ravel(),Y.ravel());a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
 starts=[]
 for ya in np.arange(angles[0]-5,angles[0]+5.01,.5):
  for pi in np.arange(angles[1]-4,angles[1]+4.01,.5):
   v=np.radians([ya,pi]);starts.append((loss(v),v))
 starts.sort(key=lambda r:r[0]);fits=[minimize(loss,v,method='Nelder-Mead',options={'maxiter':1500,'xatol':1e-9})for _,v in starts[:8]];fit=min(fits,key=lambda r:r.fun);held=np.corrcoef(predict(camera(fit.x),70,XT.ravel(),YT.ravel()),im[YT,XT].ravel())[0,1]
 row={'frame_name':name,'phase_fixed_from_later':list(best['player_phase']),'fov_fixed':70,'player_height_above_plane_fixed':.62,'angles_deg':np.degrees(fit.x).tolist(),'train':float(-fit.fun),'heldout':float(held),'parameters':camera(fit.x)};out.append(row);print(row,flush=True)
(P/'backward-link.json').write_text(json.dumps(out,indent=2))
