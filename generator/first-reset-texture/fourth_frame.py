from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from profile import predict,X,Y,XT,YT
P=Path(__file__).parent;fit=max(json.loads((P/'shared-phase.json').read_text()),key=lambda r:r['mean_train']);fov=fit['fov'];px,pz=fit['shared_player_phase'];yaw,pitch=np.radians(json.loads((P/'manhattan-86.50.json').read_text())['angles_deg']);im=cv2.imread(str(P/'f-86.50.png'))[:,:,1].astype(float);obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)
def camera(v):
 ya,pi,h=v;return [ya,pi,h,px-.3*np.sin(ya)*np.cos(pi),pz+.3*np.cos(ya)*np.cos(pi)]
def loss(v):
 if not 1<v[2]<15:return 1
 a=predict(camera(v),fov,X.ravel(),Y.ravel());a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
results=[]
for rot in range(4):
 starts=sorted([(loss([yaw+rot*np.pi/2,pitch,h]),h)for h in np.arange(2,9,.025)])[:8]
 fits=[minimize(loss,[yaw+rot*np.pi/2,pitch,h],method='Nelder-Mead',options={'maxiter':2500})for _,h in starts];best=min(fits,key=lambda r:r.fun);held=np.corrcoef(predict(camera(best.x),fov,XT.ravel(),YT.ravel()),im[YT,XT].ravel())[0,1]
 row={'quadrant':rot,'fixed_shared_fov':fov,'fixed_player_phase':[px,pz],'parameters':camera(best.x),'train':float(-best.fun),'heldout':float(held)};results.append(row);print(row,flush=True)
(P/'fourth-frame-heldout.json').write_text(json.dumps(results,indent=2))
