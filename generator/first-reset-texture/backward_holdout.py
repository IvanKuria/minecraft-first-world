from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from profile import predict,X,Y,XT,YT
P=Path(__file__).parent;im=cv2.imread(str(P/'f-85.95.png'))[:,:,1].astype(float);obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs);free=max(json.loads((P/'fit-85.95.json').read_text()),key=lambda r:r['train']);out=[]
for sf in json.loads((P/'shared-fixed70.json').read_text()):
 px,pz=sf['shared_player_phase']
 def camera(v):
  ya,pi,h=v;return [ya,pi,h,px-.3*np.sin(ya)*np.cos(pi),pz+.3*np.cos(ya)*np.cos(pi)]
 def loss(v):
  a=predict(camera(v),70,X.ravel(),Y.ravel());a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
 fit=minimize(loss,free['parameters'][:3],method='Nelder-Mead',options={'maxiter':5000});held=np.corrcoef(predict(camera(fit.x),70,XT.ravel(),YT.ravel()),im[YT,XT].ravel())[0,1]
 row={'fixed_player_phase':[px,pz],'fixed_fov':70,'pose':camera(fit.x),'train':float(-fit.fun),'heldout':float(held)};out.append(row);print(row)
(P/'backward-holdout.json').write_text(json.dumps(out,indent=2))
