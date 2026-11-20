from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from profile import predict,X,Y,XT,YT
P=Path(__file__).parent;s=max(json.loads((P/'shared-phase.json').read_text()),key=lambda r:r['mean_train']);p=s['shared_player_phase'];fov=s['fov'];im=cv2.imread(str(P/'f-86.50.png'))[:,:,1].astype(float);obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)
def camera(v):
 ya,pi,h=v;return [ya,pi,h,p[0]-.3*np.sin(ya)*np.cos(pi),p[1]+.3*np.cos(ya)*np.cos(pi)]
def loss(v):
 a=predict(camera(v),fov,X.ravel(),Y.ravel());a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
free=max(json.loads((P/'fit-86.50.json').read_text()),key=lambda r:r['train']);start=np.array(free['parameters'][:3]);out=[]
for dy in [-1,0,1]:
 for dh in [-.1,0,.1]:
  v=start+np.array([0,np.radians(dy),dh]);fit=minimize(loss,v,method='Nelder-Mead',options={'maxiter':5000});held=np.corrcoef(predict(camera(fit.x),fov,XT.ravel(),YT.ravel()),im[YT,XT].ravel())[0,1];out.append({'parameters':camera(fit.x),'train':float(-fit.fun),'heldout':float(held),'phase_fixed':p,'fov_fixed':fov})
out.sort(key=lambda r:-r['train']);(P/'fourth-frame-warm-start.json').write_text(json.dumps(out,indent=2));print(out[0])
# Independent original-class render calibration using this lane's exact sampler.
control=cv2.imread(str(P.parents[1]/'runtime/render-texture-control/actual-game-render.png'))[:,:,1].astype(float);yy,xx=np.mgrid[278:355,300:405];ya,pi=np.radians([31.65757,14.64737]);h=10.3+.3*np.sin(pi);cx=223.41498-.3*np.sin(ya)*np.cos(pi);cz=123.76827+.3*np.cos(ya)*np.cos(pi);corr=np.corrcoef(predict([ya,pi,h,cx,cz],70,xx.ravel(),yy.ravel(),True),control[yy,xx].ravel())[0,1];(P/'actual-renderer-control.json').write_text(json.dumps({'actual_source':'runtime/render-texture-control/actual-game-render.png','known_camera_height':h,'known_camera_phase':[cx%1,cz%1],'nearest_prediction_correlation':float(corr)},indent=2));print('actualrenderer',corr)
