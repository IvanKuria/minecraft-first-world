"""Test the previously fixed .62 height in the failed opening frame35 bridge."""
from pathlib import Path
import sys,json,cv2,numpy as np
from scipy.optimize import minimize
P=Path(__file__).parent;Q=P.parents[1]/'generator/opening-texture';sys.path.insert(0,str(Q))
from model import predict
best=max([r for r in json.loads((Q/'joint-opening.json').read_text())if r['mode']=='fixed70'],key=lambda r:r['mean_train']);px,pz=best['player_phase']
im=cv2.imread(str(Q/'f-1.80.png'));green=im[:,:,1].astype(float)
Y,X=np.mgrid[335:347:2,380:620:2];YT,XT=np.mgrid[350:359,440:600]
obs=green[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)
b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);boundary=np.argmax(~sky,axis=0);horizon=float(np.median(boundary));pitch_guess=-np.arctan((horizon-180)/(180/np.tan(np.deg2rad(35))))
def camera(v):
 ya,pi,playerh=v
 return [ya,pi,playerh+.3*np.sin(pi),px-.3*np.sin(ya)*np.cos(pi),pz+.3*np.cos(ya)*np.cos(pi)]
def loss(v):
 a=predict(camera(v),70,X.ravel(),Y.ravel());a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
starts=[]
for ya in np.deg2rad(np.arange(315,322.01,.25)):
 for pi in np.arange(pitch_guess-.025,pitch_guess+.0251,.0025):
  for playerh in np.arange(.3,1.5001,.04):
   v=np.array([ya,pi,playerh]);starts.append((loss(v),v))
starts.sort(key=lambda r:r[0]);out=[]
for _,v in starts[:30]:
 sol=minimize(loss,v,method='Nelder-Mead',options={'maxiter':1800,'xatol':1e-9},bounds=[(np.deg2rad(313),np.deg2rad(324)),(pitch_guess-.035,pitch_guess+.035),(.2,2.)])
 pred=predict(camera(sol.x),70,XT.ravel(),YT.ravel());out.append(dict(parameters=sol.x.tolist(),yaw_pitch_deg=np.rad2deg(sol.x[:2]).tolist(),player_height=float(sol.x[2]),train=float(-sol.fun),heldout=float(np.corrcoef(pred,green[YT,XT].ravel())[0,1])))
out.sort(key=lambda r:-r['train'])
result=dict(status='Diagnostic inherited fixed opening phase; no candidate inputs and no reset/physics assumption.',phase=[px,pz],horizon_row=horizon,horizon_pitch_deg=float(np.rad2deg(pitch_guess)),training_roi=[380,335,620,347],heldout_roi=[440,350,600,359],fits=out)
(P/'opening-free-height.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
