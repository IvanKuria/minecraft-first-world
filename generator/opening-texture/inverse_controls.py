from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from model import predict,player
P=Path(__file__).parent;B=P.parents[1]/'runtime';truth=json.loads((B/'render-opening-control-native/truth.json').read_text());native=cv2.imread(str(B/'render-opening-control-native/actual-game-render.png'))[:,:,1].astype(float);hd=cv2.imread(str(B/'render-opening-control-hd/actual-game-render.png'))[:,:,1].astype(float);cases={'native':native,'hd_area':cv2.resize(hd,(640,360),interpolation=cv2.INTER_AREA),'hd_lanczos':cv2.resize(hd,(640,360),interpolation=cv2.INTER_LANCZOS4)}
yaw,pitch=np.radians([truth['yaw_deg'],truth['pitch_deg']]);px,py,pz=truth['player'];H=py-truth['grass_plane_y']+.3*np.sin(pitch);Cx=px-.3*np.sin(yaw)*np.cos(pitch);Cz=pz+.3*np.cos(yaw)*np.cos(pitch);start=np.array([yaw,pitch,H,Cx%1,Cz%1]);Y,X=np.mgrid[200:290:4,360:620:4];YT,XT=np.mgrid[300:340,480:620];out=[]
for label,im in cases.items():
 obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)
 def loss(v):
  a=predict(v,70,X.ravel(),Y.ravel());a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
 for shift in [0,.005]:
  init=start+np.array([0,0,0,shift,-shift]);fit=minimize(loss,init,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-9});ph=np.array(player(fit.x));err=(ph-np.array([px%1,pz%1])+.5)%1-.5;held=np.corrcoef(predict(fit.x,70,XT.ravel(),YT.ravel()),im[YT,XT].ravel())[0,1]
  row={'image':label,'initial_phase_perturbation':shift,'true_player_phase':[px%1,pz%1],'inferred_player_phase':ph.tolist(),'signed_phase_error':err.tolist(),'train':float(-fit.fun),'heldout':float(held),'parameters':fit.x.tolist()};out.append(row);print(row,flush=True)
(P/'inverse-controls.json').write_text(json.dumps({'truth':truth,'tests':out,'limit':'Actual original renderer atoneknownsyntheticpose; not coverage of everyhistoricalcaptureartifact.'},indent=2))
