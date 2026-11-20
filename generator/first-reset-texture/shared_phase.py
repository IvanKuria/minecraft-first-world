"""Test stationary interpolated-player X/Z and shared FOV against separate frames."""
from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from profile import predict,player,X,Y,XT,YT
P=Path(__file__).parent;names=['86.00','86.15','86.30'];profile=json.loads((P/'fov-profile.json').read_text());images=[cv2.imread(str(P/f'f-{s}.png'))[:,:,1].astype(float)for s in names];obs=[]
for im in images:
 a=im[Y,X].ravel();a-=a.mean();a/=np.linalg.norm(a);obs.append(a)
def camera(v,j):
 yaw,pitch,h=v[3+3*j:6+3*j];return [yaw,pitch,h,v[1]-.3*np.sin(yaw)*np.cos(pitch),v[2]+.3*np.cos(yaw)*np.cos(pitch)]
def score(v,j,x=X.ravel(),y=Y.ravel()):return predict(camera(v,j),v[0],x,y)
def loss(v):
 if not 65<v[0]<75 or min(v[5::3])<1:return 2
 total=0
 for j in range(3):
  a=score(v,j);a-=a.mean();total+=a@obs[j]/(np.linalg.norm(a)+1e-9)
 return -total/3
results=[]
for fov in [69.,70.,71.]:
 rs=[next(r for r in profile if r['frame_name']==n and r['fov']==fov)for n in names];ph=np.array([r['player_phase']for r in rs]);ph[:,1][ph[:,1]<.4]+=1
 start=np.r_[fov,ph.mean(axis=0),*[r['parameters'][:3]for r in rs]]
 simplex=np.repeat(start[None,:],len(start)+1,axis=0);steps=[.10,.01,.01]+[.001,.001,.01]*3
 for i,d in enumerate(steps):simplex[i+1,i]+=d
 sol=minimize(loss,start,method='Nelder-Mead',options={'maxiter':22000,'xatol':1e-7,'initial_simplex':simplex})
 rows=[]
 for j,n in enumerate(names):
  pr=score(sol.x,j);pr-=pr.mean();held=np.corrcoef(score(sol.x,j,XT.ravel(),YT.ravel()),images[j][YT,XT].ravel())[0,1]
  rows.append({'frame':n,'camera_parameters':camera(sol.x,j),'train':float(pr@obs[j]/np.linalg.norm(pr)),'heldout':float(held)})
 rec={'start_fov':fov,'mean_train':float(-sol.fun),'fov':float(sol.x[0]),'shared_player_phase':(sol.x[1:3]%1).tolist(),'iterations':sol.nit,'converged':sol.success,'frames':rows,'all_parameters':sol.x.tolist()};results.append(rec);print(rec,flush=True);(P/'shared-phase.json').write_text(json.dumps(results,indent=2))
