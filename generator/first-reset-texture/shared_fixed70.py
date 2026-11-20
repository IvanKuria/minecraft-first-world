from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from profile import predict,X,Y,XT,YT
P=Path(__file__).parent;names=['86.00','86.15','86.30'];images=[cv2.imread(str(P/f'f-{s}.png'))[:,:,1].astype(float)for s in names];obs=[]
for im in images:
 a=im[Y,X].ravel();a-=a.mean();a/=np.linalg.norm(a);obs.append(a)
def camera(v,j):
 ya,pi,h=v[2+3*j:5+3*j];return [ya,pi,h,v[0]-.3*np.sin(ya)*np.cos(pi),v[1]+.3*np.cos(ya)*np.cos(pi)]
def loss(v):
 score=0
 for j in range(3):
  a=predict(camera(v,j),70,X.ravel(),Y.ravel());a-=a.mean();score+=a@obs[j]/(np.linalg.norm(a)+1e-9)
 return -score/3
out=[]
for old in json.loads((P/'shared-phase.json').read_text()):
 start=np.array(old['all_parameters'][1:]);simplex=np.repeat(start[None,:],len(start)+1,axis=0)
 for i,d in enumerate([.01,.01]+[.001,.001,.01]*3):simplex[i+1,i]+=d
 sol=minimize(loss,start,method='Nelder-Mead',options={'maxiter':14000,'xatol':1e-7,'initial_simplex':simplex});rows=[]
 for j,n in enumerate(names):
  pr=predict(camera(sol.x,j),70,X.ravel(),Y.ravel());pr-=pr.mean();held=np.corrcoef(predict(camera(sol.x,j),70,XT.ravel(),YT.ravel()),images[j][YT,XT].ravel())[0,1];rows.append({'frame':n,'camera_parameters':camera(sol.x,j),'train':float(pr@obs[j]/np.linalg.norm(pr)),'heldout':float(held)})
 r={'fov_fixed':70,'shared_player_phase':(sol.x[:2]%1).tolist(),'mean_train':float(-sol.fun),'frames':rows,'all_parameters':sol.x.tolist()};out.append(r);print(r,flush=True)
(P/'shared-fixed70.json').write_text(json.dumps(out,indent=2))
