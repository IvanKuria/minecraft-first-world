from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from model import predict
P=Path(__file__).parent;names=['2.00','2.10','2.30'];obs=[];coords=[];held=[];images=[];angles=[]
for n in names:
 im=cv2.imread(str(P/f'f-{n}.png'))[:,:,1].astype(float);images.append(im)
 if n=='2.00':Y,X=np.mgrid[240:300:3,360:620:3];YT,XT=np.mgrid[310:350,480:620]
 else:Y,X=np.mgrid[200:290:4,360:620:4];YT,XT=np.mgrid[300:340,480:620]
 coords.append((X.ravel(),Y.ravel()));held.append((XT.ravel(),YT.ravel()));a=im[Y,X].ravel();a-=a.mean();obs.append(a/np.linalg.norm(a));best=max(json.loads((P/f'fit-{n}.json').read_text()),key=lambda a:a['train']);angles+=best['parameters'][:2]
def camera(v,j):
 ya,pi=v[3+2*j:5+2*j];px,pz=v[1:3];return [ya,pi,.62+.3*np.sin(pi),px-.3*np.sin(ya)*np.cos(pi),pz+.3*np.cos(ya)*np.cos(pi)]
def loss(v):
 total=0
 if not 65<v[0]<75:return 1
 for j in range(3):
  a=predict(camera(v,j),v[0],*coords[j]);a-=a.mean();total+=a@obs[j]/(np.linalg.norm(a)+1e-9)
 return -total/3
out=[]
for mode in ['fixed70','free_fov']:
 for dx in [0,-.006,.006]:
  full=np.r_[70,.095+dx,.805,angles];start=full[1:]if mode=='fixed70'else full
  def expand(v):return np.r_[70,v]if mode=='fixed70'else v
  simplex=np.repeat(start[None,:],len(start)+1,axis=0);steps=([.10]if mode=='free_fov'else[])+[.002,.002]+[.001,.001]*3
  for i,d in enumerate(steps):simplex[i+1,i]+=d
  fit=minimize(lambda v:loss(expand(v)),start,method='Nelder-Mead',options={'maxiter':10000,'xatol':1e-8,'initial_simplex':simplex});v=expand(fit.x);rs=[]
  for j,n in enumerate(names):
   a=predict(camera(v,j),v[0],*coords[j]);a-=a.mean();xt,yt=held[j];rs.append({'frame_name':n,'camera_parameters':camera(v,j),'train':float(a@obs[j]/np.linalg.norm(a)),'heldout':float(np.corrcoef(predict(camera(v,j),v[0],xt,yt),images[j][yt,xt])[0,1])})
  row={'mode':mode,'fov':float(v[0]),'player_phase':v[1:3].tolist(),'player_height_above_plane_fixed':.62,'mean_train':float(-fit.fun),'frames':rs,'parameters':v.tolist(),'converged':fit.success};out.append(row);print(row,flush=True)
(P/'joint-opening.json').write_text(json.dumps(out,indent=2))
