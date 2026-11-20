from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
from profile import predict,X,Y,XT,YT
P=Path(__file__).parent;names=['86.00','86.15','86.30'];im=[cv2.imread(str(P/f'f-{n}.png'))[:,:,1].astype(float)for n in names];seeds=json.loads((P/'shared-fixed70.json').read_text());seeds=sorted(seeds,key=lambda r:-r['mean_train'])[:2];out=[]
def camera(v,j):
 ya,pi,h=v[2+3*j:5+3*j];return [ya,pi,h,v[0]-.3*np.sin(ya)*np.cos(pi),v[1]+.3*np.cos(ya)*np.cos(pi)]
for label,mask in [('left_half',X.ravel()<500),('right_half',X.ravel()>=500)]:
 xx=X.ravel()[mask];yy=Y.ravel()[mask];obs=[]
 for image in im:
  a=image[yy,xx];a=a-a.mean();obs.append(a/np.linalg.norm(a))
 def loss(v):
  total=0
  for j in range(3):
   a=predict(camera(v,j),70,xx,yy);a-=a.mean();total+=a@obs[j]/(np.linalg.norm(a)+1e-9)
  return -total/3
 for seed in seeds:
  start=np.array(seed['all_parameters']);simplex=np.repeat(start[None,:],len(start)+1,axis=0)
  for i,d in enumerate([.01,.01]+[.001,.001,.01]*3):simplex[i+1,i]+=d
  fit=minimize(loss,start,method='Nelder-Mead',options={'maxiter':12000,'initial_simplex':simplex,'xatol':1e-7});held=[]
  for j in range(3):held.append(float(np.corrcoef(predict(camera(fit.x,j),70,XT.ravel(),YT.ravel()),im[j][YT,XT].ravel())[0,1]))
  row={'training_subset':label,'fixed_fov':70,'player_phase':(fit.x[:2]%1).tolist(),'mean_train':float(-fit.fun),'bottom_heldout_correlations':held,'initial_phase':seed['shared_player_phase'],'parameters':fit.x.tolist()};out.append(row);print(row,flush=True)
(P/'spatial-sensitivity.json').write_text(json.dumps(out,indent=2))
