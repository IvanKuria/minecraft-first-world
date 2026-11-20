from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent
exec((P/'grass_height_sequence.py').read_text().split('out=[]')[0])
orig=tile.copy(); name='152.951';frame=cv2.imread(str(P/f'f-{name}.png'));im=frame[:,:,1].astype(float);ob=im[Y,X].ravel();ob-=ob.mean();ob/=np.linalg.norm(ob)
rec=[a for a in json.loads((P/'grass-height-sequence.json').read_text()) if a['t_requested']==float(name)][0];yaw,pitch=np.deg2rad(rec['skyline_yaw_pitch']);s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
out=[]
for seed in [-1,0,1,2,3,4,5]:
 if seed==-1:tile=orig.copy()
 else:
  rng=np.random.default_rng(seed);noise=rng.normal(size=(16,16));nfft=np.fft.rfft2(noise);tile=np.fft.irfft2(np.abs(np.fft.rfft2(orig-orig.mean()))*nfft/np.maximum(np.abs(nfft),1e-12),s=orig.shape).real+orig.mean()
 best=[]
 for h in np.arange(4,20.001,.1):
  u=((phase[:,0,None]-h*dx/dy)*16)%16;v=((phase[:,1,None]-h*dz/dy)*16)%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
  val=tile[v0,u0]*(1-a)*(1-b)+tile[v0,(u0+1)%16]*a*(1-b)+tile[(v0+1)%16,u0]*(1-a)*b+tile[(v0+1)%16,(u0+1)%16]*a*b;val-=val.mean(axis=1)[:,None];sc=val@ob/(np.linalg.norm(val,axis=1)+1e-9);i=np.argmax(sc);best.append((float(sc[i]),h,*phase[i]))
 best.sort(reverse=True)
 def loss(v):a=tex(v,rx,ry);a-=a.mean();return -np.dot(a,ob)/(np.linalg.norm(a)+1e-9)
 fits=[minimize(loss,[yaw,pitch,h,px,pz],method='Nelder-Mead',options={'maxiter':1600})for sc,h,px,pz in best[:6]];fits.sort(key=lambda a:a.fun);sol=fits[0];corr=np.corrcoef(tex(sol.x,rxt,ryt),im[YT,XT].ravel())[0,1]
 ans={'seed':seed,'kind':'authentic' if seed<0 else 'phase randomized; same tile power spectrum','height':float(sol.x[2]),'parameters':sol.x.tolist(),'train':float(-sol.fun),'heldout':float(corr)};out.append(ans);print(ans,flush=True)
(P/'grass-null-controls.json').write_text(json.dumps(out,indent=2))
