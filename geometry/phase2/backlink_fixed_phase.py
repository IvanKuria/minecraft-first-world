from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent;common=json.loads((P/'joint-clear-grass.json').read_text());px,pz=common['shared_player_phase'];fov=common['shared_fov'];f=180/np.tan(np.deg2rad(fov/2));tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float);initial=json.loads((P/'grass-height-sequence.json').read_text());out=[]
for rec in initial:
 name=str(rec['t_requested']);im=cv2.imread(str(P/rec['file']))[:,:,1].astype(float);Y,X=np.mgrid[278:355:2,300:405:2];YT,XT=np.mgrid[312:350,220:295];rx=(X.ravel()+.5-320)/f;ry=(Y.ravel()+.5-180)/f;rxt=(XT.ravel()+.5-320)/f;ryt=(YT.ravel()+.5-180)/f;obs=im[Y,X].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)
 def values(v,x=rx,y=ry):
  yaw,pitch,h=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);cx=px-.3*s*cp;cz=pz+.3*c*cp;dx=c*x-s*sp*y+s*cp;dy=-cp*y-sp;dz=s*x+c*sp*y-c*cp;u=(((cx-h*dx/dy)%1)*15.984-.5)%16;w=(((cz-h*dz/dy)%1)*15.984-.5)%16;u0=np.floor(u).astype(int);w0=np.floor(w).astype(int);a=u-u0;b=w-w0;return tile[w0,u0]*(1-a)*(1-b)+tile[w0,(u0+1)%16]*a*(1-b)+tile[(w0+1)%16,u0]*(1-a)*b+tile[(w0+1)%16,(u0+1)%16]*a*b
 def loss(v):a=values(v);a-=a.mean();return -a@obs/(np.linalg.norm(a)+1e-9)
 a=np.deg2rad(rec['skyline_yaw_pitch']);starts=[np.r_[a,h]for h in np.arange(9,11.21,.1)];fits=[minimize(loss,s,method='Nelder-Mead',options={'maxiter':1500})for s in starts];fits.sort(key=lambda s:s.fun);v=fits[0].x;held=np.corrcoef(values(v,rxt,ryt),im[YT,XT].ravel())[0,1];result={'time':rec['t_requested'],'fixed_player_phase':[px,pz],'fixed_fov':fov,'yaw_pitch_height':v.tolist(),'train':float(-fits[0].fun),'heldout':float(held),'nextbest':[{'score':float(-s.fun),'height':float(s.x[2])}for s in fits[:4]]};out.append(result);print(result,flush=True)
(P/'backlink-fixed-phase.json').write_text(json.dumps(out,indent=2))
