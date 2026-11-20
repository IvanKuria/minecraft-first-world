"""Conditional joint fit; recorded code trajectory is a candidate prior, not original code."""
from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent;tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float)
records=json.loads((P/'grass-height-sequence.json').read_text());times=np.array([r['t_requested']for r in records]);angles=np.deg2rad([r['skyline_yaw_pitch']for r in records]);train=[0,2,4];test=[1,3];ys,xs=np.mgrid[278:355:2,300:405:2];yt,xt=np.mgrid[312:350:2,220:295:2];ims=[cv2.imread(str(P/r['file']))[:,:,1].astype(float)for r in records]
def normalized(a):return(a-a.mean(axis=-1,keepdims=True))/(np.linalg.norm(a-a.mean(axis=-1,keepdims=True),axis=-1,keepdims=True)+1e-9)
obs=[normalized(im[ys,xs].ravel())for im in ims]
cases=[json.loads(l)for l in (P.parent.parent/'runtime/respawn-motion/cases.jsonl').read_text().splitlines()];base={s:[a for a in cases if a['scenario']==s and a['reset_held_ticks']==1 and a['tick']>0]for s in ['grounded_still','grounded_jump','falling_terminal','grounded_forward']};fw=base['grounded_forward'];forward=np.array([[r['x']-99.969657898,r['z']-228.695343018]for r in fw]);ticks=np.arange(1,61)
def pix(v,i,scenario,xx=xs,yy=ys):
 px,pz,tr,fov,walk=v[:5];yaw,pitch=angles[i]+[np.interp(times[i],times[train],v[5:8]),np.interp(times[i],times[train],v[8:11])];tick=np.clip((times[i]-tr)*60,1,60);height=np.interp(tick,ticks,[a['y']for a in base[scenario]])+.3*np.sin(pitch)-64;fx=180/np.tan(np.deg2rad(fov/2));rx=(xx.ravel()+.5-320)/fx;ry=(yy.ravel()+.5-180)/fx;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch)
 camx=px+walk*np.interp(tick,ticks,forward[:,0])-.3*s*cp;camz=pz+walk*np.interp(tick,ticks,forward[:,1])+.3*c*cp;dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
 u=(((camx-height*dx/dy)%1)*15.984-.5)%16;w=(((camz-height*dz/dy)%1)*15.984-.5)%16;u0=np.floor(u).astype(int);w0=np.floor(w).astype(int);a=u-u0;b=w-w0
 return tile[w0,u0]*(1-a)*(1-b)+tile[w0,(u0+1)%16]*a*(1-b)+tile[(w0+1)%16,u0]*(1-a)*b+tile[(w0+1)%16,(u0+1)%16]*a*b
bounds=[(-2,3),(-2,3),(152.63,152.71),(68,70.5),(0,1.5)]+[(-.012,.012)]*6
out=[]
for scenario in ['grounded_still']:
 def loss(v):return -np.mean([np.dot(normalized(pix(v,i,scenario)),obs[i])for i in train])+0.015*np.mean((np.array(v[5:])/.012)**2)
 coarse=[]
 for tr in [152.65,152.685]:
  for fov in [68.5,70.]:
   for walk in [0.,1.]:
    for px in np.arange(0,1,1/16):
     for pz in np.arange(0,1,1/16):
      v=[px,pz,tr,fov,walk]+[0]*6;coarse.append((loss(v),v))
 coarse.sort(key=lambda a:a[0]);fits=[]
 for score,v in coarse[:8]:
  sol=minimize(loss,v,method='L-BFGS-B',bounds=bounds,options={'maxiter':250,'ftol':1e-10,'maxls':35});fits.append(sol)
 fits.sort(key=lambda a:a.fun);v=fits[0].x;rec={'scenario':scenario,'assumed_surface_Y':64,'parameters':v.tolist(),'parameter_order':['spawn_phase_x','spawn_phase_z','first_reset_time','vertical_fov_deg','forward_case_multiplier','yaw_delta_frame0','yaw_delta_frame2','yaw_delta_frame4','pitch_delta_frame0','pitch_delta_frame2','pitch_delta_frame4'],'train_objective':float(-fits[0].fun),'frames':[],'alternatives':[{'score':float(-s.fun),'parameters':s.x.tolist()}for s in fits]}
 for i,t in enumerate(times):
  rec['frames'].append({'t':float(t),'role':'training' if i in train else 'HELD OUT frame; orientation correction interpolated from training frames','same_patch_corr':float(np.dot(normalized(pix(v,i,scenario)),obs[i])),'heldout_patch_corr':float(np.corrcoef(pix(v,i,scenario,xt,yt),ims[i][yt,xt].ravel())[0,1])})
 out.append(rec);print(json.dumps(rec),flush=True);(P/'joint-grass-motion-centered.json').write_text(json.dumps(out,indent=2))
