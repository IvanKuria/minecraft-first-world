from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;R1=P.parent/'first-reset';records=json.loads((R1/'skyline-rays.json').read_text())['records'][1:7];names=[str(r['frame'])for r in records];times=np.array([r['pts']for r in records]);ang=np.deg2rad([next(a['yaw_pitch_degrees'] for a in r['orientation_and_camera_rays'] if a['fov_degrees']==70)for r in records]);xx=np.arange(0,640,4);obs=[]
for name in names:
 im=cv2.imread(str(R1/f'frame-{name}.png'));b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);obs.append(np.argmax(~sky,axis=0)[xx])
cases=[json.loads(l)for l in (P.parent.parent/'runtime/respawn-motion/cases.jsonl').read_text().splitlines()];cases=[a for a in cases if a['scenario']=='grounded_still'and a['reset_held_ticks']==1 and a['tick']>0];ys=np.array([a['y']for a in cases]);ticks=np.arange(1,61)
def predict(v,i):
 x,z,tR,fov=v[:4];yaw,pitch=v[4:].reshape(-1,2)[i];s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]]);y=np.interp((times[i]-tR)*60,ticks,ys);C=np.array([x,y,z])-.3*R[2];f=180/np.tan(np.deg2rad(fov/2));q=R@(np.array([256,64,0])-C);xc,yc=f*q[:2]/q[2]+[320,180];vpX=f*R[:2,0]/R[2,0]+[320,180];vpZ=f*R[:2,2]/R[2,2]+[320,180];ml=(vpX[1]-yc)/(vpX[0]-xc);mr=(vpZ[1]-yc)/(vpZ[0]-xc);return yc+np.where(xx<xc,ml,mr)*(xx-xc)
def residual(v):return np.concatenate([predict(v,i)-obs[i]for i in range(len(names))])
v0=np.r_[195,210,85.42,69,ang.ravel()];lo=np.r_[100,100,85.398,68,(ang-.04).ravel()];hi=np.r_[245,255,85.46,70.5,(ang+.04).ravel()];sol=least_squares(residual,v0,bounds=(lo,hi),loss='soft_l1',f_scale=1,max_nfev=500);v=sol.x;print(v[:4],np.sqrt(np.mean(sol.fun**2)));out={'status':'conditional grounded-still, outer skyline on Y64, world256x256, later camera projection; not RNG observation','parameters':v.tolist(),'global_player_xz':v[:2].tolist(),'reset_time':float(v[2]),'fov':float(v[3]),'rms_px':float(np.sqrt(np.mean(sol.fun**2))),'perframe_rms':[float(np.sqrt(np.mean((predict(v,i)-obs[i])**2)))for i in range(len(names))],'angles_degrees':np.rad2deg(v[4:].reshape(-1,2)).tolist()}
# FOV/time grid profiles reveal sensitivity, not statistical confidence.
profiles=[]
for fov in [68,69,70,70.5]:
 for tr in [85.4,85.425,85.45]:
  def unwrap(a):return np.r_[a[:2],tr,fov,a[2:]]
  ss=least_squares(lambda a:residual(unwrap(a)),np.r_[v[:2],v[4:]],loss='soft_l1',f_scale=1,max_nfev=300);vv=unwrap(ss.x);profiles.append({'fov':fov,'reset_time':tr,'x':float(vv[0]),'z':float(vv[1]),'rms':float(np.sqrt(np.mean(ss.fun**2)))})
out['profiles']=profiles;(P/'joint-first-skyline-motion.json').write_text(json.dumps(out,indent=2));print(profiles)
for i,name in enumerate(names):
 im=cv2.imread(str(R1/f'frame-{name}.png'));cv2.polylines(im,[np.array(list(zip(xx,predict(v,i))),np.int32)],False,(0,0,255),1);cv2.imwrite(str(P/f'joint-first-skyline-{name}.png'),im)
