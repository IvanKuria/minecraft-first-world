"""Register first-reset exterior skyline, preserving height/FOV ambiguity."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
BASE=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
pts=json.loads((BASE/'timing/original2011-cadence.json').read_text())['pts_seconds']
selected=[i for i,t in enumerate(pts) if 85.45<=t<=86.0]
cap=cv2.VideoCapture(str(BASE/'footage-sources/wayback-fakeurl.mp4'));frames={}
for i in range(max(selected)+1):
 ok,im=cap.read()
 if not ok:break
 if i in selected:frames[i]=im
cap.release();rows=[]
for i,im in frames.items():
 b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25)
 ys=np.argmax(~sky,axis=0);xs=np.arange(640);valid=(ys>20)&(ys<300)
 x,y=xs[valid],ys[valid]
 def predict(v):
  xc,yc,ml,mr=v
  return yc+np.where(x<xc,ml,mr)*(x-xc)
 fitted=least_squares(lambda v:predict(v)-y,[x[np.argmin(y)],min(y),-.05,.15],
   loss='soft_l1',f_scale=1.5,bounds=([10,0,-2,0],[630,300,0,2]))
 xc,yc,ml,mr=map(float,fitted.x);rms=float(np.sqrt(np.mean(fitted.fun**2)))
 record={'frame':i,'pts':pts[i],'corner_px':[xc,yc],'slopes':[ml,mr],
         'skyline_rms_px':rms,'status':'conditional exterior-corner model, not a recovered spawn coordinate',
         'orientation_and_camera_rays':[]}
 for fov in [68,69,70]:
  f=180/np.tan(np.deg2rad(fov/2));q=np.poly1d([1,-yc])
  roots=np.roots((xc+q/ml-320)*(xc+q/mr-320)+np.poly1d([1,-180])**2+f*f)
  if np.any(np.iscomplex(roots)):continue
  horizon=float(min(roots));vx=xc+(horizon-yc)/ml
  pitch=np.arctan((180-horizon)/f);yaw=np.arctan(f/((vx-320)*np.cos(pitch)))
  s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch)
  R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]])
  ray=R.T@np.array([(xc-320)/f,(yc-180)/f,1.]);off=ray/ray[1]
  record['orientation_and_camera_rays'].append({'fov_degrees':fov,
       'yaw_pitch_degrees':np.rad2deg([yaw,pitch]).tolist(),
       'camera_offset_per_height_above_corner':off.tolist(),
       'conditional_camera_at_height10_above_corner256_64_0':(np.array([256,64,0])+10*off).tolist()})
 overlay=im.copy()
 for xx,yy in zip(x,predict(fitted.x)):cv2.circle(overlay,(int(xx),int(yy)),0,(0,0,255),1)
 cv2.imwrite(str(OUT/f'skyline-{i}.png'),overlay)
 cv2.imwrite(str(OUT/f'frame-{i}.png'),im)
 rows.append(record);print(json.dumps(record),flush=True)
(OUT/'skyline-rays.json').write_text(json.dumps({'coordinate_convention':'observed exterior corner placed at(256,64,0); rotations remain unregistered',
 'limits':['Corner height64 is conditional, not inferred by skyline',
           'Absolute camera position equals corner+H*ray; raw player RNG coordinates require setback and motion correction',
           'Initial transition may interpolate old and new positions; later frames can include movement',
           'No confidence interval or validated plane identity follows from least-squares residual alone'],
 'records':rows},indent=2)+'\n')
