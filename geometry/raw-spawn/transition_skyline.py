"""Unseen-transition world-boundary rays, without reset/candidate inputs."""
from pathlib import Path
import cv2,json,numpy as np
from scipy.optimize import least_squares
P=Path(__file__).parent;ROOT=P.parents[1];frames={}
cap=cv2.VideoCapture(str(ROOT/'footage-sources/wayback-fakeurl.mp4'));wanted={1651,1652,1653,2951,2952,2953}
for i in range(2954):
 ok,im=cap.read()
 if not ok:break
 if i in wanted:
  frames[i]=(cap.get(cv2.CAP_PROP_POS_MSEC)/1000,im);cv2.imwrite(str(P/f'original-{i}.png'),im)
cap.release();records=[]
for idx in [1652,1653,2952,2953]:
 pts,im=frames[idx];b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);ys=np.argmax(~sky,axis=0).astype(float);xs=np.arange(640,dtype=float)+.5
 def pred(v):xc,yc,ml,mr=v;return yc+np.where(xs<xc,ml,mr)*(xs-xc)
 fit=least_squares(lambda v:pred(v)-ys,[350,float(ys.min()),-.05,.05],loss='soft_l1',f_scale=.7,bounds=([20,0,-2,0],[620,330,0,2]));xc,yc,ml,mr=fit.x
 def geom(v):
  x,y,left,right=map(float,v);f=180/np.tan(np.deg2rad(35));q=np.poly1d([1,-y]);v1=x+q/left-320;v2=x+q/right-320;hy=min(np.roots(v1*v2+np.poly1d([1,-180])**2+f*f));vx=x+(hy-y)/left;pitch=np.arctan((180-hy)/f);yaw=np.arctan(f/((vx-320)*np.cos(pitch)));s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]]);ray=R.T@np.array([(x-320)/f,(y-180)/f,1]);return ray/ray[1],R[2],np.rad2deg([yaw,pitch])
 off,fw,angles=geom(fit.x);samples=[]
 for dc in [-1,0,1]:
  for dl in [-1,0,1]:
   for dr in [-1,0,1]:
    v=fit.x.copy();v[1]+=dc;v[2]+=dl/max(xc,20);v[3]+=dr/max(640-xc,20)
    if v[2]>=0 or v[3]<=0:continue
    a,w,ang=geom(v);samples.append(dict(edge_bias_px=[dc,dl,dr],camera_offset_per_height=a.tolist(),forward=w.tolist(),yaw_pitch_deg=ang.tolist()))
 records.append(dict(frame=idx,pts=pts,corner_pixel_center=[float(xc),float(yc)],slopes=[float(ml),float(mr)],rms_px=float(np.sqrt(np.mean(fit.fun**2))),yaw_pitch_deg=angles.tolist(),camera_offset_per_height=off.tolist(),forward=fw.tolist(),sensitivity=samples))
 cv2.polylines(im,[np.int32(np.c_[xs-.5,pred(fit.x)-.5])],False,(0,0,255),1);cv2.imwrite(str(P/f'skyline-{idx}.png'),im)
 print(idx,pts,angles,off,flush=True)
out=dict(status='Conditional visible-corner rays only. No height, physics, interpolation weight, RNG candidate, or original world is supplied.',projection='Original70deg FOV;640x360 centered square pixels; visible corner X256,Z0; pixel centers.',camera_equation='C=(256,Yedge,0)+H*camera_offset_per_height; effective interpolated player P=C+.3*forward.',sensitivity='Finite ±1px corner and endpoint-slope perturbations; not a probabilistic confidence region.',records=records)
(P/'transition-skyline.json').write_text(json.dumps(out,indent=2))
