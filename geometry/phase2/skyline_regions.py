"""Export observed skyline + conditional global pose curves; no seed coordinates claimed."""
from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent
names=['85.600','152.744','152.796','152.847','152.899','152.951'];out=[]
for name in names:
 im=cv2.imread(str(P/f'f-{name}.png'));b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);ys=np.argmax(~sky,axis=0);xs=np.arange(640)
 def pred(v):xc,yc,ml,mr=v;return yc+np.where(xs<xc,ml,mr)*(xs-xc)
 sol=least_squares(lambda v:pred(v)-ys,[240,float(ys.min()),-.05,.2],loss='soft_l1',f_scale=1.5,bounds=([30,0,-2,0],[600,300,0,2]));xc,yc,ml,mr=map(float,sol.x)
 def geom(fov,delta=(0,0,0)):
  x,y,left,right=xc,yc,ml,mr; y+=delta[0];left+=delta[1]/240;right+=delta[2]/400;f=180/np.tan(np.deg2rad(fov/2));q=np.poly1d([1,-y]);v1=x+q/left-320;v2=x+q/right-320;hy=min(np.roots(v1*v2+np.poly1d([1,-180])**2+f*f));vx=x+(hy-y)/left;pitch=np.arctan((180-hy)/f);yaw=np.arctan(f/((vx-320)*np.cos(pitch)));s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]]);ray=R.T@np.array([(x-320)/f,(y-180)/f,1]);return ray/ray[1],R[2],np.rad2deg([yaw,pitch])
 off,forward,angles=geom(70);curves=[]
 for h in [7,8,9,10,10.5,10.75,11,12]:
  cam=np.array([256.,64.,0])+h*off;player=cam+.3*forward;curves.append({'camera_height_above_boundary':h,'camera':cam.tolist(),'interpolated_player':player.tolist()})
 # This is a sensitivity envelope, not probabilistic confidence interval.
 envelope=[]
 for fov in [68,69,70,70.5]:
  for dc in [-1,0,1]:
   for dl in [-1,0,1]:
    for dr in [-1,0,1]:
     a,fw,ang=geom(fov,(dc,dl,dr));envelope.append({'vertical_fov':fov,'edge_perturbation_px':[dc,dl,dr],'offset_per_height':a.tolist(),'forward':fw.tolist()})
 rec={'file':f'f-{name}.png','requested_time':float(name),'actual_time':85.605 if name=='85.600' else float(name),'corner_px':[float(xc),float(yc)],'edge_slopes':[float(ml),float(mr)],'fit_rms_px':float(np.sqrt(np.mean(sol.fun**2))),'vFOV70_yaw_pitch':angles.tolist(),'camera_offset_per_height_above_boundary':off.tolist(),'coordinate_convention':'visible far corner=(256,Yboundary,0); four rotations remain until texture UV axes established','conditional_curves_Yboundary64':curves,'sensitivity_samples':envelope};out.append(rec)
 cv2.polylines(im,[np.array(list(zip(xs,pred(sol.x))),np.int32)],False,(0,0,255),1);cv2.imwrite(str(P/f'skyline-{name}.png'),im);print(name,angles,off,curves[5])
(P/'skyline-regions.json').write_text(json.dumps(out,indent=2))
