from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;f=180/np.tan(np.deg2rad(35));cx,cy=320,180
out={}
for n in [2951,2953,2955,2957,2959,2961]:
 im=cv2.imread(str(P/f'respawn-frame-{n}.png')); b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25)
 # First non-sky on every column; this is world's exterior skyline in these specific views.
 yy=np.argmax(~sky,axis=0);xx=np.arange(640)
 good=(yy>20)&(yy<320)
 xx=xx[good];yy=yy[good]
 def pred(v):
  xc,yc,ml,mr=v;return yc+np.where(xx<xc,ml,mr)*(xx-xc)
 fit=least_squares(lambda v:pred(v)-yy,[xx[np.argmin(yy)],min(yy),-.05,.15],loss='soft_l1',f_scale=1.5,bounds=([20,0,-2,0],[620,300,0,2]))
 xc,yc,ml,mr=map(float,fit.x)
 q=np.poly1d([1,-yc]);p1=xc+q/ml-cx;p2=xc+q/mr-cx
 roots=np.roots(p1*p2+np.poly1d([1,-cy])**2+f*f)
 horizon=float(min(roots));vx=xc+(horizon-yc)/ml;vz=xc+(horizon-yc)/mr
 pitch=np.arctan((cy-horizon)/f);yaw=np.arctan(f/((vx-cx)*np.cos(pitch)))
 s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]])
 ray=R.T@np.array([(xc-cx)/f,(yc-cy)/f,1.]);off=ray/ray[1] # Camera-to-corner displacement divided by camera height difference is -ray/rayy.
 # C = [256,planeY,0] + H * [offx,1,offz].
 conditional=np.array([256.,64.,0])+10.72*off
 rec={'frame':n,'corner_px':[xc,yc],'slopes':[ml,mr],'skyline_fit_rms_px':float(np.sqrt(np.mean(fit.fun**2))),'horizon_y':horizon,'yaw_pitch_deg':np.rad2deg([yaw,pitch]).tolist(),'camera_offset_per_unit_height_above_top':off.tolist(),'conditional_camera_if_corner256_64_0_and_height74_72':conditional.tolist()};rec['status']='provisional skyline-only conditional estimate; not an RNG observation' if n<=2957 else 'REJECTED: chosen skyline is no longer a usable two-edge outer corner'; rec['height_assumption_warning']='Uses later-build74.72height and full-height64edge; not established for missingbuild or interpolation/resettick'; out[n]=rec
 print(n,'corner',np.round([xc,yc],2),'yaw/pitch',np.round(np.rad2deg([yaw,pitch]),2),'perheight',np.round(off,3),'conditionalC',np.round(conditional,2))
 for x,y in zip(xx,pred(fit.x)):cv2.circle(im,(int(x),int(y)),0,(0,0,255),1)
 cv2.imwrite(str(P/f'respawn-skyline-{n}.png'),im)
(P/'respawn-corner.json').write_text(json.dumps(out,indent=2))
