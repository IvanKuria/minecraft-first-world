from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent;tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float);im=cv2.imread(str(P.parent.parent/'runtime/render-texture-control/actual-game-render.png'))[:,:,1].astype(float);Y,X=np.mgrid[278:355,300:405];f=180/np.tan(np.deg2rad(35));yaw,pitch=np.deg2rad([31.65757,14.64737]);s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);H=10.3+.3*sp;cx=223.41498-.3*s*cp;cz=123.76827+.3*c*cp;rx=(X-320)/f;ry=(Y-180)/f;dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
out=[]
for shift in [0.,.5,-.5]:
 for rxshift,ryshift in [(0,0),(.5,.5),(-.5,-.5)]:
  rxx=(X+rxshift-320)/f;ryy=(Y+ryshift-180)/f;dxx=c*rxx-s*sp*ryy+s*cp;dyy=-cp*ryy-sp;dzz=s*rxx+c*sp*ryy-c*cp;u=np.floor(((cx-H*dxx/dyy)%1)*15.984+shift).astype(int)%16;v=np.floor(((cz-H*dzz/dyy)%1)*15.984+shift).astype(int)%16;pred=tile[v,u];corr=np.corrcoef(pred.ravel(),im[Y,X].ravel())[0,1];out.append({'texel_shift':shift,'pixel_xy_shift':[rxshift,ryshift],'corr':float(corr)});print(out[-1])
(P/'authentic-renderer-control.json').write_text(json.dumps({'truth_camera_height':H,'truth_camera_phase':[cx%1,cz%1],'results':out},indent=2))
