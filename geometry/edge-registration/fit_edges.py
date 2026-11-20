from pathlib import Path
import numpy as np,cv2,json
from scipy.optimize import least_squares
P=Path(__file__).parent;Q=P.parent/'phase2';ns={'__file__':str(Q/'joint_clear_grass.py')};exec((Q/'joint_clear_grass.py').read_text().split('v0=np.r_')[0],ns);base=np.array(json.loads((Q/'joint-clear-grass.json').read_text())['parameters'])
def unpack(a):return np.r_[a[:2],70.,a[2:]]
sol=least_squares(lambda a:ns['residual'](unpack(a)),np.r_[base[:2],base[3:]],x_scale=np.r_[.05,.05,[.01,.01,.1]*3],max_nfev=700);v=unpack(sol.x);out={'fixed_fov':70,'player_phase':v[:2].tolist(),'frames':[]}
for i,d in enumerate(ns['data']):
 name=d['name'];yaw,pitch,h=v[3+3*i:6+3*i];im=cv2.imread(str(Q/f'f-{name}.png'));b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);ys=np.argmax(~sky,axis=0);xs=np.arange(640);mask=(xs>=20)&(xs<620)&(ys>=3)&(ys<120);x=xs[mask];y=ys[mask]-.5
 lf=least_squares(lambda a:a[0]*x+a[1]-y,[0,40],loss='soft_l1',f_scale=.7);m,inter=lf.x;f=180/np.tan(np.deg2rad(35));rx=(x+.5-320)/f;ry=(m*x+inter+.5-180)/f;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp;ratio=dx/dy;forward=np.array([s*cp,-sp,-c*cp]);coef=float(np.median(ratio));klist=[]
 for k in range(-8,9):
  Cx=256+(h-k)*coef;Px=Cx+.3*forward[0];klist.append({'boundary_minus_nearplane_integer':k,'player_x':float(Px),'phase_residual':float((Px-v[0]+.5)%1-.5)})
 rec={'frame':name,'yaw_pitch_height':[float(yaw),float(pitch),float(h)],'skyline_slope_intercept':[float(m),float(inter)],'skyline_fit_rms':float(np.sqrt(np.mean(lf.fun**2))),'skyline_xrange':[int(x.min()),int(x.max())],'camera_dx_dy_ratio':coef,'ratio_range':np.percentile(ratio,[5,95]).tolist(),'forward':forward.tolist(),'conditional_integer_height_positions':klist};out['frames'].append(rec);print(name,'sky',lf.x,'ratio',coef,'H',h,'Px(k0)',klist[8]['player_x'],flush=True)
 cv2.polylines(im,[np.int32(np.c_[x,m*x+inter])],False,(0,0,255),1);cv2.imwrite(str(P/f'skyline-{name}.png'),im)
(P/'edge-measurements.json').write_text(json.dumps(out,indent=2))
