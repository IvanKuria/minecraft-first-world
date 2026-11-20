from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;Q=P.parent/'phase2';ns={'__file__':str(Q/'joint_four_clear_grass.py')};exec((Q/'joint_four_clear_grass.py').read_text().split('v0=np.r_')[0],ns);base=np.array(json.loads((Q/'joint-four-clear-grass.json').read_text())['parameters'])
def unpack(a):return np.r_[a[:2],70.,a[2:]]
fit=least_squares(lambda a:ns['residual'](unpack(a)),np.r_[base[:2],base[3:]],x_scale=np.r_[.05,.05,[.01,.01,.1]*4],max_nfev=700);v=unpack(fit.x);im=cv2.imread(str(Q/'f-153.050.png'));b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);ys=np.argmax(~sky,axis=0)-.5;xs=np.arange(640)
def pred(a):xc,yc,ml,mr=a;return yc+np.where(xs<xc,ml,mr)*(xs-xc)
sol=least_squares(lambda a:pred(a)-ys,[35,30,-.1,.2],bounds=([1,0,-3,0],[180,180,0,3]),loss='soft_l1',f_scale=.7);xc,yc,ml,mr=sol.x;yaw,pitch,h=v[3:6];s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]]);f=180/np.tan(np.deg2rad(35));ray=R.T@np.array([(xc+.5-320)/f,(yc+.5-180)/f,1]);off=ray/ray[1];rows=[]
for k in range(-5,6):
 C=np.array([256.,0,0])+(h-k)*off;player=C+.3*R[2];rows.append({'outer_minus_nearplane':k,'player_xz':player[[0,2]].tolist(),'phase_error':(((player[[0,2]]-v[:2]+.5)%1)-.5).tolist()})
out={'shared_phase':v[:2].tolist(),'all_pose_parameters_fixed70':v.tolist(),'source_pts':153.054,'corner_pixel':sol.x[:2].tolist(),'slopes':sol.x[2:].tolist(),'fit_rms':float(np.sqrt(np.mean(sol.fun**2))),'texture_pose_yaw_pitch_height':v[3:6].tolist(),'ray_height_coefficients':off.tolist(),'positions_by_integer_plane_offset':rows};(P/'clear-corner-measurement.json').write_text(json.dumps(out,indent=2));print(out);cv2.polylines(im,[np.int32(np.c_[xs,pred(sol.x)])],False,(0,0,255),1);cv2.circle(im,(int(xc),int(yc)),5,(255,0,255),1);cv2.imwrite(str(P/'clear-corner-153.054.png'),im)
