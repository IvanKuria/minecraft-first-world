"""Clear-frame shared player-phase calibration, independent height per frame; no gravity prior."""
from pathlib import Path
import numpy as np,json,cv2
from scipy.optimize import least_squares
P=Path(__file__).parent;tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float)
settings=[('153.150','grass-153.150-fit.json',0,(30,220,220,331),(30,335,220,355)),('153.300','late-grass-fit.json',1,(20,205,210,331),(20,335,200,355)),('153.500','grass-153.500-fit.json',0,(20,205,210,331),(20,335,200,355))];data=[]
for name,file,k,box,test in settings:
 v=np.array(json.loads((P/file).read_text())[k]['parameters']);im=cv2.imread(str(P/f'f-{name}.png'))[:,:,1].astype(float);x0,y0,x1,y1=box;Y,X=np.mgrid[y0:y1:3,x0:x1:3];x0,y0,x1,y1=test;YT,XT=np.mgrid[y0:y1,x0:x1];data.append({'name':name,'v':v,'im':im,'X':X,'Y':Y,'XT':XT,'YT':YT})
def normalize(a):return (a-a.mean())/(a.std()+1e-9)
def prediction(v,i,held=False):
 px,pz,fov=v[:3];yaw,pitch,h=v[3+3*i:6+3*i];d=data[i];X,Y=(d['XT'],d['YT'])if held else(d['X'],d['Y']);f=180/np.tan(np.deg2rad(fov/2));x=(X.ravel()+.5-320)/f;y=(Y.ravel()+.5-180)/f;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*x-s*sp*y+s*cp;dy=-cp*y-sp;dz=s*x+c*sp*y-c*cp;cx=px-.3*s*cp;cz=pz+.3*c*cp;u=(((cx-h*dx/dy)%1)*15.984-.5)%16;w=(((cz-h*dz/dy)%1)*15.984-.5)%16;u0=np.floor(u).astype(int);w0=np.floor(w).astype(int);a=u-u0;b=w-w0;return tile[w0,u0]*(1-a)*(1-b)+tile[w0,(u0+1)%16]*a*(1-b)+tile[(w0+1)%16,u0]*(1-a)*b+tile[(w0+1)%16,(u0+1)%16]*a*b
ob=[normalize(d['im'][d['Y'],d['X']].ravel())for d in data]
def residual(v):return np.concatenate([(normalize(prediction(v,i))-ob[i])/np.sqrt(len(ob[i]))for i in range(3)])
v0=np.r_[.8,.8,70,np.concatenate([d['v'][:3]for d in data])];low=np.r_[0,0,68,np.concatenate([d['v'][:3]-[.05,.05,1]for d in data])];high=np.r_[2,2,72,np.concatenate([d['v'][:3]+[.05,.05,1]for d in data])];sol=least_squares(residual,v0,bounds=(low,high),x_scale=np.r_[.05,.05,1,[.01,.01,.1]*3],max_nfev=700,ftol=1e-10,xtol=1e-10);v=sol.x;out={'status':'conditional constant playerxz across3frames, sharedFOV, independentmeasuredheights; no resetphysics','parameters':v.tolist(),'shared_player_phase':(v[:2]%1).tolist(),'shared_fov':float(v[2]),'frames':[]}
for i,d in enumerate(data):
 out['frames'].append({'frame':d['name'],'yaw_pitch_deg':np.rad2deg(v[3+3*i:5+3*i]).tolist(),'height_above_plane':float(v[5+3*i]),'train':float(np.corrcoef(prediction(v,i),ob[i])[0,1]),'heldout':float(np.corrcoef(prediction(v,i,True),d['im'][d['YT'],d['XT']].ravel())[0,1])})
print(json.dumps(out),flush=True);(P/'joint-clear-grass.json').write_text(json.dumps(out,indent=2))
