from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent;tile=cv2.imread(str(P.parents[1]/'geometry/phase2/terrain.png'))[:16,:16,1].astype(float)
Y,X=np.mgrid[220:330:5,380:620:5];YT,XT=np.mgrid[335:355,400:600]
def lookup(wx,wz,nearest=False):
 if nearest:return tile[np.floor((wz%1)*15.984).astype(int)%16,np.floor((wx%1)*15.984).astype(int)%16]
 u=((wx%1)*15.984-.5)%16;v=((wz%1)*15.984-.5)%16;ui=np.floor(u).astype(int);vi=np.floor(v).astype(int);a=u-ui;b=v-vi
 return tile[vi,ui]*(1-a)*(1-b)+tile[vi,(ui+1)%16]*a*(1-b)+tile[(vi+1)%16,ui]*(1-a)*b+tile[(vi+1)%16,(ui+1)%16]*a*b

def predict(v,fov,x,y,nearest=False):
 f=180/np.tan(np.deg2rad(fov/2));rx=(x+.5-320)/f;ry=(y+.5-180)/f;yaw,pitch,h,px,pz=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
 return lookup(px-h*dx/dy,pz-h*dz/dy,nearest)
def player(v):
 ya,pi,h,cx,cz=v
 return [(cx+.3*np.sin(ya)*np.cos(pi))%1,(cz-.3*np.cos(ya)*np.cos(pi))%1]
