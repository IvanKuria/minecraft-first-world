from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent;frame=cv2.imread(str(P/'f-152.899.png'));tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float);im=frame[:,:,1].astype(float);f=180/np.tan(np.deg2rad(35))
Y,X=np.mgrid[260:360,210:415];rx=(X.ravel()-320)/f;ry=(Y.ravel()-180)/f
train=(X>=300)&(X<405)&(Y>=278)&(Y<355);test=(X>=220)&(X<295)&(Y>=312)&(Y<350)
obs=im[Y,X];ob=obs[train];ob=ob-ob.mean();ob=ob/np.linalg.norm(ob)
v0=np.array(sorted(json.loads((P/'grass-orientations.json').read_text()),key=lambda x:-x['score'])[0]['params'])

def predict(v):
 yaw,pitch,h,px,pz=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);val=np.zeros(rx.shape)
 for ax,ay in [(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)]:
  x=rx+ax/f;y=ry+ay/f;dx=c*x-s*sp*y+s*cp;dy=-cp*y-sp;dz=s*x+c*sp*y-c*cp
  # True OpenGLnearest magnification, actual u1=15.984texturepixels.
  u=np.floor(((px-h*dx/dy)%1)*15.984).astype(int);z=np.floor(((pz-h*dz/dy)%1)*15.984).astype(int);val+=tile[z,u]/4
 return cv2.GaussianBlur(val.reshape(X.shape),(0,0),.45)

def fun(v):
 a=predict(v)[train];a=a-a.mean();return -np.dot(a,ob)/(np.linalg.norm(a)+1e-9)
# Bilineartexturephase uses texelcenters differently; accountfor ~halftexel.
v0[3:]+=.03125
fits=[]
for perturb in [0,-.02,.02]:
 v=v0.copy();v[2]+=perturb
 sol=minimize(fun,v,method='Nelder-Mead',options={'maxiter':2500,'xatol':1e-6});fits.append(sol)
sol=min(fits,key=lambda a:a.fun);pred=predict(sol.x);print('fit',sol.fun,sol.x);print('heldout',np.corrcoef(pred[test],obs[test])[0,1]);
out={'score':float(-sol.fun),'heldout_score':float(np.corrcoef(pred[test],obs[test])[0,1]),'parameters':sol.x.tolist(),'sampling':'4subpixelnearestplusgaussian.45','train_pixels':int(train.sum()),'test_pixels':int(test.sum())};(P/'grass-sampling-fit.json').write_text(json.dumps(out,indent=2));cv2.imwrite(str(P/'grass-sampling-comparison.png'),np.uint8(np.hstack([obs,pred])))
