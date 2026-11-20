from pathlib import Path
import cv2,numpy as np,json,time
from scipy.optimize import differential_evolution,minimize
P=Path(__file__).parent
frame=cv2.imread(str(P/'f-153.300.png'));tile=cv2.imread(str(P/'terrain.png'))[:16,:16,1].astype(float)
# Source depicts an uninterrupted level grass surface in this rectangle.
yy,xx=np.mgrid[200:331:3,20:211:3];f=180/np.tan(np.deg2rad(35));rx=(xx.ravel()-320)/f;ry=(yy.ravel()-180)/f
im=frame[:,:,1].astype(float);obs=im[yy,xx].ravel();obs=obs-obs.mean();obs=obs/(np.linalg.norm(obs)+1e-9)
# camera y height above visible plane; x/z only matter moduloone textureunit.
def values(v,x=rx,y=ry):
 yaw,pitch,h,phase_x,phase_z=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch)
 dx=c*x-s*sp*y+s*cp;dy=-cp*y-sp;dz=s*x+c*sp*y-c*cp
 wx=(phase_x-h*dx/dy)*16;wz=(phase_z-h*dz/dy)*16
 # Bilinear sampling periodically tiled actual16x16 texture.
 u=wx%16;v=wz%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
 val=tile[v0,u0]*(1-a)*(1-b)+tile[v0,(u0+1)%16]*a*(1-b)+tile[(v0+1)%16,u0]*(1-a)*b+tile[(v0+1)%16,(u0+1)%16]*a*b
 return val
calls=0;best=-1
start=time.time()
def loss(v):
 global calls,best
 a=values(v);a-=a.mean();corr=np.dot(a,obs)/(np.linalg.norm(a)+1e-9);calls+=1
 if corr>best:
  best=corr
  if corr>.18:print('best',calls,round(corr,4),np.round(v,4),flush=True)
 return -corr
bounds=[(-np.pi,np.pi),(.3,1.3),(3,16),(0,1),(0,1)]
sol=differential_evolution(loss,bounds,maxiter=400,popsize=20,tol=.0001,seed=16,polish=True,workers=1,updating='immediate')
print('FINISH',time.time()-start,sol.fun,sol.x,flush=True)
out={'source':'authenticrd132211terrain16x16grass','frame':'f-153.300.png','training_rect':[20,200,211,331],'score_correlation':float(-sol.fun),'parameters':sol.x.tolist(),'parameter_order':['yaw_rad','pitch_rad','height_above_plane','phase_x','phase_z'],'calls':calls}
(P/'grass-fit.json').write_text(json.dumps(out,indent=2))
Y,X=np.mgrid[:360,:640];pred=values(sol.x,(X.ravel()-320)/f,(Y.ravel()-180)/f).reshape(360,640)
cv2.imwrite(str(P/'grass-prediction.png'),np.uint8(np.clip(pred,0,255)))
# Check separate visible flatpatch, no refitting.
for box in [(20,200,211,331),(20,160,190,195),(20,335,200,355)]:
 x0,y0,x1,y1=box;a=pred[y0:y1,x0:x1].ravel();b=im[y0:y1,x0:x1].ravel();print('corr',box,np.corrcoef(a,b)[0,1])
