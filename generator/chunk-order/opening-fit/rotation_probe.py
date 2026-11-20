import cv2,numpy as np,json
from pathlib import Path
from scipy.optimize import differential_evolution, minimize
P=Path(__file__).parent;f=180/np.tan(np.deg2rad(35));K=np.array([[f,0,320],[0,f,180],[0,0,1.]])
ref=cv2.imread(str(P/'frames/frame-15.png')).astype(float)/255
# Blur suppresses codec block artifacts while retaining horizon and small holes.
ref=cv2.GaussianBlur(ref,(3,3),.6)
rows=[]
for frame in [28,29,31,32,33,34,35]:
 im=cv2.imread(str(P/f'frames/frame-{frame-25:02}.png')).astype(float)/255
 im=cv2.GaussianBlur(im,(3,3),.6)
 mask=(im[:,:,1]>im[:,:,0]+.05)&(im[:,:,1]>im[:,:,2])
 mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
 yy,xx=np.where(mask);keep=np.arange(len(xx))[::2];xx=xx[keep];yy=yy[keep]
 rays=np.c_[xx,yy,np.ones(len(xx))]@np.linalg.inv(K).T
 col=im[yy,xx,:].astype(np.float32);col=col-col.mean(axis=0)
 norm=np.linalg.norm(col)
 def get(r):
  R=cv2.Rodrigues(np.radians(np.array(r)))[0];v=rays@R.T@K.T
  uv=(v[:,:2]/v[:,2:]).astype(np.float32)
  vals=cv2.remap(ref.astype(np.float32),uv[:,0:1],uv[:,1:2],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1,3)
  return uv,vals
 def cost(r):
  uv,vals=get(r);vals-=vals.mean(axis=0)
  return -float((vals*col).sum()/max(1e-8,np.linalg.norm(vals)*norm))
 bounds=[(25,37),(-3,6),(-2,2)]
 opt=differential_evolution(cost,bounds,popsize=10,maxiter=45,seed=frame,tol=1e-5,polish=False)
 row={'frame':frame,'rotation_vector_deg':opt.x.tolist(),'color_correlation':-opt.fun,'pixels':len(xx)}
 rows.append(row);print(row,flush=True)
 R=cv2.Rodrigues(np.radians(opt.x))[0];H=K@R@np.linalg.inv(K)
 warped=cv2.warpPerspective((im*255).astype('uint8'),H,(640,360));cv2.imwrite(str(P/f'rectified-{frame}.png'),warped)
(P/'rotation-probe.json').write_text(json.dumps(rows,indent=2))
