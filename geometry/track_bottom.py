from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
exec(open(__file__.replace('track_bottom.py','boundary_fit.py')).read().split('results={}')[0])
anchor=3391;end=3230;cap=cv2.VideoCapture(str(P.parent/'cave-game-tech-test.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);frames={};cap.set(cv2.CAP_PROP_POS_FRAMES,end)
for idx in range(end,anchor+1):
 ok,im=cap.read()
 if not ok:break
 frames[idx]=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
H=cv2.getPerspectiveTransform(np.float32(X[:,[0,2]]),np.float32(obs[18]['bottom']))
v=np.float64(json.loads((P/'boundary-fit.json').read_text())['18']['v']);results=[]

def detect(gray,H):
 p=cv2.perspectiveTransform(np.float32(X[:,[0,2]])[None],H)[0]
 mask=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(mask,np.int32(p),255);mask=cv2.erode(mask,np.ones((17,17),np.uint8))
 pts=cv2.goodFeaturesToTrack(gray,maxCorners=600,qualityLevel=.02,minDistance=7,mask=mask)
 if pts is None:return None,None
 wxz=cv2.perspectiveTransform(pts,np.linalg.inv(H)).reshape(-1,2)
 return pts,np.column_stack([wxz[:,0],np.zeros(len(wxz)),wxz[:,1]])
pts,world=detect(frames[anchor],H)
for idx in range(anchor-1,end-1,-1):
 p1,st,er=cv2.calcOpticalFlowPyrLK(frames[idx+1],frames[idx],pts,None,winSize=(21,21),maxLevel=3)
 back,st2,er2=cv2.calcOpticalFlowPyrLK(frames[idx],frames[idx+1],p1,None,winSize=(21,21),maxLevel=3)
 keep=(st.ravel()>0)&(st2.ravel()>0)&(np.linalg.norm(back-pts,axis=2).ravel()<.8)
 uv=p1.reshape(-1,2)[keep];ww=world[keep]
 if len(uv)<30:print('STOP tracks',idx,len(uv));break
 newH,inliers=cv2.findHomography(np.float32(ww[:,[0,2]]),np.float32(uv),cv2.RANSAC,1.5)
 if newH is None:break
 keep=inliers.ravel()>0;uv=uv[keep];ww=ww[keep]
 fit=least_squares(lambda a:(proj(ww,a)-uv).ravel(),v,max_nfev=80,loss='huber',f_scale=1.)
 v=fit.x;C=-R(*v[:2]).T@v[2:5];rms=float(np.sqrt(np.mean(fit.fun**2)*2));hproj=cv2.perspectiveTransform(np.float32(X[:,[0,2]])[None],newH)[0]
 results.append({'frame':idx,'t':idx/fps,'tracks':len(uv),'rms':rms,'camera':C.tolist(),'v':v.tolist(),'projected_bottom_corners':hproj.tolist()})
 if idx%10==0:print(idx,round(idx/fps,2),len(uv),'rms',round(rms,2),'C',np.round(C,2),flush=True)
 if rms>5:print('STOP high residual');break
 H=newH;pts,world=detect(frames[idx],H)
 if pts is None:break
 if idx%10==0:
  out=cv2.cvtColor(frames[idx],cv2.COLOR_GRAY2BGR);cv2.polylines(out,[hproj.astype(int)],True,(0,255,255),1);cv2.imwrite(str(P/'boundary'/f'track-{idx}.png'),out)
(P/'bottom-backtrack.json').write_text(json.dumps(results,indent=2))
