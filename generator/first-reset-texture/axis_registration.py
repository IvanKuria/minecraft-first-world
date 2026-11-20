from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;G=P.parents[1]/'geometry';f=180/np.tan(np.radians(35));K=np.array([[f,0,320],[0,f,180],[0,0,1.]])
source=cv2.imread(str(G/'first-reset/frame-1658.png'));target=cv2.imread(str(P/'f-86.00.png'))
d=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.005);ks,ds=d.detectAndCompute(source,None);kt,dt=d.detectAndCompute(target,None);good=[a for a,b in cv2.BFMatcher().knnMatch(ds,dt,k=2)if a.distance<.7*b.distance]
u=np.float64([ks[m.queryIdx].pt for m in good]);v=np.float64([kt[m.trainIdx].pt for m in good]);H,mask=cv2.findHomography(u,v,cv2.RANSAC,3);u=u[mask.ravel()>0];v=v[mask.ravel()>0]
# Furthest visible features minimize translational parallax; source grass starts~150.
keep=u[:,1]<235;u=u[keep];v=v[keep];_,unique=np.unique(np.round(u,2),axis=0,return_index=True);u=u[unique];v=v[unique]
def R(ya,pi):
 s,c=np.sin(ya),np.cos(ya);sp,cp=np.sin(pi),np.cos(pi);return np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]])
late=max(json.loads((P/'fit-86.00.json').read_text()),key=lambda r:r['train'])['parameters'];rl=R(*late[:2]);rays=np.c_[u,np.ones(len(u))]@np.linalg.inv(K).T
j=json.loads((G/'phase2/joint-first-skyline-motion.json').read_text());a=np.radians(j['angles_degrees'][-1]);rows=[]
for quadrant in range(4):
 guess=a+[quadrant*np.pi/2,0]
 def project(p):
  q=rays@R(*p)@rl.T@K.T
  return q[:,:2]/q[:,2:]
 fit=least_squares(lambda p:(project(p)-v).ravel(),guess,bounds=(guess-np.radians([15,15]),guess+np.radians([15,15])),loss='soft_l1')
 residual=np.linalg.norm(project(fit.x)-v,axis=1)
 rows.append({'quadrant_rotation_degrees':90*quadrant,'source_fitted_yaw_pitch':np.degrees(fit.x).tolist(),'rms_px':float(np.mean(residual**2)**.5),'median_px':float(np.median(residual)),'source_angle_initial_deg':np.degrees(guess).tolist()})
print('matches',len(u),rows)
# Reprojection overlay correct quadrant.
best=min(rows,key=lambda r:r['rms_px']);pp=np.radians(best['source_fitted_yaw_pitch']);q=rays@R(*pp)@rl.T@K.T;pred=q[:,:2]/q[:,2:];overlay=target.copy()
for aa,bb in zip(pred,v):cv2.circle(overlay,tuple(np.round(bb).astype(int)),3,(0,255,0),1);cv2.drawMarker(overlay,tuple(np.round(aa).astype(int)),(0,0,255),cv2.MARKER_CROSS,7,1)
cv2.imwrite(str(P/'axis-registration-overlay.png'),overlay)
(P/'axis-registration.json').write_text(json.dumps({'source_frame':1658,'target_frame':1663,'correspondences':len(u),'source_pixels':u.tolist(),'target_pixels':v.tolist(),'fits':rows,'limitations':['Approximate rotation-only far-feature fit; rejects90degreeambiguity, not a precision pose.','Correspondences homography-filtered; not a heldoutregistrationtest.','Skyline corner model andabsoluteXYbounds remainconditional.']},indent=2))
