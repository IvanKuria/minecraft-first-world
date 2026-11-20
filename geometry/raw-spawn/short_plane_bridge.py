"""Short geometric backlink, independent of reset physics and earlier texture phase."""
from pathlib import Path
import cv2,json,numpy as np
from scipy.optimize import least_squares
P=Path(__file__).parent;G=P.parent;ROOT=G.parent
F=180/np.tan(np.deg2rad(35));K=np.array([[F,0,320],[0,F,180],[0,0,1.]])
pose=json.loads((G/'edge-registration/four-texture-poses.json').read_text())
phase=np.array(pose['player_phase']);yaw,pitch,h=pose['frames'][0]['yaw_pitch_height']
def rc(v):
 px,pz,h,yaw,pitch=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch)
 return np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]]),np.array([px-.3*s*cp,h,pz+.3*c*cp])
def project(v,W):
 R,C=rc(v);q=(W-C)@R.T;return F*q[:,:2]/q[:,2:]+[320,180]
refv=np.r_[phase,h,yaw,pitch];R,C=rc(refv)
ref=cv2.imread(str(ROOT/'timing/original-reset2-2959.png'));sift=cv2.SIFT_create(nfeatures=10000)
blue,green,red=cv2.split(ref.astype(float));mask=((green>red*1.15)&(green>blue*1.15)).astype('uint8')*255
mask=cv2.erode(mask,np.ones((5,5),np.uint8));mask[:180]=0
kr,dr=sift.detectAndCompute(cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY),mask)
results=[]
for idx in range(2952,2960):
 im=cv2.imread(str(ROOT/f'timing/original-reset2-{idx}.png'));ki,di=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
 pairs=cv2.BFMatcher().knnMatch(dr,di,k=2);reverse=cv2.BFMatcher().knnMatch(di,dr,k=2)
 rev={a.queryIdx:a.trainIdx for a,b in reverse if a.distance<.7*b.distance}
 good=[a for a,b in pairs if a.distance<.7*b.distance and rev.get(a.trainIdx)==a.queryIdx]
 a=np.array([kr[m.queryIdx].pt for m in good]);b=np.array([ki[m.trainIdx].pt for m in good]);out=dict(frame=idx,matches=len(good))
 if len(good)<10: results.append(out);continue
 H,inside=cv2.findHomography(a,b,cv2.USAC_MAGSAC,2.0);sel=np.flatnonzero(inside.ravel());a=a[sel]+.5;b=b[sel]+.5
 rays=np.c_[(a-[320,180])/F,np.ones(len(a))]@R
 W=C-rays*(C[1]/rays[:,1])[:,None]
 order=np.argsort(a[:,0]);train=order[::2];test=order[1::2]
 v0=np.r_[phase,10.5,np.deg2rad(32),np.deg2rad(15)] if idx<2959 else refv.copy()
 sol=least_squares(lambda v:(project(v,W[train])-b[train]).ravel(),v0,loss='soft_l1',f_scale=1.,max_nfev=500)
 error=np.linalg.norm(project(sol.x,W)-b,axis=1)
 out.update(homography_inliers=len(a),parameters=sol.x.tolist(),player_xz_delta_from_reference=(sol.x[:2]-phase).tolist(),yaw_pitch_deg=np.rad2deg(sol.x[3:]).tolist(),train_median_px=float(np.median(error[train])),heldout_median_px=float(np.median(error[test])),heldout_p90_px=float(np.percentile(error[test],90)),heldout_count=len(test),reference_pixel_extent=np.ptp(a,axis=0).tolist())
 for point,pred in zip(b[test],project(sol.x,W[test])):
  cv2.circle(im,tuple(np.rint(point-.5).astype(int)),3,(0,255,0),1);cv2.line(im,tuple(np.rint(point-.5).astype(int)),tuple(np.rint(pred-.5).astype(int)),(0,0,255),1)
 cv2.imwrite(str(P/f'bridge-{idx}.png'),im);results.append(out)
 print(json.dumps(out),flush=True)
(P/'short-plane-bridge.json').write_text(json.dumps({'reference_frame':2959,'reference_pose':refv.tolist(),'selection':'Mutual SIFT ratio.7 then2px homography selection; alternating spatially sorted matches train/test. Selection uses all correspondences, so heldout is conditional, not entirely independent.','results':results},indent=2))
