"""Actual original-footage measurements; conditional geometry tests, not seed recovery."""
from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent
f=180/np.tan(np.radians(35));K=np.array([[f,0,320],[0,f,180],[0,0,1.]])
def frame(n):return cv2.imread(str(P/f'frames/frame-{n-25:02}.png'))
def rot_fit(u,v,train):
 rays=np.c_[u,np.ones(len(u))]@np.linalg.inv(K).T
 def pred(r):
  q=rays@cv2.Rodrigues(r)[0].T@K.T;return q[:,:2]/q[:,2:]
 opt=least_squares(lambda r:(pred(r)[train]-v[train]).ravel(),[.4,0,0],loss='soft_l1')
 return opt.x,pred(opt.x)
def translation_fit(u,v,train):
 rays=np.c_[u,np.ones(len(u))]@np.linalg.inv(K).T
 def pred(p):
  H=np.eye(3)+np.outer(p[:3],p[3:]);q=rays@H.T@K.T;return q[:,:2]/q[:,2:]
 opts=[]
 for x in [[0,-.4,0,0,1,0],[0,-.3,.2,0,.8,-.5],[0,-.1,0,0,0,1]]:
  o=least_squares(lambda p:(pred(p)[train]-v[train]).ravel(),x,loss='soft_l1',max_nfev=500)
  opts.append(o)
 opt=min(opts,key=lambda o:o.cost);return opt.x,pred(opt.x)
def rms(p,v,m):return float(np.mean(np.sum((p[m]-v[m])**2,axis=1))**.5)
d=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.01);ref=frame(40);kr,dr=d.detectAndCompute(ref,None);motion=[]
for n in [35,36,37,38,39]:
 ks,ds=d.detectAndCompute(frame(n),None);good=[a for a,b in cv2.BFMatcher().knnMatch(ds,dr,k=2) if a.distance<.7*b.distance]
 u=np.float64([ks[m.queryIdx].pt for m in good]);v=np.float64([kr[m.trainIdx].pt for m in good]);H,mask=cv2.findHomography(u,v,cv2.RANSAC,2)
 u=u[mask.ravel()>0];v=v[mask.ravel()>0]
 _,unique=np.unique(np.round(u,2),axis=0,return_index=True);u=u[unique];v=v[unique]
 train=np.arange(len(u))%3!=0;test=~train
 r,pred=rot_fit(u,v,train);t,tpred=translation_fit(u,v,train)
 row={'source_frame':n,'reference_frame':40,'correspondences':len(u),'train_count':int(train.sum()),'heldout_count':int(test.sum()),'rotation_vector_degrees':np.degrees(r).tolist(),'rotation_train_rms_px':rms(pred,v,train),'rotation_heldout_rms_px':rms(pred,v,test),'translation_plane_train_rms_px':rms(tpred,v,train),'translation_plane_heldout_rms_px':rms(tpred,v,test),'source_pixels':u.tolist(),'reference_pixels':v.tolist()}
 motion.append(row)
 overlay=ref.copy()
 for a,b,m in zip(pred,v,test):
  if m:cv2.circle(overlay,tuple(np.round(b).astype(int)),3,(0,255,0),1);cv2.drawMarker(overlay,tuple(np.round(a).astype(int)),(0,0,255),cv2.MARKER_CROSS,7,1)
 cv2.imwrite(str(P/f'rotation-heldout-{n}.png'),overlay)
 print({k:v for k,v in row.items() if 'pixels' not in k})
# Right silhouette is sampled only in the small intact segment at rows351..358.
points=[]
for n in [29,31,32,33]:
 im=frame(n).astype(int);mask=(im[:,:,0]<150)&(im[:,:,1]<180)
 pts=np.array([[y,np.where(mask[y])[0].max()] for y in range(351,359)],dtype=float)
 slope,intercept=np.polyfit(pts[:,0],pts[:,1],1)
 points.append({'frame':n,'row_and_rightmost_nonsky_x':pts.tolist(),'dx_dy':float(slope),'intercept':float(intercept)})
def common_vp(data,vh):
 ys=np.array([np.array(a)[:,0]-vh for a in data]);xx=np.array([np.array(a)[:,1] for a in data]);num=len(data)-1;X=np.zeros((8*num,num+1));X[:,0]=1
 for i in range(num):X[i*8:(i+1)*8,i+1]=ys[i]
 co=np.linalg.lstsq(X,xx[:num].ravel(),rcond=None)[0]
 h=ys[-1];mh=np.dot(h,xx[-1]-co[0])/np.dot(h,h)
 row={'assumed_horizon_y':vh,'fitted_vanishing_x':float(co[0]),'fitted_slopes':co[1:].tolist()+[float(mh)],'train_rms_px':float(np.mean((X@co-xx[:num].ravel())**2)**.5),'heldout_rms_px':float(np.mean((co[0]+mh*h-xx[-1])**2)**.5)}
 if vh==180:
  cos_yaw=np.cos(np.arctan((co[0]-320)/f))
  row['conditional_eye_heights_for_adjacent_16unit_planes']=(16/(np.diff(co[1:])*cos_yaw)).tolist()
  row['conditional_plane_spacings_if_eye_height_1_62']=(np.diff(co[1:])*cos_yaw*1.62).tolist()
 return row
tests=[common_vp([p['row_and_rightmost_nonsky_x'] for p in points],vh) for vh in [345,347,349,180]]
# Positive control: four genuine parallel ground cuts, spaced16worldunits, same pixel rows.
rng=np.random.default_rng(1023);vh=347;alpha=np.arctan((vh-180)/f);h=1.62
control=[]
for dx in [-24,-8,8,24]:
 y=np.arange(351,359,dtype=float);x=320+dx*np.cos(alpha)/h*(y-vh)+rng.normal(0,.3,8);control.append(np.c_[y,x].tolist())
positive=common_vp(control,347)
# Independent known-rotation positive control, with held-out correspondences.
u=np.c_[rng.uniform(40,600,60),rng.uniform(280,355,60)];rays=np.c_[u,np.ones(60)]@np.linalg.inv(K).T;known=np.radians([30,2,-.4]);q=rays@cv2.Rodrigues(known)[0].T@K.T;v=q[:,:2]/q[:,2:]+rng.normal(0,.3,(60,2));train=np.arange(60)%3!=0;r,pred=rot_fit(u,v,train)
rotation_control={'known_rotation_vector_degrees':np.degrees(known).tolist(),'recovered_rotation_vector_degrees':np.degrees(r).tolist(),'heldout_rms_px':rms(pred,v,~train)}
out={'original_video':'footage-sources/wayback-fakeurl.mp4','resolution':[640,360],'frame_numbering':'zero-based decoded frames; extracted filename index = original frame -25','focal_px':float(f),'motion':motion,'frontiers':points,'conditional_parallel_ground_plane_tests':tests,'synthetic_parallel_ground_plane_positive_control':positive,'synthetic_rotation_positive_control':rotation_control,'limitations':['No historical ground elevation or camera pose assumed measured.','Correspondences filtered by full homography RANSAC before splitting; heldout excludes rotation parameter fit, not correspondence selection.','Only8vertical pixels per frontier; compression/partial silhouettes limit line interpretation.','No modulo16 or global spawn coordinate recovered.']}
(P/'measurements.json').write_text(json.dumps(out,indent=2)+'\n')
print('frontier',tests,'positive',positive,'rotationcontrol',rotation_control)
