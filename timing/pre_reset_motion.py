"""Test whether pre-reset images admit a common camera center plus rotation.

Image residuals cannot alone prove onGround or recover world velocity.
"""
import json, argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'timing'
PTS=json.loads((OUT/'original2011-cadence.json').read_text())['pts_seconds']
parser=argparse.ArgumentParser()
parser.add_argument('--reference',type=int,default=2951)
parser.add_argument('--start',type=int,default=2931)
parser.add_argument('--label',default='pre-reset')
args=parser.parse_args()
SELECT=list(range(args.start,args.reference+1))
cap=cv2.VideoCapture(str(BASE/'footage-sources/wayback-fakeurl.mp4'))
frames={}
for n in range(args.reference+1):
    ok,im=cap.read()
    if not ok:break
    if n in SELECT: frames[n]=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cap.release()
sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.025)
features={n:sift.detectAndCompute(im,None) for n,im in frames.items()}
ref=args.reference;kp1,d1=features[ref];matcher=cv2.BFMatcher()
f=180/np.tan(np.deg2rad(35));K=np.array([[f,0,320],[0,f,180],[0,0,1.]])
records=[]
for n in SELECT[:-1]:
    kp0,d0=features[n]
    pairs=matcher.knnMatch(d0,d1,k=2)
    matches=[a for a,b in pairs if a.distance<.65*b.distance]
    if len(matches)<4:
        records.append({'frame':n,'pts':PTS[n],'status':'insufficient descriptor matches','ratio_test_matches':len(matches)})
        continue
    p0=np.array([kp0[m.queryIdx].pt for m in matches]);p1=np.array([kp1[m.trainIdx].pt for m in matches])
    H,mask=cv2.findHomography(p0,p1,cv2.USAC_MAGSAC,1.5)
    if mask is None or int(mask.sum())<8:
        records.append({'frame':n,'pts':PTS[n],'status':'insufficient homography inliers',
                        'ratio_test_matches':len(matches),'homography_inliers':int(mask.sum()) if mask is not None else 0})
        continue
    mask=mask.ravel().astype(bool);p0,p1=p0[mask],p1[mask]
    rays=np.column_stack([p0,np.ones(len(p0))])@np.linalg.inv(K).T
    def project(rotvec):
        R=cv2.Rodrigues(rotvec)[0]
        screen=(rays@R.T)@K.T
        return screen[:,:2]/screen[:,2,None]
    fit=least_squares(lambda r:(project(r)-p1).ravel(),np.zeros(3),loss='soft_l1',f_scale=.5)
    residual=np.linalg.norm(project(fit.x)-p1,axis=1)
    record={'frame':n,'pts':PTS[n],'reference_frame':ref,'reference_pts':PTS[ref],
            'ratio_test_matches':len(matches),'homography_inliers':len(p0),
            'rotation_vector_deg':np.rad2deg(fit.x).tolist(),
            'raw_displacement_px_percentiles':np.percentile(np.linalg.norm(p1-p0,axis=1),[50,90,99]).tolist(),
            'rotation_only_residual_px_percentiles':np.percentile(residual,[50,90,99]).tolist(),
            'image_mean_abs_difference':float(np.abs(frames[n].astype(float)-frames[ref]).mean()),
            'point_bbox':[p0.min(axis=0).tolist(),p0.max(axis=0).tolist()]}
    records.append(record)
    if n in [args.start,ref-6,ref-3,ref-1]:
        im=cv2.cvtColor(frames[ref],cv2.COLOR_GRAY2BGR)
        for before,after,error in zip(project(fit.x),p1,residual):
            if error>3:continue
            cv2.circle(im,tuple(np.round(after).astype(int)),2,(0,220,0),1)
            cv2.line(im,tuple(np.round(after).astype(int)),tuple(np.round(after+10*(before-after)).astype(int)),(0,0,255),1)
        cv2.imwrite(str(OUT/f'{args.label}-motion-{n}.png'),im)
    print(json.dumps(record),flush=True)
(OUT/f'{args.label}-motion.json').write_text(json.dumps({'method':'SIFT ratio0.65, homography MAGSAC1.5px then robust rotation-only fit at70deg FOV',
    'limits':['Homography inlier selection can suppress nonplanar motion; not an exhaustive translation test',
              'Small residuals support approximate stationarity but do not establish collision state or raw velocity',
              'Subpixel feature localization in compressed video may be systematically biased'],
    'records':records},indent=2)+'\n')
