"""Test overlap between terrain visible at the two respawns; not RNG inference."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'timing'
pts=json.loads((OUT/'original2011-cadence.json').read_text())['pts_seconds']
groups={
 'first':[i for i,t in enumerate(pts) if 85.45<=t<=85.92],
 'second':[i for i,t in enumerate(pts) if 152.744<=t<=153.0],
 'control':[i for i,t in enumerate(pts) if 5.0<=t<=5.31]}
selected=set(sum(groups.values(),[]));frames={}
cap=cv2.VideoCapture(str(BASE/'footage-sources/wayback-fakeurl.mp4'))
for i in range(max(selected)+1):
 ok,im=cap.read()
 if not ok:break
 if i in selected:frames[i]=im
cap.release()
sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.025)
features={i:sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None) for i,im in frames.items()}
matcher=cv2.BFMatcher();rows=[];best=None
for group in ['first','control']:
 for i in groups[group]:
  for j in groups['second']:
   k0,d0=features[i];k1,d1=features[j]
   pairs=matcher.knnMatch(d0,d1,k=2)
   good=[a for a,b in pairs if a.distance<.65*b.distance]
   if len(good)<4:continue
   p0=np.array([k0[m.queryIdx].pt for m in good]);p1=np.array([k1[m.trainIdx].pt for m in good])
   H,mask=cv2.findHomography(p0,p1,cv2.USAC_MAGSAC,2)
   if mask is None:continue
   valid=mask.ravel().astype(bool);count=int(valid.sum())
   row={'group':group,'first_frame':i,'first_pts':pts[i],'second_frame':j,'second_pts':pts[j],
        'ratio_matches':len(good),'homography_inliers':count,
        'first_inlier_bbox':[p0[valid].min(axis=0).tolist(),p0[valid].max(axis=0).tolist()] if count else None,
        'homography':H.tolist() if H is not None else None}
   rows.append(row)
   if group=='first' and (best is None or count>best[0]):best=(count,i,j,k0,k1,good,valid)
if best:
 count,i,j,k0,k1,good,valid=best
 im=cv2.drawMatches(frames[i],k0,frames[j],k1,[m for m,v in zip(good,valid) if v],None,flags=2)
 cv2.imwrite(str(OUT/'cross-respawn-best-matches.jpg'),im)
 pair=Image.new('RGB',(1280,384));draw=ImageDraw.Draw(pair)
 for col,n in enumerate([i,j]):
  pair.paste(Image.fromarray(cv2.cvtColor(frames[n],cv2.COLOR_BGR2RGB)),(col*640,24))
  draw.text((col*640+8,5),f'Original 2011: frame {n}, PTS {pts[n]}',fill='white')
 pair.save(OUT/'cross-respawn-best-pair.jpg')
result={'method':'SIFT0.65 ratio; MAGSAC homography2px; feature overlap alone does not establish identical camera position',
        'records':rows}
(OUT/'cross-respawn-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for group in ['first','control']:
 print(group,sorted([r for r in rows if r['group']==group],key=lambda r:r['homography_inliers'],reverse=True)[:3])
