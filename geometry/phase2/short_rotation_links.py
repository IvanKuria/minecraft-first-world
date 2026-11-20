from pathlib import Path
import cv2,numpy as np,json
P=Path(__file__).parent;f=180/np.tan(np.deg2rad(35));K=np.array([[f,0,320],[0,f,180],[0,0,1]]);sift=cv2.SIFT_create(nfeatures=6000);out=[]
for a,b in [('152.951','153.050'),('153.050','153.150'),('153.150','153.300')]:
 i1=cv2.imread(str(P/f'f-{a}.png'));i2=cv2.imread(str(P/f'f-{b}.png'));k1,d1=sift.detectAndCompute(i1,None);k2,d2=sift.detectAndCompute(i2,None);pairs=cv2.BFMatcher().knnMatch(d1,d2,k=2);m=[x for x,y in pairs if x.distance<.65*y.distance];p1=np.float32([k1[x.queryIdx].pt for x in m]);p2=np.float32([k2[x.trainIdx].pt for x in m]);H,mask=cv2.findHomography(p1,p2,cv2.USAC_MAGSAC,2.);n,rot,trans,norm=cv2.decomposeHomographyMat(H,K);solutions=[]
 for R in rot:
  angle=np.rad2deg(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)));solutions.append({'total_rotation_degrees':float(angle),'rotation_matrix':R.tolist()})
 rec={'source':a,'target':b,'ratio_matches':len(m),'homography_inliers':int(mask.sum()),'rotation_solutions':solutions};out.append(rec);print(a,b,len(m),int(mask.sum()),[round(s['total_rotation_degrees'],2)for s in solutions]);draw=cv2.drawMatches(i1,k1,i2,k2,[mm for mm,yes in zip(m,mask.ravel())if yes][:80],None,flags=2);cv2.imwrite(str(P/f'rotation-links-{a}-{b}.jpg'),draw)
(P/'short-rotation-links.json').write_text(json.dumps(out,indent=2))
