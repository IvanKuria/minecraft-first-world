"""Rotation-align observed far features; check for large newly filled sky areas."""
from pathlib import Path
import json
import cv2
import numpy as np
import fit_opening as m

P=Path(__file__).parent
K=np.array([[m.F,0,319.5],[0,m.F,179.5],[0,0,1.]])

def rotation(fr):
    ya,pi,*_=fr['camera_parameters'];s,c=np.sin(ya),np.cos(ya);sp,cp=np.sin(pi),np.cos(pi)
    return np.array([[c,-s*sp,s*cp],[0,-cp,-sp],[s,c*sp,-c*cp]])

frames=m.pose['frames'];images=[cv2.imread(str(m.SOURCE/f"f-{fr['frame_name']}.png")) for fr in frames]
results=[]
for i,j in [(0,1),(0,2),(1,2)]:
    H=K@rotation(frames[j]).T@rotation(frames[i])@np.linalg.inv(K)
    warped=cv2.warpPerspective(images[i],H,(640,360))
    valid=cv2.warpPerspective(np.ones((360,640),np.uint8),H,(640,360),flags=cv2.INTER_NEAREST)>0
    target=images[j]
    sky=lambda im: ((im[:,:,0].astype(float)-im[:,:,1])>15)&(im[:,:,2]<170)
    sky1=sky(warped);sky2=sky(target)
    kernel=np.ones((5,5),np.uint8)
    # Require a mismatch safely inside both regions, rather than subpixel edges.
    new_ground=(cv2.erode(sky1.astype(np.uint8),kernel)>0)&(cv2.erode((~sky2).astype(np.uint8),kernel)>0)&valid
    removed_ground=(cv2.erode((~sky1).astype(np.uint8),kernel)>0)&(cv2.erode(sky2.astype(np.uint8),kernel)>0)&valid
    roi=np.zeros((360,640),bool);roi[150:195,8:632]=True
    gray1=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY).astype(float);gray2=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY).astype(float)
    feature_roi=np.zeros((360,640),bool);feature_roi[170:190,20:620]=True
    use=feature_roi&valid
    results.append({'source_frame':frames[i]['frame_name'],'target_frame':frames[j]['frame_name'],'rotation_homography':H.tolist(),'far_feature_region':[20,170,620,190],'far_grayscale_correlation':float(np.corrcoef(gray1[use],gray2[use])[0,1]),'new_ground_pixels_after_2px_erosion':int((new_ground&roi).sum()),'removed_ground_pixels_after_2px_erosion':int((removed_ground&roi).sum()),'caveat':'Rotation-only warp neglects small rotation-dependent camera-center shift; correlation is not a proof of full world loading.'})
    strip=np.concatenate([warped[150:195],target[150:195]],axis=0)
    cv2.imwrite(str(P/f'rotation-compare-{i}-{j}.png'),cv2.resize(strip,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
(P/'skyline-consistency.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
