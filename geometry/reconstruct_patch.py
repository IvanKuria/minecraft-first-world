"""Measured-corner fit; no source-world or seed claim. Coordinates in source pixels."""
from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent
cube=np.float64([[0,1,0],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[0,0,1],[0,0,0]])
cliff=np.float64([[-4,1,1],[-2,1,1],[-2,1,3],[-1,1,3]])
observations={
 11:{'cube':[[545,197],[586,201],[594,221],[548,216],[583,255],[541,251],[538,234]],'cliff':[[379,200],[460,207],[456,254],[499,259]]},
 12:{'cube':[[574,213],[625,217],[640,246],[582,241],[624,286],[568,280],[564,249]],'cliff':[[372,213],[470,224],[462,300],[525,310]]},
 13:{'cube':[[604,233],[672,240],[700,282],[620,276],[671,332],[598,324],[587,283]],'cliff':[[341,243],[472,256],[460,380],[565,399]]}}
f0=240/np.tan(np.deg2rad(35));K=lambda f:np.float64([[f,0,426],[0,f,240],[0,0,1]])
proj=lambda X,r,t,f:cv2.projectPoints(X,r,t,K(f),None)[0].reshape(-1,2)
# Frame13 establishes grid hypothesis from cube+cliff, with focal free for a check.
train=np.vstack([cube,cliff]);uv=np.float64(observations[13]['cube']+observations[13]['cliff'])
ok,r,t=cv2.solvePnP(train,uv,K(f0),None,flags=cv2.SOLVEPNP_SQPNP)
fit=least_squares(lambda v:(proj(train,v[:3],v[3:6],v[6])-uv).ravel(),np.r_[r.ravel(),t.ravel(),f0],bounds=(np.r_[[-np.inf]*6,200],np.r_[[np.inf]*6,700]))
f=fit.x[6]; print('fit focal',f,'vFOV',2*np.rad2deg(np.arctan(240/f)),'train rms',np.sqrt(np.mean(fit.fun**2)*2))
result={'basis':'unit cube at origin, arbitrary world axes; top y=1','source_resolution':[852,480],'provisional_corner_uncertainty_px':2,'calibration':{'focal_px':f,'vertical_fov_deg':2*np.rad2deg(np.arctan(240/f)),'later_build_prior_focal_px':f0,'train_rms_px':float(np.sqrt(np.mean(fit.fun**2)*2))},'views':{}}
for no,ob in observations.items():
 cuv=np.float64(ob['cube']);keep=[0,1,2,4,5,6]
 if no==13:r,t=fit.x[:3],fit.x[3:6]
 else:
  ok,r,t=cv2.solvePnP(cube[keep],cuv[keep],K(f),None,flags=cv2.SOLVEPNP_SQPNP);r=r.ravel();t=t.ravel()
 pred=proj(cliff,r,t,f);err=np.linalg.norm(pred-np.float64(ob['cliff']),axis=1)
 print(no,'cliff errors',err,'RMS',np.sqrt(np.mean(err**2)))
 R=cv2.Rodrigues(r)[0];C=-R.T@t
 result['views'][str(no)]={'rotation_vector':r.tolist(),'translation':t.tolist(),'camera_world':C.tolist(),'cliff_observed':ob['cliff'],'cliff_predicted':pred.tolist(),'cliff_error_px':err.tolist(),'held_out_cliff':no!=13}
 im=cv2.imread(str(P/'frames'/f'hd{no:03}.png'))
 # Candidate surface patch is 2x2 squares plus adjacent near strip, no filled depth.
 for x in range(-4,-1):
  for z in range(1,4):
   if x==-2 and z<3:continue
   q=np.float64([[x,1,z],[x+1,1,z],[x+1,1,z+1],[x,1,z+1]])
   poly=proj(q,r,t,f).astype(int);cv2.polylines(im,[poly],True,(240,210,0),1,cv2.LINE_AA)
 for j,(a,b) in enumerate(zip(np.float64(ob['cliff']),pred)):
  cv2.circle(im,tuple(a.astype(int)),4,(0,255,0),1);cv2.drawMarker(im,tuple(b.astype(int)),(0,0,255),cv2.MARKER_CROSS,8,1)
  cv2.putText(im,'ABCD'[j],tuple((a+np.array([4,-5])).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
 p=proj(cube,r,t,f).astype(int)
 for a,b in [(0,1),(1,2),(2,3),(3,0),(2,4),(4,5),(5,3),(0,6),(6,5)]:cv2.line(im,tuple(p[a]),tuple(p[b]),(0,210,255),1,cv2.LINE_AA)
 cv2.putText(im,f'View {no}: '+('grid fit' if no==13 else 'cliff corners HELD OUT'),(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
 cv2.imwrite(str(P/f'patch-reprojection-{no}.png'),im)
result['observations']=observations
(P/'patch-fit.json').write_text(json.dumps(result,indent=2))
