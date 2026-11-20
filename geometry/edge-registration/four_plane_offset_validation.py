from pathlib import Path
import numpy as np,cv2,json
P=Path(__file__).parent;Q=P.parent/'phase2';d=json.loads((P/'four-texture-poses.json').read_text());matches=json.loads((P/'four-plane-correspondence.json').read_text());phase=d['player_phase'];f=180/np.tan(np.deg2rad(35));K=np.array([[f,0,320],[0,f,180],[0,0,1]])
def H(rec,delta=(0,0,0)):
 yaw,pitch,h=rec['yaw_pitch_height'];s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]]);C=np.array([phase[0]-.3*s*cp,h,phase[1]+.3*c*cp])+delta;return K@np.c_[R[:,0],R[:,2],-R@C]
out=[]
for j,mm in enumerate(matches):
 a,b=d['frames'][j:j+2];p1=np.float32([m['source']for m in mm['matches']]);p2=np.float32([m['target']for m in mm['matches']]);seen=set();keep=[]
 for i,p in enumerate(p1):
  key=tuple(np.round(p,1))
  if key not in seen:keep.append(i);seen.add(key)
 p1=p1[keep];p2=p2[keep];rows=[]
 for dy in range(-3,4):
  for dx in range(-1,2):
   for dz in range(-1,2):
    transform=H(b,(dx,dy,dz))@np.linalg.inv(H(a));pred=cv2.perspectiveTransform(p1[:,None,:],transform)[:,0,:];error=np.linalg.norm(pred-p2,axis=1);rows.append({'delta_camera_xyz':[dx,dy,dz],'count_below2px':int(np.sum(error<2)),'median_px':float(np.median(error)),'trimmed_best_half_rms':float(np.sqrt(np.mean(np.sort(error)[:len(error)//2]**2)))})
 rows.sort(key=lambda r:(-r['count_below2px'],r['median_px']));rec={'frames':mm['frames'],'unique_ratio_matches':len(p1),'ranked_integer_offset_hypotheses':rows};out.append(rec);print(mm['frames'],len(p1),rows[:6])
(P/'four-plane-offset-validation.json').write_text(json.dumps(out,indent=2))
