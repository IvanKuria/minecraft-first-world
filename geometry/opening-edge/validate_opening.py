"""Pose/measurement sensitivity and frame holdouts, still candidate-blind."""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
import fit_opening as m

P=Path(__file__).parent
original=json.loads((P/'fit-results.json').read_text())
obs=np.array(original['observed_edge_rows'])
out=[]
for variant,pose in enumerate(m.poses):
  m.F=180/np.tan(np.deg2rad(pose['fov']/2))
  for held in range(3):
    train=[j for j in range(3) if j!=held]
    for delta in range(3):
      for bias in [-1,-.5,0,.5,1]:
        def residual(x,ids=train):
          return np.concatenate([m.boundary(x,pose['frames'][j],delta)+bias-obs[j] for j in ids])
        best=None
        for start in [[70,60],[180,150],[235,210]]:
          fit=least_squares(residual,start,bounds=([.5,.5],[255.5,255.5]),loss='soft_l1',f_scale=.5,max_nfev=100)
          mse=np.mean(residual(fit.x)**2)
          if best is None or mse<best[0]:best=(mse,fit.x)
        mse,x=best
        out.append({'pose_variant':variant,'pose_mode':pose['mode'],'fov':pose['fov'],'heldout_frame_index':held,'height_delta':delta,'common_pixel_bias':bias,'player_xz':x.tolist(),'train_rmse':float(mse**.5),'heldout_rmse':float(np.mean(residual(x,[held])**2)**.5)})

# Independent color-midpoint extractor: threshold RGB sky/ground color
# halfway between median sky and a local ground estimate 4–7 rows below.
# Chroma is subsampled; this is an alternative, not more precise truth.
alt=[]
for j,frame in enumerate(m.pose['frames']):
 im=cv2.imread(str(m.SOURCE/f"f-{frame['frame_name']}.png")).astype(float)
 vals=[]
 for k,x in enumerate(m.xs.astype(int)):
  line=np.median(im[:,x-3:x+4],axis=1)
  yy=obs[j,k];sky=np.median(line[int(yy)-10:int(yy)-5],axis=0)
  ground=np.median(line[int(yy)+4:int(yy)+8],axis=0)
  direction=ground-sky;mix=(line-sky)@direction/(direction@direction)
  start=int(yy)-4;end=int(yy)+5
  idx=start+np.argmax(mix[start:end]>=.5)
  prev=mix[idx-1];curr=mix[idx]
  vals.append(float(idx-1+(.5-prev)/(curr-prev)))
 alt.append(vals)
out2=[]
m.F=180/np.tan(np.deg2rad(m.pose['fov']/2))
alt=np.array(alt)
for delta in [0,1,2]:
 for bias in [-1,-.5,0,.5,1]:
  def residual(x,ids=[0,1]):return np.concatenate([m.boundary(x,m.pose['frames'][j],delta)+bias-alt[j] for j in ids])
  best=None
  for start in [[70,60],[180,150],[235,210]]:
   f=least_squares(residual,start,bounds=([.5,.5],[255.5,255.5]),loss='soft_l1',f_scale=.5,max_nfev=100);mse=np.mean(residual(f.x)**2)
   if best is None or mse<best[0]:best=(mse,f.x)
  mse,x=best
  out2.append({'height_delta':delta,'common_pixel_bias':bias,'player_xz':x.tolist(),'train_rmse':float(mse**.5),'heldout_rmse':float(np.mean(residual(x,[2])**2)**.5)})
(P/'sensitivity.json').write_text(json.dumps({'pose_and_leave_one_out':out,'alternative_color_midpoint_rows':alt.tolist(),'alternative_measurement_fits':out2,'color_minus_luma_mean_by_frame':np.mean(alt-obs,axis=1).tolist(),'color_minus_luma_rms_by_frame':np.sqrt(np.mean((alt-obs)**2,axis=1)).tolist()},indent=2)+'\n')
print('Saved',len(out),'pose/holdout fits and',len(out2),'measurement alternatives')
for mode in ['fixed70','free_fov']:
 for delta in [0,1,2]:
  keep=[r for r in out if r['pose_mode']==mode and r['height_delta']==delta and r['train_rmse']<.5 and r['heldout_rmse']<.5]
  print(mode,delta,'fits under .5 px',len(keep),'range',np.min([r['player_xz'] for r in keep],axis=0).tolist()if keep else None,np.max([r['player_xz'] for r in keep],axis=0).tolist()if keep else None)
print('Alternative measurement means',np.mean(alt-obs,axis=1))
