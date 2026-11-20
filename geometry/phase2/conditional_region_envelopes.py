from pathlib import Path
import numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;out={}
for label,script,resname,times in [('second','joint_skyline_motion.py','joint-skyline-motion.json',[152.64,152.70]),('first','joint_first_skyline_motion.py','joint-first-skyline-motion.json',[85.398,85.45])]:
 ns={'__file__':str(P/script)};exec((P/script).read_text().split('v0=np.r_')[0],ns);nom=np.array(ns['obs']);xx=ns['xx'];base=np.array(json.loads((P/resname).read_text())['parameters']);start=np.r_[base[:2],base[4:]];rows=[]
 for fov in [68,70.5]:
  for tr in times:
   def unpack(a):return np.r_[a[:2],tr,fov,a[2:]]
   for dc in [-1,0,1]:
    for dl in [-1,0,1]:
     for dr in [-1,0,1]:
      ns['obs']=[o+dc+np.where(xx<240,dl/240,dr/400)*(xx-240)for o in nom];sol=least_squares(lambda a:ns['residual'](unpack(a)),start,loss='soft_l1',f_scale=1,max_nfev=200);rows.append({'fov':fov,'reset_time':tr,'edge_bias':dc,'left_endpoint_bias':dl,'right_endpoint_bias':dr,'xz':sol.x[:2].tolist(),'rms':float(np.sqrt(np.mean(sol.fun**2)))})
 coords=np.array([a['xz']for a in rows]);low=coords.min(axis=0);high=coords.max(axis=0);box=[np.floor(low).tolist(),np.ceil(high).tolist()];out[label]={'conditional_unregistered_player_xz_box_floor_ceil':box,'tested_numeric_extrema':[low.tolist(),high.tolist()],'assumptions':['World footprint256x256; observed farcorner assigned(256,64,0)','Skyline actual outer boundary topY64','Later grounded-still reset trajectory; no horizontal input; oneRtick','Later 0.3 camera setback, zero roll, principalpoint320,180','FOV68..70.5 endpoint sensitivity; no unknown crop or distortion','Common bounded pixel-edge perturbations +/-1px; not full modelerror bound','Four axis rotations not registered; not accepted RNG intervals'],'trials':rows};print(label,box,flush=True)
(P/'conditional-region-envelopes.json').write_text(json.dumps(out,indent=2))
