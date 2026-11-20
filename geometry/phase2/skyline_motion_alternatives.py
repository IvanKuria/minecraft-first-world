from pathlib import Path
import numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;out=[]
for label,script,results,trange in [('second','joint_skyline_motion.py','joint-skyline-motion.json',(152.64,152.70)),('first','joint_first_skyline_motion.py','joint-first-skyline-motion.json',(85.398,85.45))]:
 for scenario in ['grounded_still','grounded_jump','falling_terminal']:
  ns={'__file__':str(P/script)};code=(P/script).read_text().split('v0=np.r_')[0].replace("a['scenario']=='grounded_still'",f"a['scenario']=='{scenario}'");exec(code,ns);v=np.array(json.loads((P/results).read_text())['parameters']);v[2]=np.clip(v[2],*trange);ang=ns['ang'];lo=np.r_[0,0,trange[0],68,(ang-.1).ravel()];hi=np.r_[256,256,trange[1],70.5,(ang+.1).ravel()];sol=least_squares(ns['residual'],v,bounds=(lo,hi),loss='soft_l1',f_scale=1,max_nfev=500);rec={'reset':label,'scenario':scenario,'x':float(sol.x[0]),'z':float(sol.x[1]),'reset_time':float(sol.x[2]),'fov':float(sol.x[3]),'skyline_rms_px':float(np.sqrt(np.mean(sol.fun**2))),'max_abs_residual':float(np.max(np.abs(sol.fun))),'parameters':sol.x.tolist()};out.append(rec);print(rec,flush=True)
(P/'skyline-motion-alternatives.json').write_text(json.dumps(out,indent=2))
