from pathlib import Path
import numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;ns={'__file__':str(P/'joint_clear_grass.py')};code=(P/'joint_clear_grass.py').read_text().split('v0=np.r_')[0].replace('cx=px-.3*s*cp;cz=pz+.3*c*cp','cx=px-v[-1]*s*cp;cz=pz+v[-1]*c*cp');exec(code,ns);base=np.array(json.loads((P/'joint-clear-grass.json').read_text())['parameters']);a0=np.r_[base[:2],base[3:],.3]
def unwrap(a):return np.r_[a[:2],70.,a[2:]]
sol=least_squares(lambda a:ns['residual'](unwrap(a)),a0,x_scale=np.r_[.05,.05,[.01,.01,.1]*3,.05],max_nfev=1000,ftol=1e-10,xtol=1e-10);v=unwrap(sol.x);out={'status':'conditional constantplayerxz andFOV70; setbackfree, not independentlyconfirmedcode','player_phase':(v[:2]%1).tolist(),'camera_setback':float(v[-1]),'frames':[]}
for i,d in enumerate(ns['data']):out['frames'].append({'train':float(np.corrcoef(ns['prediction'](v,i),ns['ob'][i])[0,1]),'heldout':float(np.corrcoef(ns['prediction'](v,i,True),d['im'][d['YT'],d['XT']].ravel())[0,1])})
print(out);(P/'camera-setback-fit.json').write_text(json.dumps(out,indent=2))
