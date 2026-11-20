from pathlib import Path
import numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;ns={'__file__':str(P/'joint_clear_grass.py')};exec((P/'joint_clear_grass.py').read_text().split('v0=np.r_')[0],ns);base=np.array(json.loads((P/'joint-clear-grass.json').read_text())['parameters']);original=[(d['X'].copy(),d['Y'].copy())for d in ns['data']];out=[]
def unwrap(a):return np.r_[a[:2],70.,a[2:]]
for variant in ['full','left_half','right_half','top_half','bottom_half','robust_full']:
 for d,(X,Y)in zip(ns['data'],original):
  if variant=='left_half':mask=X<X.mean()
  elif variant=='right_half':mask=X>=X.mean()
  elif variant=='top_half':mask=Y<Y.mean()
  elif variant=='bottom_half':mask=Y>=Y.mean()
  else:mask=np.ones_like(X,bool)
  d['X']=X[mask];d['Y']=Y[mask]
 ns['ob']=[ns['normalize'](d['im'][d['Y'],d['X']].ravel())for d in ns['data']];a0=np.r_[base[:2],base[3:]];sol=least_squares(lambda a:ns['residual'](unwrap(a)),a0,x_scale=np.r_[.05,.05,[.01,.01,.1]*3],loss='soft_l1'if variant=='robust_full'else'linear',f_scale=.02,max_nfev=700,ftol=1e-10,xtol=1e-10);v=unwrap(sol.x);rec={'variant':variant,'fixed_fov':70,'shared_player_phase':(v[:2]%1).tolist(),'frames':[]}
 for i,d in enumerate(ns['data']):rec['frames'].append({'train':float(np.corrcoef(ns['prediction'](v,i),ns['ob'][i])[0,1]),'heldout':float(np.corrcoef(ns['prediction'](v,i,True),d['im'][d['YT'],d['XT']].ravel())[0,1])})
 out.append(rec);print(rec,flush=True)
(P/'clear-phase-roi-sensitivity.json').write_text(json.dumps(out,indent=2))
