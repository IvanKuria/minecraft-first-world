from pathlib import Path
import numpy as np,json
from scipy.optimize import minimize
P=Path(__file__).parent;ns={'__file__':str(P/'fit_late_grass.py')};exec((P/'fit_late_grass.py').read_text().split('for rot in range')[0],ns);v0=np.array(json.loads((P/'late-grass-fit.json').read_text())[1]['parameters']);out=[]
for fov in [66,68,69,70,70.5,72,74]:
 f=180/np.tan(np.deg2rad(fov/2));rx=(ns['X'].ravel()+.5-320)/f;ry=(ns['Y'].ravel()+.5-180)/f;rxt=(ns['XT'].ravel()+.5-320)/f;ryt=(ns['YT'].ravel()+.5-180)/f
 def loss(v):a=ns['values'](v,rx,ry);a-=a.mean();return -a@ns['obs']/(np.linalg.norm(a)+1e-9)
 start=v0.copy();start[2]*=70/fov;sol=minimize(loss,start,method='Nelder-Mead',options={'maxiter':2500});v=sol.x;held=np.corrcoef(ns['values'](v,rxt,ryt),ns['im'][ns['YT'],ns['XT']].ravel())[0,1];y,p,h,px,pz=v;out.append({'fov':fov,'parameters':v.tolist(),'train':float(-sol.fun),'heldout':float(held),'interpolated_player_phase':[(px+.3*np.sin(y)*np.cos(p))%1,(pz-.3*np.cos(y)*np.cos(p))%1]});print(out[-1],flush=True)
(P/'late-grass-fov-profile.json').write_text(json.dumps(out,indent=2))
