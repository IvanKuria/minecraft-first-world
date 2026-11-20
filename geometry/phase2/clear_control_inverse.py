from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;root=P.parent.parent;hd=cv2.imread(str(root/'runtime/render-clear-control-hd/actual-game-render.png'));native=cv2.imread(str(root/'runtime/render-clear-control-native/actual-game-render.png'));sources={'native':native,'HD-area':cv2.resize(hd,(640,360),interpolation=cv2.INTER_AREA),'HD-lanczos':cv2.resize(hd,(640,360),interpolation=cv2.INTER_LANCZOS4)};ns={'__file__':str(P/'fit_late_grass.py')};exec((P/'fit_late_grass.py').read_text().split('for rot in range')[0],ns);truth=np.array(json.loads((P/'late-grass-fit.json').read_text())[1]['parameters']);out=[]
def norm(a):return(a-a.mean())/(a.std()+1e-9)
for label,im in sources.items():
 cv2.imwrite(str(P/f'control-{label}.png'),im);g=im[:,:,1].astype(float);ob=norm(g[ns['Y'],ns['X']].ravel());sol=least_squares(lambda v:norm(ns['values'](v))-ob,truth,x_scale=[.01,.01,.1,.05,.05],max_nfev=1000,ftol=1e-11,xtol=1e-11);v=sol.x;y,p,h,px,pz=v;player=np.array([(px+.3*np.sin(y)*np.cos(p))%1,(pz-.3*np.cos(y)*np.cos(p))%1]);trplayer=np.array([.796530,.795970]);rec={'capture':label,'parameters':v.tolist(),'camera_parameter_error':(v-truth).tolist(),'interpolated_player_phase':player.tolist(),'player_phase_bias':(player-trplayer).tolist(),'train':float(np.corrcoef(ns['values'](v),ob)[0,1]),'heldout':float(np.corrcoef(ns['values'](v,ns['rxt'],ns['ryt']),g[ns['YT'],ns['XT']].ravel())[0,1])};out.append(rec);print(rec,flush=True)
(P/'clear-control-inverse.json').write_text(json.dumps(out,indent=2))
