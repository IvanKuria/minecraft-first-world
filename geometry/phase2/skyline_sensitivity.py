from pathlib import Path
import json,numpy as np
from scipy.optimize import least_squares
P=Path(__file__).parent;out={}
for label,script,filename,tr in [('second','joint_skyline_motion.py','joint-skyline-motion.json',152.67),('first','joint_first_skyline_motion.py','joint-first-skyline-motion.json',85.425)]:
 ns={'__file__':str(P/script)};exec((P/script).read_text().split('v0=np.r_')[0],ns);nom=np.array(ns['obs']);xx=ns['xx'];record=json.loads((P/filename).read_text());base=np.array(record['parameters']);a0=np.r_[base[:2],base[4:]];fov=69
 def unpack(a):return np.r_[a[:2],tr,fov,a[2:]]
 samples=[]
 for dc in [-1,0,1]:
  for dl in [-1,0,1]:
   for dr in [-1,0,1]:
    ns['obs']=[o+dc+np.where(xx<240,dl/240,dr/400)*(xx-240)for o in nom]
    sol=least_squares(lambda a:ns['residual'](unpack(a)),a0,loss='soft_l1',f_scale=1,max_nfev=300);samples.append({'perturbation_px':[dc,dl,dr],'x':float(sol.x[0]),'z':float(sol.x[1]),'rms':float(np.sqrt(np.mean(sol.fun**2)))})
 ns['obs']=list(nom);rois=[]
 for name,mask in [('full',np.ones(len(xx),bool)),('edges_trimmed',(xx>=40)&(xx<=600)),('corner_excluded',(xx<180)|(xx>300)),('middle_sides',((xx>40)&(xx<180))|((xx>300)&(xx<600)))]:
  sol=least_squares(lambda a:np.concatenate([(ns['predict'](unpack(a),i)-o)[mask]for i,o in enumerate(nom)]),a0,loss='soft_l1',f_scale=1,max_nfev=300);rois.append({'roi':name,'x':float(sol.x[0]),'z':float(sol.x[1]),'rms':float(np.sqrt(np.mean(sol.fun**2)))})
 # Hold out odd frames. Fit their orientation using left edge only; predict right edge.
 train=list(range(0,len(nom),2));held=list(range(1,len(nom),2));t0=np.r_[base[:2],base[4:].reshape(-1,2)[train].ravel()]
 def trainfull(a):
  v=base.copy();v[:4]=[a[0],a[1],tr,fov];angles=v[4:].reshape(-1,2);angles[train]=a[2:].reshape(-1,2);return v
 fit=least_squares(lambda a:np.concatenate([ns['predict'](trainfull(a),i)-nom[i]for i in train]),t0,loss='soft_l1',f_scale=1);v=trainfull(fit.x);checks=[]
 for i in held:
  def withangle(a):w=v.copy();w[4+2*i:6+2*i]=a;return w
  pose=least_squares(lambda a:(ns['predict'](withangle(a),i)-nom[i])[xx<180],base[4+2*i:6+2*i],loss='soft_l1',f_scale=1);err=ns['predict'](withangle(pose.x),i)-nom[i];checks.append({'time':float(ns['times'][i]),'training_left_edge_rms':float(np.sqrt(np.mean(err[xx<180]**2))),'HELDOUT_right_edge_rms':float(np.sqrt(np.mean(err[xx>300]**2)))})
 out[label]={'fixed_fov':fov,'fixed_reset_time':tr,'perturbations_are':'bounded +/-1px synthetic edge shifts/slope endpoint changes, not confidence intervals','common_edge_perturbations':samples,'ROI_variants':rois,'heldout_frames_right_edges':checks,'heldout_fit_global_xz':fit.x[:2].tolist()};print(label,'noise x/z',np.min([[a['x'],a['z']]for a in samples],axis=0),np.max([[a['x'],a['z']]for a in samples],axis=0),'ROI',rois,'heldout',checks,flush=True)
(P/'skyline-sensitivity.json').write_text(json.dumps(out,indent=2))
