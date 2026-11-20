"""Experimental combined silhouette+texture fit; multimodal search, held-out frames."""
from pathlib import Path
import numpy as np,json
from scipy.optimize import minimize,least_squares
P=Path(__file__).parent;ns={'__file__':str(P/'joint_skyline_motion.py')};exec((P/'joint_skyline_motion.py').read_text().split('v0=np.r_')[0],ns);gs={'__file__':str(P/'joint_grass_motion.py')};exec((P/'joint_grass_motion.py').read_text().split('bounds=')[0],gs);train=[0,2,4];test=[1,3];ang=ns['ang'];nom=ns['obs'];pix=gs['pix'];obs=gs['obs'];names=ns['names'];tr0=152.681;fov0=68.51
# v=[global player x,z,resetTime,fov,training yaw/pitch pairs]
def full(v):
 a=ang.copy();a[train]=np.array(v[4:]).reshape(-1,2);return np.r_[v[:4],a.ravel()]
def photo(v,i,held=False):
 vf=full(v);a=vf[4:].reshape(-1,2)-ang;pv=np.r_[v[:4],0,a[train,0],a[train,1]];return pix(pv,i,'grounded_still',gs['xt'],gs['yt'])if held else pix(pv,i,'grounded_still')
def loss(v):
 vf=full(v);sky=np.concatenate([ns['predict'](vf,i)-nom[i]for i in train]);rob=np.mean(np.sqrt(1+sky*sky)-1);return -np.mean([np.dot(gs['normalized'](photo(v,i)),obs[i])for i in train])+.025*rob
bounds=[(221,226),(110,138),(152.64,152.70),(68,70.5)]+[(float(a-.04),float(a+.04))for a in ang[train].ravel()];out=[]
for z in np.arange(115,135,2):
 start=np.r_[223.635,z+.139,tr0,fov0,ang[train].ravel()]
 # Use skyline-only pose for each particular startingglobalposition, then photometricrefinement.
 solang=least_squares(lambda a:np.concatenate([ns['predict'](full(np.r_[start[:4],a]),i)-nom[i]for i in train]),start[4:],loss='soft_l1',f_scale=1);start[4:]=solang.x
 sol=minimize(loss,start,method='L-BFGS-B',bounds=bounds,options={'maxiter':350,'maxls':40,'ftol':1e-11});v=sol.x;vf=full(v);rec={'score':float(-sol.fun),'parameters':v.tolist(),'training_sky_rms':float(np.sqrt(np.mean(np.concatenate([ns['predict'](vf,i)-nom[i]for i in train])**2))),'checks':[]}
 for i in test:
  def at(a):w=vf.copy();w[4+2*i:6+2*i]=a;return w
  pp=least_squares(lambda a:(ns['predict'](at(a),i)-nom[i])[ns['xx']<180],ang[i],loss='soft_l1',f_scale=1);yaw,pitch=pp.x
  # pix helper interpolates corrections from3trainingframes, so overwrite all correctionknots with desiredheld-framecorrection.
  correction=np.array([yaw,pitch])-ang[i];pv=np.r_[v[:4],0,np.repeat(correction[0],3),np.repeat(correction[1],3)];pred=pix(pv,i,'grounded_still');pred2=pix(pv,i,'grounded_still',gs['xt'],gs['yt']);rec['checks'].append({'t':float(ns['times'][i]),'HELDOUT_right_sky_RMS':float(np.sqrt(np.mean((ns['predict'](at(pp.x),i)-nom[i])[ns['xx']>300]**2))),'HELDOUT_texture_main':float(np.dot(gs['normalized'](pred),obs[i])),'HELDOUT_texture_other':float(np.corrcoef(pred2,gs['ims'][i][gs['yt'],gs['xt']].ravel())[0,1])})
 out.append(rec);print(json.dumps(rec),flush=True);(P/'joint-global-texture.json').write_text(json.dumps(sorted(out,key=lambda r:-r['score']),indent=2))
