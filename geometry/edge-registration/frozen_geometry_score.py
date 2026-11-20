"""Frozen before viewing historical RNG candidate coordinates. No reset height/physics.
Camera/model and surface-identity assumptions are documented in README.md.
"""
from pathlib import Path
import json,numpy as np,cv2,hashlib
P=Path(__file__).parent;Q=P.parent/'phase2'
GROUPS=[('153.050',0,'Z',10,55),('153.050',0,'X',110,620),('153.150',1,'X',20,620),('153.300',2,'X',20,300),('153.500',3,'X',20,280)]
F=180/np.tan(np.deg2rad(35));BIAS_LIMIT=1.0

def measurements():
 out=[]
 for name,i,axis,lo,hi in GROUPS:
  im=cv2.imread(str(Q/f'f-{name}.png'));b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);ys=np.argmax(~sky,axis=0);xs=np.arange(lo,hi,2);valid=(ys[xs]>=3)&(ys[xs]<180);xs=xs[valid]
  out.append({'frame':name,'pose_index':i,'axis':axis,'x':xs.astype(float)+.5,'y':ys[xs].astype(float)})
 return out

DATA=measurements();VARIANTS=json.loads((P/'four-pose-roi-variants.json').read_text())

def line_for_pose(pose,player_x,player_z,k,axis):
 yaw,pitch,h=pose;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]])
 if axis=='X':q=R@np.array([256-(player_x-.3*s*cp),k-h,0.]);direction=R[:,2]
 else:q=R@np.array([0.,k-h,-(player_z+.3*c*cp)]);direction=R[:,0]
 a=F*q[:2]/q[2]+[320,180];vp=F*direction[:2]/direction[2]+[320,180];m=(vp[1]-a[1])/(vp[0]-a[0]);return m,a[1]-m*a[0]

def score(player_x,player_z):
 records=[]
 for variant in VARIANTS:
  params=np.array(variant['parameters']);poses=params[3:].reshape(4,3)
  for k in range(-63,64):
   if k>=min(poses[:,2]):continue
   groups=[]
   for d in DATA:
    m,b=line_for_pose(poses[d['pose_index']],player_x,player_z,k,d['axis']);error=m*d['x']+b-d['y'];bias=float(np.clip(np.mean(error),-BIAS_LIMIT,BIAS_LIMIT));groups.append({'frame':d['frame'],'axis':d['axis'],'raw_rms':float(np.sqrt(np.mean(error**2))),'profiled_bias':bias,'profiled_rms':float(np.sqrt(np.mean((error-bias)**2)))})
   train=float(np.sqrt(np.mean([g['profiled_rms']**2 for g in groups[:4]])));held=groups[4]['profiled_rms'];records.append({'variant':variant['variant'],'integer_edge_minus_grass_height':k,'training_rms_px':train,'heldout_lastframe_rms_px':held,'all_groups_rms_px':float(np.sqrt(np.mean([g['profiled_rms']**2 for g in groups]))),'groups':groups})
 records.sort(key=lambda r:r['training_rms_px'])
 return {'player_xz':[float(player_x),float(player_z)],'selection':'lowest first4group RMS; fifthgroup neverusedfor modelselection','best':records[0],'alternatives':records[1:5]}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--x',type=float);ap.add_argument('--z',type=float);ap.add_argument('--freeze',action='store_true');args=ap.parse_args()
 if args.freeze:
  files=['frozen_geometry_score.py','four-pose-roi-variants.json','clear-corner-measurement.json','four-plane-offset-validation.json'];frozen={'status':'frozen before candidate-coordinate inspection','selection':'integerk +ROIvariant profiled onfirst4groups; last153.520Xedgeheldout','pixel_bias_limit_each_group':BIAS_LIMIT,'groups':[{k:v for k,v in d.items()if k not in ['x','y']}|{'n':len(d['x']),'x_range':[float(d['x'].min()),float(d['x'].max())]}for d in DATA],'files_sha256':{name:hashlib.sha256((P/name).read_bytes()).hexdigest()for name in files},'assumptions':['original70degreeFOV effectivelyuncropped; principalpoint320,180','measuredauthentictextureposes and0.3setback','stationaryhorizontalposition acrossclearframes','same neargrassplane supportedbyheldout featurehomographies','visibleoutlines areworldX256/Z0 outeredges onone commoninteger-heightplane','candidate-to-observedposition linkage remains conditional on rawresetstationarity'],'not_claimed':'no confidenceprobability, no acceptance threshold calibratedon candidates, no recoveredseed'};(P/'frozen-score-manifest.json').write_text(json.dumps(frozen,indent=2));print(json.dumps(frozen,indent=2))
 elif args.x is not None and args.z is not None:print(json.dumps(score(args.x,args.z),indent=2))
 else:ap.error('Provide--freeze or --x/--z')
