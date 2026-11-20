from pathlib import Path
import numpy as np,json
P=Path(__file__).parent;d=json.loads((P/'edge-measurements.json').read_text());px=d['player_phase'][0];f=180/np.tan(np.deg2rad(35));frames=d['frames']
def predline(rec,Px,k):
 yaw,pitch,h=rec['yaw_pitch_height'];s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]]);Cx=Px-.3*s*cp;q=R@np.array([256-Cx,k-h,0.]);a=f*q[:2]/q[2]+[320,180];vp=f*R[:2,2]/R[2,2]+[320,180];m=(vp[1]-a[1])/(vp[0]-a[0]);return m,a[1]-m*a[0]
def err(rec,Px,k):
 m,b=predline(rec,Px,k);x=np.linspace(*rec['skyline_xrange'],30);mo,bo=rec['skyline_slope_intercept'];return np.sqrt(np.mean((m*(x+.5)+b-(mo*x+bo+.5))**2))
common=[];independent=[]
for n in range(256):
 X=n+px
 for k in range(-63,64):
  if k>=min(r['yaw_pitch_height'][2]for r in frames):continue
  errors=[float(err(r,X,k))for r in frames];common.append({'playerX':X,'k_common':k,'training_twoframe_RMS':float(np.sqrt(np.mean(np.array(errors[:2])**2))),'heldout_thirdframe_RMS':errors[2],'all_RMS':float(np.sqrt(np.mean(np.array(errors)**2))),'errors':errors})
 bests=[]
 for r in frames:
  opts=[(err(r,X,k),k)for k in range(-63,64)if k<r['yaw_pitch_height'][2]];e,k=min(opts);bests.append((e,k))
 ks=[k for e,k in bests];blo=max(0,max(ks));bhi=min(64,min(k+64 for k in ks))
 if blo<=bhi:independent.append({'playerX':X,'height_offsets':ks,'perframe_errors':[float(e)for e,k in bests],'all_RMS':float(np.sqrt(np.mean([e*e for e,k in bests]))),'feasible_outer_y':[blo,bhi]})
common.sort(key=lambda r:r['training_twoframe_RMS']);independent.sort(key=lambda r:r['all_RMS']);out={'conventions':'k=outeredgeY-neargrassY; noresetheight/gravity; fixed70 measuredtexturepose; commonphase assumedstatic','common_plane_ranked_by_first_two_frames':common[:30],'independent_integer_planes_ranked_all_frames':independent[:30]};(P/'integer-edge-search.json').write_text(json.dumps(out,indent=2));print('COMMON',common[:6]);print('FREEPLANE',independent[:6])
