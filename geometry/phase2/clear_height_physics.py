from pathlib import Path
import numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent;j=json.loads((P/'joint-clear-grass.json').read_text());t=np.array([153.158,153.313,153.520]);observed=np.array([r['height_above_plane']-.3*np.sin(np.deg2rad(r['yaw_pitch_deg'][1]))for r in j['frames']]);out=[]
def model(v):tr,dy=v[:2];scale=v[2]if len(v)>2 else 1.;n=(t-tr)*60*scale;return 74.72-.25*(n-49*(1-.98**n))+dy*(1-.98**n)/.02
for plane in range(43,65):
 sol=least_squares(lambda v:model(v)-plane-observed,[152.67,0.],bounds=([152.64,-.245],[152.70,.12]));out.append({'planes':[plane]*3,'reset_time':float(sol.x[0]),'initial_yd':float(sol.x[1]),'predicted_relative_player_height':(model(sol.x)-plane).tolist(),'rms_blocks':float(np.sqrt(np.mean(sol.fun**2)))})
ranked=[]
for last in range(60,65):
 for d1 in range(-2,3):
  for d2 in range(-2,3):
   planes=np.array([last+d1,last+d2,last]);
   if np.any(planes>64):continue
   sol=least_squares(lambda v:model(v)-planes-observed,[152.67,0.],bounds=([152.64,-.245],[152.70,.12]));ranked.append({'planes':planes.tolist(),'reset_time':float(sol.x[0]),'initial_yd':float(sol.x[1]),'rms_blocks':float(np.sqrt(np.mean(sol.fun**2)))})
ranked.sort(key=lambda a:a['rms_blocks']);same=sorted(out,key=lambda a:a['rms_blocks']);best=same[0];plane=best['planes'][0];scaled=least_squares(lambda v:model(v)-plane-observed,[152.67,best['initial_yd'],1],bounds=([152.64,-.245,.3],[152.70,.12,2]));res={'times':t.tolist(),'measured_player_height_above_ROI_plane':observed.tolist(),'assumptions':'later reset74.72, unmodifiedfreefall formula,60ticks/video second, one reset tick; tests not truth','same_plane_profiles':out,'best_integer_offset_alternatives':ranked[:15],'free_speed_best_same_plane':{'plane':plane,'reset_time':float(scaled.x[0]),'initial_yd':float(scaled.x[1]),'ticks_per_video_second':float(60*scaled.x[2]),'rms':float(np.sqrt(np.mean(scaled.fun**2)))},'warning':'threeheights cannotidentifyfreeplane/timing/initialvelocity/offsets all atonce; nofloorcollisionmodeled'};print('best same',same[:3],'different',ranked[:3],'scaled',res['free_speed_best_same_plane']);(P/'clear-height-physics.json').write_text(json.dumps(res,indent=2))
