"""Fixed player phase, inherited clear-frame photometric experiment.

No candidate is loaded by this module. Method copies the pre-candidate
phase2/joint_clear_grass.py: same ROIs, sampler, loss, initialization, pose
bounds and stopping criteria; player phase and FOV are now fixed inputs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import cv2
from scipy.optimize import least_squares

P = Path(__file__).parent
Q = P.parent / 'phase2'
SETTINGS = [
    ('153.150', 'grass-153.150-fit.json', 0, (30,220,220,331), (30,335,220,355)),
    ('153.300', 'late-grass-fit.json', 1, (20,205,210,331), (20,335,200,355)),
    ('153.500', 'grass-153.500-fit.json', 0, (20,205,210,331), (20,335,200,355)),
]
TILE = cv2.imread(str(Q / 'terrain.png'))[:16,:16,1].astype(float)
DATA = []
for name, file, index, box, heldout in SETTINGS:
    initial = np.array(json.loads((Q / file).read_text())[index]['parameters'])[:3]
    im = cv2.imread(str(Q / f'f-{name}.png'))[:,:,1].astype(float)
    x0,y0,x1,y1 = box
    Y,X = np.mgrid[y0:y1:3,x0:x1:3]
    x0,y0,x1,y1 = heldout
    YT,XT = np.mgrid[y0:y1,x0:x1]
    DATA.append(dict(name=name,initial=initial,im=im,X=X,Y=Y,XT=XT,YT=YT))

def normalize(a):
    return (a-a.mean())/(a.std()+1e-9)

OBS = [normalize(d['im'][d['Y'],d['X']].ravel()) for d in DATA]
INITIAL = np.concatenate([d['initial'] for d in DATA])
LOW = INITIAL - np.tile([.05,.05,1.],3)
HIGH = INITIAL + np.tile([.05,.05,1.],3)

def prediction(pose, phase, i, held=False):
    px,pz = phase
    yaw,pitch,h = pose
    d = DATA[i]
    X,Y = (d['XT'],d['YT']) if held else (d['X'],d['Y'])
    f = 180/np.tan(np.deg2rad(35))
    x=(X.ravel()+.5-320)/f
    y=(Y.ravel()+.5-180)/f
    s,c=np.sin(yaw),np.cos(yaw)
    sp,cp=np.sin(pitch),np.cos(pitch)
    dx=c*x-s*sp*y+s*cp
    dy=-cp*y-sp
    dz=s*x+c*sp*y-c*cp
    cx=px-.3*s*cp
    cz=pz+.3*c*cp
    u=(((cx-h*dx/dy)%1)*15.984-.5)%16
    w=(((cz-h*dz/dy)%1)*15.984-.5)%16
    u0=np.floor(u).astype(int)
    w0=np.floor(w).astype(int)
    a=u-u0
    b=w-w0
    return (TILE[w0,u0]*(1-a)*(1-b)
            +TILE[w0,(u0+1)%16]*a*(1-b)
            +TILE[(w0+1)%16,u0]*(1-a)*b
            +TILE[(w0+1)%16,(u0+1)%16]*a*b)

def score(player_x, player_z):
    phase=np.array([player_x,player_z])%1
    def residual(v):
        return np.concatenate([
            (normalize(prediction(v[3*i:3*i+3],phase,i))-OBS[i])/np.sqrt(len(OBS[i]))
            for i in range(3)])
    sol=least_squares(residual,INITIAL.copy(),bounds=(LOW,HIGH),
        x_scale=np.tile([.01,.01,.1],3),max_nfev=700,ftol=1e-10,xtol=1e-10)
    frames=[]
    for i,d in enumerate(DATA):
        pose=sol.x[3*i:3*i+3]
        frames.append(dict(frame=d['name'],yaw_pitch_deg=np.rad2deg(pose[:2]).tolist(),
            height_above_plane=float(pose[2]),
            train=float(np.corrcoef(prediction(pose,phase,i),OBS[i])[0,1]),
            heldout=float(np.corrcoef(prediction(pose,phase,i,True),d['im'][d['YT'],d['XT']].ravel())[0,1])))
    return dict(player_phase=phase.tolist(),fov=70.0,camera_setback=.3,
        training_cost=float(sol.cost),mean_training_correlation=float(np.mean([r['train'] for r in frames])),
        mean_heldout_correlation=float(np.mean([r['heldout'] for r in frames])),
        optimizer_success=bool(sol.success),optimizer_message=sol.message,nfev=int(sol.nfev),
        active_bounds=sol.active_mask.tolist(),parameters=sol.x.tolist(),frames=frames,
        interpretation='Conditional photometric ranking only; local optimization, no calibrated acceptance cutoff.')

def freeze():
    files=[Path(__file__),Q/'joint_clear_grass.py',Q/'terrain.png']
    for name,file,_,_,_ in SETTINGS:
        files.extend([Q/file,Q/f'f-{name}.png'])
    out=dict(status='Frozen before any candidate photometric score; edge-ranked candidate coordinates were previously inspected.',
        inherited_method='Pre-candidate phase2/joint_clear_grass.py. Identical training/heldout ROIs, pixel-center sampler, normalized residual, pose initializations/bounds and optimizer tolerances.',
        deliberate_changes='Player fractional X/Z fixed to caller input; FOV fixed70. Optimize only nine independent yaw/pitch/height parameters.',
        settings=SETTINGS,training_stride=3,heldout_stride=1,loss='linear',max_nfev=700,
        selection='Training loss only. Heldout pixels never used to fit poses.',
        files_sha256={str(f.relative_to(P.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        limits=['Camera capture/filter/compression systematics remain; approximate bilinear sampler is not exact nearest original rendering.',
                'Actual-render controls showed phase biases of several thousandths; ROI variants moved inferred phases further.',
                'Local optimizer can retain aliases; no multimodal posterior or confidence interval is claimed.',
                'Measured phase belongs to later clear views, not proven identical to raw reset outputs.',
                'No candidate rejection threshold or seed authentication claim.'])
    (P/'frozen-photometric-manifest.json').write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--freeze',action='store_true')
    ap.add_argument('--x',type=float)
    ap.add_argument('--z',type=float)
    args=ap.parse_args()
    if args.freeze: print(json.dumps(freeze(),indent=2))
    elif args.x is not None and args.z is not None: print(json.dumps(score(args.x,args.z),indent=2))
    else: ap.error('Provide --freeze or --x/--z')
