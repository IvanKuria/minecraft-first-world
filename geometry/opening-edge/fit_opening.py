"""Independent skyline fit. Reads texture poses and images, never RNG candidates."""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from PIL import Image, ImageDraw

P=Path(__file__).parent
SOURCE=P.parents[1]/'generator/opening-texture'
poses=json.loads((P/'pose-snapshot.json').read_text())
pose=max((r for r in poses if r['mode']=='fixed70'),key=lambda r:r['mean_train'])
F=180/np.tan(np.deg2rad(pose['fov']/2))
xs=np.arange(12,632,8,dtype=float)

def extract(name):
    im=cv2.imread(str(SOURCE/f'f-{name}.png'))
    lum=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(float)
    window=(192,202) if name=='2.00' else (158,172)
    ys=[]
    # Center of strongest sky-to-ground luminance transition. Pixel centers
    # use the same +.5 convention as the supplied texture pose calibration.
    for x in xs.astype(int):
        line=np.median(lum[:,x-3:x+4],axis=1)
        grad=np.maximum(0,line[:-1]-line[1:])
        peak=window[0]+np.argmax(grad[window[0]:window[1]])
        inds=np.arange(max(window[0],peak-1),min(window[1],peak+2))
        weight=grad[inds]
        ys.append(float(np.sum((inds+.5)*weight)/(weight.sum()+1e-10)))
    return np.array(ys)

def boundary(player,frame,delta):
    ya,pi,h,_,_=frame['camera_parameters']
    s,c=np.sin(ya),np.cos(ya);sp,cp=np.sin(pi),np.cos(pi)
    cx=player[0]-.3*s*cp;cz=player[1]+.3*c*cp
    h=h+delta
    rx=(xs+.5-320)/F
    candidates=[]
    for axis,q in [('x',0),('x',256),('z',0),('z',256)]:
        if axis=='x':
            r=( -(cx-q)*sp-h*(c*rx+s*cp))/((cx-q)*cp-h*s*sp)
        else:
            r=( -(cz-q)*sp-h*(s*rx-c*cp))/((cz-q)*cp+h*c*sp)
        dy=-cp*r-sp
        t=-h/dy
        wx=cx+t*(c*rx-s*sp*r+s*cp)
        wz=cz+t*(s*rx+c*sp*r-c*cp)
        good=(t>0)&(wx>=-1e-5)&(wx<=256.00001)&(wz>=-1e-5)&(wz<=256.00001)
        candidates.append(np.where(good,r*F+180-.5,np.nan))
    return np.nanmax(np.array(candidates),axis=0)

if __name__=='__main__':
    obs=[extract(f['frame_name']) for f in pose['frames']]
    fits=[]
    for delta in range(7):
      for bias in np.linspace(-1,1,9):
        def residual(x,ids=(0,1)):
          return np.concatenate([boundary(x,pose['frames'][j],delta)+bias-obs[j] for j in ids])
        best=None
        for start in [[30,35],[80,90],[160,170],[230,230]]:
          fit=least_squares(residual,start,bounds=([.5,.5],[255.5,255.5]),loss='soft_l1',f_scale=.5,max_nfev=150)
          if best is None or np.mean(residual(fit.x)**2)<best[0]:best=(np.mean(residual(fit.x)**2),fit)
        mse,fit=best
        phase=np.array(pose['player_phase']);snapped=np.round(fit.x-phase)+phase
        allr=residual(fit.x,(0,1,2));hold=residual(fit.x,(2,))
        fits.append({'plane_below_foreground_blocks':delta,'common_pixel_bias':float(bias),'player_xz':fit.x.tolist(),'train_rmse_px':float(np.sqrt(mse)),'heldout_rmse_px':float(np.sqrt(np.mean(hold**2))),'frame_mean_residual_px':[float(np.mean(residual(fit.x,(j,)))) for j in range(3)],'phase_snapped_xz':snapped.tolist(),'phase_snapped_train_rmse_px':float(np.sqrt(np.mean(residual(snapped)**2))),'phase_snapped_heldout_rmse_px':float(np.sqrt(np.mean(residual(snapped,(2,))**2))),'at_bounds':bool(np.any(fit.x<.51)|np.any(fit.x>255.49))})
    fits.sort(key=lambda r:r['train_rmse_px'])
    result={'input_pose':pose,'frame_names':[f['frame_name'] for f in pose['frames']],'sample_x':xs.tolist(),'observed_edge_rows':[o.tolist() for o in obs],'fit_frames':[0,1],'heldout_frame':2,'fits':fits,'edge_method':'median luminance across 7 columns; weighted center of peak downward gradient and its two neighboring gradients'}
    (P/'fit-results.json').write_text(json.dumps(result,indent=2)+'\n')
    for delta in range(7):
     r=min((r for r in fits if r['plane_below_foreground_blocks']==delta),key=lambda r:r['train_rmse_px']);print(r,flush=True)
    best=fits[0]
    for j,fr in enumerate(pose['frames']):
     im=Image.open(SOURCE/f"f-{fr['frame_name']}.png").convert('RGB');d=ImageDraw.Draw(im)
     pred=boundary(best['player_xz'],fr,best['plane_below_foreground_blocks'])+best['common_pixel_bias']
     for x,y,z in zip(xs,obs[j],pred):d.ellipse((x-1,y-1,x+1,y+1),fill='red');d.ellipse((x-1,z-1,x+1,z+1),fill='yellow')
     im.crop((0,185 if j==0 else 152,640,212 if j==0 else 179)).resize((1280,108)).save(P/f"overlay-{fr['frame_name']}.png")
