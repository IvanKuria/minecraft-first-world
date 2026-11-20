from pathlib import Path
import cv2,numpy as np,json,sys
from scipy.optimize import minimize
P=Path(__file__).parent;name=sys.argv[1];im=cv2.imread(str(P/f'boundary-{name}.png'));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);lines=cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1,4);lengths=np.linalg.norm(lines[:,2:]-lines[:,:2],axis=1);lines=lines[lengths>12];lengths=lengths[lengths>12];f=180/np.tan(np.deg2rad(35));K=np.array([[f,0,320],[0,f,180],[0,0,1]]);l=np.cross(np.c_[lines[:,:2],np.ones(len(lines))],np.c_[lines[:,2:],np.ones(len(lines))]);n=l@K;n/=np.linalg.norm(n,axis=1)[:,None];weights=np.minimum(lengths,50)
def R(v):yaw,pitch=v;s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);return np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]])
def loss(v):res=np.min(np.abs(n@R(v)),axis=1);return np.sum(weights*np.minimum(res*res,.015**2))/np.sum(weights)
a=[]
for yaw in np.deg2rad(np.arange(0,90,2)):
 for pitch in np.deg2rad(np.arange(-88,88,2)):a.append((loss([yaw,pitch]),[yaw,pitch]))
a.sort(key=lambda t:t[0]);fits=[minimize(loss,v,method='Nelder-Mead',options={'maxiter':1000})for sc,v in a[:12]];fits.sort(key=lambda s:s.fun);v=fits[0].x;res=np.min(np.abs(n@R(v)),axis=1);groups=np.argmin(np.abs(n@R(v)),axis=1);colors=[(0,0,255),(0,255,0),(255,100,0)]
for line,e,g in zip(lines,res,groups):
 if e<.015:cv2.line(im,tuple(line[:2].astype(int)),tuple(line[2:].astype(int)),colors[g],1)
cv2.imwrite(str(P/f'manhattan-{name}.png'),im);out={'angles_deg':np.rad2deg(v).tolist(),'inliers':int(np.sum(res<.015)),'all_lines':len(lines),'weighted_loss':float(fits[0].fun),'alternatives':[{'angles_deg':np.rad2deg(s.x).tolist(),'loss':float(s.fun)}for s in fits]};print(out);(P/f'manhattan-{name}.json').write_text(json.dumps(out,indent=2))
