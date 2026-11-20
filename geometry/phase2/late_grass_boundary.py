from pathlib import Path
import numpy as np,cv2,json
P=Path(__file__).parent;v=json.loads((P/'late-grass-fit.json').read_text())[1]['parameters'];yaw,pitch,h,px,pz=v;f=180/np.tan(np.deg2rad(35));s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);im=cv2.imread(str(P/'f-153.300.png'));b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);yy=np.argmax(~sky,axis=0);xs=np.arange(20,360);ys=yy[xs];rx=(xs+.5-320)/f;ry=(ys+.5-180)/f;dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp;delta=-h*np.array([dx/dy,dz/dy]).T
# If boundary skyline is on same horizontal grass plane and its normal is X, inferred cameraX=256-deltaX.
camx=256-delta[:,0];print('C_x acrossedge',np.percentile(camx,[0,5,50,95,100]),'texture camera phase',px%1)
(P/'late-grass-boundary.json').write_text(json.dumps({'status':'Conditional SAME Y for neargrass and distantboundary; test, not acceptedposition','cameraX_percentiles_0_5_50_95_100':np.percentile(camx,[0,5,50,95,100]).tolist(),'camera_phase_x':px%1,'boundaryplane_deltaX_at_givenheight':float(np.median(delta[:,0]))},indent=2))
