from pathlib import Path
import cv2,numpy as np,json
from scipy.optimize import least_squares
P=Path(__file__).parent
# Import definitions only, preserving image fit frozen independently.
ns={'__file__':str(P/'fit_wall.py')};exec((P/'fit_wall.py').read_text().split('angles=json.loads')[0],ns)
fit=max(json.loads((P/'wall-fit.json').read_text()),key=lambda r:r['train']);v=np.array(fit['parameters']);ya,pi,d,u,w=v;s,c=np.sin(ya),np.cos(ya);sp,cp=np.sin(pi),np.cos(pi);R=np.array([[c,0,s],[-s*sp,-cp,c*sp],[s*cp,-sp,-c*cp]]);F=ns['f'];im=cv2.imread(str(P/'boundary-164.png'));b,g,r=cv2.split(im.astype(float));sky=(b>160)&(g>120)&(b-g>15)&(g-r>25);xs=np.arange(510,630);ys=np.argmax(sky,axis=0)[xs];valid=ys>30;xs=xs[valid];ys=ys[valid];xx=xs+.5;yy=ys.astype(float);train=xs%2==0
line=least_squares(lambda a:a[0]*xx[train]+a[1]-yy[train],[-1,850],loss='soft_l1',f_scale=.7);m,inter=line.x;rays=R.T@np.array([(xx-320)/F,(m*xx+inter-180)/F,np.ones(len(xx))]);heights=[]
for bias in [-1,0,1]:
 brays=R.T@np.array([(xx-320)/F,(m*xx+inter+bias-180)/F,np.ones(len(xx))]);heights.append({'edge_shift_px':bias,'camera_y_if_bottom_zero':float(np.median(d*brays[1]/brays[2]))})
cy=heights[1]['camera_y_if_bottom_zero'];phase_res=float(((-cy)-w+.5)%1-.5)
# Entire selected outer-face footprint: fixed ROI excludes sky and severely minified right edge.
Y,X=np.mgrid[15:345,15:420];U,V=ns['coords'](v,X.ravel(),Y.ravel());pred=ns['predict'](v,X.ravel(),Y.ravel());obs=im[Y,X,1].ravel().astype(float);ids=np.c_[np.floor(U).astype(int),np.floor(V).astype(int)];cells=[]
for pair in np.unique(ids,axis=0):
 mask=np.all(ids==pair,axis=1);n=int(mask.sum())
 if n<100:continue
 corr=float(np.corrcoef(obs[mask],pred[mask])[0,1]);alts=[]
 for recess in [1,2,3,4]:
  vv=v.copy();vv[2]+=recess;alt=ns['predict'](vv,X.ravel()[mask],Y.ravel()[mask]);alts.append(float(np.corrcoef(obs[mask],alt)[0,1]))
 cells.append({'local_tile_u_v':pair.tolist(),'pixel_samples':n,'front_plane_correlation':corr,'recessed_parallel_plane_correlations':alts})
ysim=m*xx+inter;errs=ysim-yy
# Vanishing points expected for horizontal vs vertical plane edge.
vpH=F*R[:2,0]/R[2,0]+[320,180];vpV=F*R[:2,1]/R[2,1]+[320,180];anchor=np.array([np.mean(xx),np.mean(yy)]);vertical_m=(vpV[1]-anchor[1])/(vpV[0]-anchor[0]);horizontal_m=(vpH[1]-anchor[1])/(vpH[0]-anchor[0]);out={'status':'actual source wall geometry; no RNG candidate read','request_seek_seconds':164.,'camera_pose_local_wallZzero':v.tolist(),'bottom_edge_line':[float(m),float(inter)],'line_train_rms':float(np.sqrt(np.mean(errs[train]**2))),'line_heldout_columns_rms':float(np.sqrt(np.mean(errs[~train]**2))),'height_variants':heights,'camera_y_mod1_from_texture':float((-w)%1),'texture_vs_edge_phase_residual':phase_res,'expected_horizontal_edge_slope':float(horizontal_m),'expected_vertical_edge_slope':float(vertical_m),'vertical_edge_rms':float(np.sqrt(np.mean((vertical_m*(xx-anchor[0])+anchor[1]-yy)**2))),'observed_face_cells':cells,'summary':{'n_cells_at_least100px':len(cells),'corr_percentiles':np.percentile([a['front_plane_correlation']for a in cells],[0,5,50,95,100]).tolist(),'cells_corr_gt_08':sum(a['front_plane_correlation']>.8 for a in cells),'cells_front_exceeds_every_1to4_recess':sum(a['front_plane_correlation']>max(a['recessed_parallel_plane_correlations'])for a in cells)},'limits':['World origin/absolute horizontal axes remain rotationally ambiguous for stone alone.','BottomYzero association follows visible outer lower edge plus later complete bottom, not texture phase alone.','Pixel-face counts are selected visible patches, not the complete boundary occupancy.','Recess tests cover parallel faces1..4blocks back, not arbitrary hidden surfaces.']};(P/'wall-measurement.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:val for k,val in out.items()if k!='observed_face_cells'},indent=2));cv2.polylines(im,[np.int32(np.c_[xs,ysim])],False,(0,0,255),1);cv2.imwrite(str(P/'wall-bottom-overlay.png'),im)
