exec(open(__file__.replace('boundary_physics_calibration.py','boundary_fit.py')).read().split('results={}')[0])
base=json.loads((P/'boundary-fit.json').read_text());tm=json.loads((P/'boundary-frame-times.json').read_text());ns=[18,19,20];t0=tm['18']['seconds'];start=[f,145,283,-113.5]
for n in ns:start.extend(np.deg2rad(base[str(n)]['yaw_pitch_degrees']))
def pred(v,i,n):
 ff,cx,cz,y0=v[:4];rot=R(*v[4+i*2:6+i*2]);C=np.array([cx,y0-15*(tm[str(n)]['seconds']-t0),cz]);cam=(X-C)@rot.T
 return cam[:,:2]/cam[:,2,None]*ff+[320,180]
def fun(v):return np.concatenate([(pred(v,i,n)-np.float64(obs[n]['bottom'])).ravel()for i,n in enumerate(ns)])
fit=least_squares(fun,start,max_nfev=2000);v=fit.x
out={'assumptions':['constant horizontal camera location','exact terminalfall15blocks/second fromlaterbuild','frame timestamps correctlyscale originaltime','worldfootprint256x256','uprightcenteredpinholecamera'],'focal_px':v[0],'fovY_deg':float(2*np.rad2deg(np.arctan(180/v[0]))),'x_z_y_at_first':[v[1],v[2],v[3]],'rms_px':float(np.sqrt(np.mean(fit.fun**2)*2)),'parameters':v.tolist()}
print(out);(P/'boundary-physics-calibration.json').write_text(json.dumps(out,indent=2))
