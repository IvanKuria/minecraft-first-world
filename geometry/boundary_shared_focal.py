exec(open(__file__.replace('boundary_shared_focal.py','boundary_fit.py')).read().split('results={}')[0])
base=json.loads((P/'boundary-fit.json').read_text());x0=[f]
for n in [18,19,20]:x0.extend(base[str(n)]['v'])
def project(X,v,ff):
 cam=X@R(*v[:2]).T+v[2:5];return cam[:,:2]/cam[:,2,None]*ff+[320,180]
def resid(v):return np.concatenate([(project(X,v[1+i*5:6+i*5],v[0])-np.float64(obs[n]['bottom'])).ravel()for i,n in enumerate([18,19,20])])
fit=least_squares(resid,x0,max_nfev=1000,bounds=(np.r_[150,[-np.inf]*15],np.r_[400,[np.inf]*15]));v=fit.x
out={'focal':v[0],'vertical_fov':float(2*np.rad2deg(np.arctan(180/v[0]))),'rms':float(np.sqrt(np.mean(fit.fun**2)*2)),'cameras':{}}
for i,n in enumerate([18,19,20]):
 vv=v[1+i*5:6+i*5];C=-R(*vv[:2]).T@vv[2:5];out['cameras'][n]=C.tolist()
print(out);(P/'boundary-shared-focal.json').write_text(json.dumps(out,indent=2))
