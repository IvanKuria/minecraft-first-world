exec(open(__file__.replace('fit_grass_orientations.py','fit_grass_skyline.py')).read().split('# Sweep')[0])
yy,xx=np.mgrid[278:355:3,300:405:3];rx=(xx.ravel()-320)/f;ry=(yy.ravel()-180)/f;obs=im[yy,xx].ravel();obs-=obs.mean();obs/=np.linalg.norm(obs)
# Redefine objective with these samples; rotate/reflect tile for unknownworld orientation/UVconvention.
def score2(v,tx):
 val=values(v,rx,ry,tx);val-=val.mean();return np.dot(val,obs)/(np.linalg.norm(val)+1e-9)
phase=np.array([(a/16,b/16)for a in range(16)for b in range(16)]);yaw,pitch=np.deg2rad([31.56,14.63]);s,c=np.sin(yaw),np.cos(yaw);sp,cp=np.sin(pitch),np.cos(pitch);dx=c*rx-s*sp*ry+s*cp;dy=-cp*ry-sp;dz=s*rx+c*sp*ry-c*cp
results=[]
for flip in [False,True]:
 for rot in range(4):
  tx=np.rot90(np.fliplr(tile) if flip else tile,rot);best=[]
  for h in np.arange(4,20.001,.1):
   u=((phase[:,0,None]-h*dx/dy)*16)%16;v=((phase[:,1,None]-h*dz/dy)*16)%16;u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);a=u-u0;b=v-v0
   val=tx[v0,u0]*(1-a)*(1-b)+tx[v0,(u0+1)%16]*a*(1-b)+tx[(v0+1)%16,u0]*(1-a)*b+tx[(v0+1)%16,(u0+1)%16]*a*b
   val-=val.mean(axis=1)[:,None];sc=val@obs/(np.linalg.norm(val,axis=1)+1e-9);ii=np.argmax(sc);best.append((float(sc[ii]),h,*phase[ii]))
  best.sort(reverse=True)
  refined=[]
  for sc,h,px,pz in best[:4]:
   fit=minimize(lambda v:-score2(v,tx),[yaw,pitch,h,px,pz],method='Nelder-Mead',options={'maxiter':1200});refined.append((float(-fit.fun),fit.x))
  refined.sort(key=lambda a:-a[0]);sc,v=refined[0];rec={'flip':flip,'rot90':rot,'score':sc,'params':v.tolist(),'coarse_peak':best[:4]};results.append(rec);print(rec,flush=True)
(P/'grass-orientations.json').write_text(json.dumps(results,indent=2))
