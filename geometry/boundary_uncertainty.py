exec(open(__file__.replace('boundary_uncertainty.py','boundary_fit.py')).read().split('results={}')[0])
base=json.loads((P/'boundary-fit.json').read_text());rng=np.random.default_rng(20260926);out={}
for n in [18,19,20]:
 v0=np.float64(base[str(n)]['v']);uv=np.float64(obs[n]['bottom']);samples=[]
 for i in range(200):
  # Measurement-noise-only experiment; excludes unknown lens/capture distortion.
  jitter=uv+rng.normal(0,1.5,uv.shape)
  fit=least_squares(lambda v:(proj(X,v)-jitter).ravel(),v0,max_nfev=100)
  C=-R(*fit.x[:2]).T@fit.x[2:5];samples.append(C)
 a=np.array(samples);out[n]={'noise_sigma_px':1.5,'n':200,'camera_p025':np.percentile(a,2.5,axis=0).tolist(),'camera_p975':np.percentile(a,97.5,axis=0).tolist(),'camera_std':np.std(a,axis=0).tolist()};print(n,out[n])
(P/'boundary-uncertainty.json').write_text(json.dumps(out,indent=2))
# Recover exact decodedsourceframe of ffmpeg fps samples.
cap=cv2.VideoCapture(str(P.parent/'cave-game-tech-test.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);times={}
for n in [18,19,20]:
 target=cv2.imread(str(P/'boundary'/f'f{n:03}.png'));best=(1e20,None)
 for idx in range(int((158+n-1-.1)*fps),int((158+n)*fps)+1):
  cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read()
  if ok:
   err=float(np.mean((im.astype(float)-target.astype(float))**2))
   if err<best[0]:best=(err,idx)
 times[n]={'frame_index':best[1],'seconds':best[1]/fps,'mse':best[0]};print('match',n,times[n])
(P/'boundary-frame-times.json').write_text(json.dumps(times,indent=2))
