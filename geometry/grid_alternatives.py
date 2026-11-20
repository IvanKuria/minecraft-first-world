exec(open(__file__.replace('grid_alternatives.py','upright_bundle.py')).read().split('sol=least_squares')[0])
results=[]
for w in [1,2,3]:
 for d in [1,2,3]:
  cliff=np.float64([[-2-w,1,1],[-2,1,1],[-2,1,1+d],[-1,1,1+d]])
  X=np.vstack([cube,cliff]);sol=least_squares(residual,x0,max_nfev=1000,bounds=(np.r_[100,100,[-np.inf]*15],np.r_[800,800,[np.inf]*15]))
  rec={'tongue_width_blocks':w,'tongue_depth_blocks':d,'rms_px':float(np.sqrt(np.mean(sol.fun**2)*2)),'focal_xy':sol.x[:2].tolist()};results.append(rec)
for r in sorted(results,key=lambda r:r['rms_px']):print(r)
(P/'grid-alternatives.json').write_text(json.dumps(results,indent=2))
