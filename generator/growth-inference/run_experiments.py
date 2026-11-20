from pathlib import Path
import subprocess,json,hashlib,time
P=Path(__file__).parent
# Predeclared hypotheses. Exactly the same experimental RNG seed in paired boundary conditions.
cases=[('mono6',4096),('mono6',32768),('mono26',4096),('voter6',2907270)]
results=[]
for mode,seeds in cases:
 for boundary in ['free','fixed']:
  label=f'{mode}-{seeds}-{boundary}';out=P/(label+'.blocks');cmd=[str(P/'growth'),mode,boundary,str(seeds),'80000000','190513',label,str(out)]
  r=json.loads(subprocess.check_output(cmd));r['artifact']=out.name;r['sha256']=hashlib.sha256(out.read_bytes()).hexdigest();results.append(r);print(label, 'occupancy',r['solid']/4194304,'top',r['layers'][-1]/65536,'boundary',r['boundary_solid']/r['boundary_total'],'islands',r['isolated_top_cells_4neighbors'],'seconds',r['seconds'],flush=True)
  (P/'experiments.json').write_text(json.dumps({'status':'hypothetical generators, experimental seeds only','cases':results},indent=2))
