"""Calibration/evaluation world set for the full-height shell model (one long lgen run per world, dumps kept)."""
import sys, json, time
import lworld as L
V = L.V
FH0 = "init=unset,ntypes={nt},rng=int,pick=xyz,nb={nb},rule=mono,K={K},unsetfill=0,margin=0,cap=none,capy=63,shell=1"
FH1 = "init=unset,ntypes={nt},rng=int,pick=xyz,nb={nb},rule=mono,K={K},unsetfill=0,margin=1,cap=fixed,capy=62,shell=1"
SETS = {
  # tag: (template, nb, K, nt, sweeps_long)
  'fh0s1024': (FH0, 'scan', 1024, 8, 16), 'fh0s4096': (FH0, 'scan', 4096, 8, 12), 'fh0s256': (FH0, 'scan', 256, 8, 24),
  'fh0r4096': (FH0, 'rand', 4096, 8, 48), 'fh0r1024': (FH0, 'rand', 1024, 8, 72),
  'fh1s1024': (FH1, 'scan', 1024, 8, 16), 'fh1r4096': (FH1, 'rand', 4096, 8, 48),
}
STATES = {'a': '0x123456789AB', 'b': '0x5DEECE66D', 'c': '0x2545F4914F6C', 'd': '0x1F123BB5159A'}
if __name__ == '__main__':
    tags = sys.argv[1].split(','); sts = sys.argv[2].split(',')
    for t in tags:
        tpl, nb, K, nt, sw = SETS[t]
        for s in sts:
            t0 = time.time()
            m = L.run(f'{t}{s}', tpl.format(nt=nt, nb=nb, K=K), STATES[s], int(sw * V))
            print(json.dumps({'tag': f'{t}{s}', 'sec': round(time.time() - t0, 1), 'unset_left': m['unset_left'], 'iters': m['iterations_run']}), flush=True)
