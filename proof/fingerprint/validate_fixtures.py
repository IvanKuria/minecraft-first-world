"""Validate the fingerprint pipeline on Java-generated JDK6-style fixtures.

For every fixture: (1) the reflected state reproduces Player.resetPos x/z with
the runtime lane's coordinate() convention (so 'state before opening x draw'
means the same thing in both lanes); (2) fpscan, given only the state and the
search ranges, returns the true (N, k, nanos, qpc) among its passes;
(3) false passes are counted to measure the empirical null rate.
"""
import json, struct, subprocess, sys, tempfile, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
A, C, M = 0x5DEECE66D, 11, 1 << 48
U = 8682522807148012

def coordinate(s):
    s2 = (A * s + C) % M
    d = ((s >> 22) * 2**27 + (s2 >> 21)) / 2**53
    return struct.unpack('f', struct.pack('f', d))[0] * 256

def adv(s, n):
    for _ in range(n): s = (A * s + C) % M
    return s

def main():
    fixtures = [json.loads(l) for l in (HERE / 'fixtures.jsonl').read_text().splitlines()]
    out = []
    nmin, nmax = 1000, 200000
    tmin, tmax = 60 * 10**9, 20 * 3600 * 10**9
    for fx in fixtures:
        s = fx['state_open']
        x = coordinate(adv(s, 1)); z = coordinate(adv(s, 3))  # runtime convention: coordinate() takes the state after the draw's first LCG step
        coord_ok = (struct.pack('f', x) == struct.pack('f', fx['x']) and struct.pack('f', z) == struct.pack('f', fx['z']))
        with tempfile.TemporaryDirectory() as td:
            sf = Path(td) / 's.txt'; sf.write_text(f"{s}\n")
            nf = Path(td) / 'n.txt'; nf.write_text(f"{nmin} {nmax}\n")
            r = subprocess.run([str(HERE / 'build/fpscan'), str(sf), str(nf), '1', '8', str(fx['f']),
                                str(tmin), str(tmax), str(U), '1000000'], capture_output=True, text=True, check=True)
        passes = [l.split() for l in r.stdout.splitlines() if l.startswith('PASS')]
        summ = r.stdout.splitlines()[-1]
        truth = any(int(p[2]) == fx['N'] and int(p[3]) == fx['k'] and int(p[4]) == fx['nanos'] for p in passes)
        trials = int(summ.split('trials=')[1].split()[0])
        out.append({'f': fx['f'], 'N': fx['N'], 'k': fx['k'], 'coord_ok': coord_ok, 'truth_recovered': truth,
                    'passes': len(passes), 'false_passes': len(passes) - (1 if truth else 0), 'trials': trials})
        print(json.dumps(out[-1]), flush=True)
    byf = {}
    for o in out:
        b = byf.setdefault(o['f'], {'fixtures': 0, 'truth_recovered': 0, 'coord_ok': 0, 'false_passes': 0, 'trials': 0})
        b['fixtures'] += 1; b['truth_recovered'] += o['truth_recovered']; b['coord_ok'] += o['coord_ok']
        b['false_passes'] += o['false_passes']; b['trials'] += o['trials']
    for f, b in byf.items():
        b['empirical_false_pass_rate_per_trial'] = b['false_passes'] / b['trials']
        b['analytic_rate_per_trial'] = min(1, f / 1e9) * (tmax - tmin) / M
    (HERE / 'fixture-validation.json').write_text(json.dumps({
        'search': {'N_range': [nmin, nmax], 'k_range': [1, 8], 'tmin_ns': tmin, 'tmax_ns': tmax},
        'per_frequency': byf, 'fixtures': out}, indent=2) + '\n')
    print(json.dumps(byf, indent=2))

if __name__ == '__main__':
    main()
