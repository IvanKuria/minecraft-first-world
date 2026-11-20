"""Run terrace/synth.py unchanged on Windows: only the renderer path points at trender.exe (MSVC build of the same
trender.c). usage: identical to synth.py."""
import sys
from pathlib import Path
T = Path(__file__).resolve().parents[2] / 'terrace'
sys.path.insert(0, str(T))
import subprocess
import synth
_run = subprocess.run


def run(args, *a, **k):
    if args and str(args[0]).endswith('trender'):
        args = [str(args[0]) + '.exe', *args[1:]]
    return _run(args, *a, **k)


synth.subprocess.run = run
if __name__ == '__main__':
    synth.main()
