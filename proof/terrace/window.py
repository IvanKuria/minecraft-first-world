"""T windows from the fine-T true-surface grid: where does each variant look like the footage near field?
Footage-implied truth (sweep calibration ratio 0.45-0.65): shallow depressions (-1..-3 below the top/modal level)
~0.35-0.65 of near-field columns, no columns above the modal level, modal level = top level.
Global criterion used here (frozen before looking at batch-3 synthetic results): mode is the highest level present
(f+ < 0.02) AND shallow fraction in [0.25, 0.60]."""
import json, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'grid2-surf.jsonl')]
by = defaultdict(list)
for r in rows: by[r['tag']].append(r)
print('%-34s %6s %5s %-28s %-8s %-8s %-9s' % ('variant', 'Tfill', 'mode', 'window T/V (shallow .25-.60, f+<.02)', 'shallow', 'bottom', 'edge=mode'))
for tag, rs in by.items():
    rs.sort(key=lambda r: r['T_over_V'])
    final_mode = rs[-1]['mode']
    # first T where unset_left stops changing (fill time, coarse)
    tfill = next((r['T_over_V'] for r in rs if r['unset_left'] == rs[-1]['unset_left']), None)
    win = []
    for r in rs:
        sh = r['f-1'] + r['f-2'] + r['f-3']; fp = r['f+1'] + r['f+2'] + r['f+3up']
        if fp < .02 and .25 <= sh <= .60: win.append((r['T_over_V'], round(sh, 2), r['bottom_solid'], r.get('edge_col_top_eq_mode')))
    ws = '%s-%s' % (win[0][0], win[-1][0]) if win else '-'
    sh = '/'.join(str(w[1]) for w in win[:4])
    bt = '/'.join(str(w[2]) for w in win[:2])
    ed = '/'.join(str(w[3]) for w in win[:2])
    print('%-34s %6s %5d %-28s %-8s %-8s %-9s' % (tag, tfill, final_mode, ws, sh, bt, ed))
