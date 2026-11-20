"""Materialize positions for independently frozen geometry scorers."""
import json
from pathlib import Path
from benchmark_spawn_recovery import advance,coordinate

BASE=Path(__file__).resolve().parent;FOLDER=BASE/'conditional-two-reset'
source=json.loads((FOLDER/'opening-holdout-results.json').read_text())
rows=[]
for item in source['survivors']:
    state=item['state_before_first_observed_reset']
    positions=[coordinate(advance(state,2*i+1)) for i in range(4)]
    rows.append({'state_before_first_observed_reset':state,
                 'opening':[item['opening_x'],item['opening_z']],
                 'first_reset':positions[:2],'second_reset':positions[2:]})
(FOLDER/'candidate-positions.json').write_text(json.dumps({'status':source['status'],
    'constraint_history':'two-reset scenario then independently frozen opening phase holdout',
    'candidates':rows},indent=2)+'\n')
print(json.dumps({'candidates':len(rows),'authenticated':False}))
