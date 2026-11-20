"""Compare independently frozen opening-image bounds with prior predictions.

Bounds were supplied by the image researcher before they inspected any RNG
candidate/prediction file. They measure a later stationary position, not an
authenticated raw initial spawn; this check carries the linkage assumption.
"""
import csv,json,hashlib
from pathlib import Path

BASE=Path(__file__).resolve().parent;FOLDER=BASE/'conditional-two-reset'
source=FOLDER/'predicted-opening.csv'
bounds=[[.075,.130],[.785,.835]]
predicate_path=BASE.parent/'generator/opening-texture/frozen-opening-predicate.json'
predicate=json.loads(predicate_path.read_text())
assert bounds==[predicate['x_fraction_interval'],predicate['z_fraction_interval']]
rows=[];total=0
with source.open() as f:
    for row in csv.DictReader(f):
        total+=1
        if all(lo<=float(row[key])<=hi for key,(lo,hi) in zip(['fraction_x','fraction_z'],bounds)):
            rows.append({k:int(v) if k=='state_before_first_observed_reset' else float(v) for k,v in row.items()})
report={
    'status':'Conditional holdout filter, NOT authenticated initial spawn or recovered seed',
    'opening_fractional_intervals':bounds,
    'bounds_frozen_before_filtering':True,
    'bounds_origin':'Independent prototype_research measurements; candidate CSV deliberately not inspected during image fit',
    'image_interval_seconds':[1.914,2.276],
    'earlier_failed_backlink_seconds':1.810,
    'input_predictions':total,'remaining_candidates':len(rows),
    'predictions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'opening_predicate_sha256':hashlib.sha256(predicate_path.read_bytes()).hexdigest(),
    'freeze_provenance':'Numeric bounds supplied in an independent message before filtering; JSON artifact saved afterward',
    'additional_assumptions':['Opening measured stationary coordinates equal raw initial random spawn',
         'No extra RNG consumption between initial spawn and first observed reset',
         'Opening image calibration/camera/height model and working intervals cover the true position'],
    'expected_survivors_if_uniform_independent_fractions':total*(bounds[0][1]-bounds[0][0])*(bounds[1][1]-bounds[1][0]),
    'interpretation':'Many accidental matches are expected. Passing is neither seed authentication nor proof of the reset model.',
    'survivors':rows}
(FOLDER/'opening-holdout-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='survivors'},indent=2))
