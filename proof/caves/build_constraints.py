"""Assemble caves/constraints-opening.json and caves/constraints-r1.json from this lane's measurement files.
Coordinates: dx = world_x - floor(X_ref), dz = world_z - floor(Z_ref), y = ABSOLUTE world y (0..63) under the
DEEP-REVIEW model 'plain = top of the world' (plain top solid y=63, top face at y=64).  Each item has a confidence in
[0,1] meaning 'probability the item is right given the stated model', judged by hand from the listed evidence."""
import json, math
from pathlib import Path
P = Path(__file__).parent

def frac_cols(f, half=.3):
    lo, hi = f - half, f + half
    return list(range(math.floor(lo), math.floor(hi - 1e-9) + 1))

# ---------------- opening ----------------
fx, fz = 0.0961, 0.8042                       # opening-texture joint fit (fixed70), stable frames 37-54 (this lane)
bx, bz = frac_cols(fx), frac_cols(fz)
W = json.loads((P / 'wide-48-61.json').read_text())
T = json.loads((P / 'track2-a.json').read_text())
traj = []
for r in T:
    if r.get('ok') and r['n'] <= 47:
        traj.append({'frame': r['n'], 'player_frac': [round(v % 1, 4) for v in r['player']], 'yaw_deg': round(r['yaw_deg'], 2),
                     'pitch_deg': round(r['pitch_deg'], 2), 'cam_height_above_plain': round(r['h'], 3), 'fit_corr': round(r['score'], 3),
                     'source': 'grasstrack2 (local tracker from the frozen frame-44 pose)'})
for r in W:
    if r['n'] <= 56 and r['score'] > .4:
        traj.append({'frame': r['n'], 'player_frac': [round(v, 4) for v in r['player_frac']], 'yaw_deg': round(math.degrees(r['yaw_rad']) % 360, 2),
                     'pitch_deg': round(math.degrees(r['pitch_rad']), 2), 'cam_height_above_plain': round(r['h'], 3), 'fit_corr': round(r['score'], 3),
                     'source': 'widefit (independent full-yaw search per frame)'})
traj.sort(key=lambda t: t['frame'])
for t in traj:
    if t['frame'] >= 55:   # walking: unwrap against the stationary phase (motion < 0.5 block/frame)
        d = [((t['player_frac'][i] - (fx, fz)[i]) + .5) % 1 - .5 for i in range(2)]
        t['displacement_from_spawn_blocks'] = [round(d[0], 3), round(d[1], 3)]
edge = json.loads((P / 'edge-robustness.json').read_text())
opening = {
    'lane': 'CAVES', 'written': '2026-10-15', 'status': 'measured constraints, conditional on the stated model; not a recovery',
    'reference': {'what': 'opening player position (interpolated player = camera + 0.3*forward), stationary 1.914-2.79 s (frames 37-54)',
                  'player_fraction_xz': [fx, fz], 'origin': 'dx=dz=0 is the block column containing the player centre',
                  'y': 'absolute world y; plain top solid y=63 (DEEP-REVIEW fresh-eyes: plain = top of world)'},
    'model_assumptions': ['plain (top level) top face at y=64 is the grass plane the opening texture fits use',
                          'camera 0.62 (0.60-0.68) above that plane at frames 39-54 => eye 64.62, feet 63.00',
                          'rd-132211 Player: bbox 0.6 x 1.8, eye 1.62 above feet, camera 0.3 behind the eye',
                          'no movement between the raw spawn and 1.914 s is NOT established (backlink fails at 1.81 s)'],
    'cells': [], 'column_constraints': [], 'trajectory': traj,
}
for x in bx:
    for z in bz:
        opening['cells'].append({'dx': x, 'y': 63, 'dz': z, 'state': 'air', 'confidence': 0.8,
                                 'why': 'player bbox (feet 63.0-64.8) overlaps this column at y=63'})
opening['column_constraints'].append({'columns_dxdz': [[x, z] for x in bx for z in bz], 'type': 'max_top_equals', 'max_top_solid_y': 62, 'rule': 'max over these columns of the top solid y == 62',
                                      'confidence': 0.75, 'why': 'player rests at feet y=63.0: the highest support under the bbox is a top face at 63'})
SW = json.loads((P / 'sweep-opening.json').read_text())
XV = json.loads((P / 'sweep-opening-crossview.json').read_text())
both = {(x, z): k for x, z, k in XV['A_yaw316']}; bothB = {(x, z): k for x, z, k in XV['B_yaw358_67']}
for x, z, k, r, mg, npx in SW['labels']['0.45/0.15']:
    dist = math.hypot(x + .5 - fx, z + .5 - fz)
    if dist >= 10: continue
    crossed = (x, z) in both and (x, z) in bothB and both[(x, z)] == bothB[(x, z)]
    conf = 0.9 if crossed else (0.8 if dist < 6 else 0.7)
    if k <= -4: conf = round(conf * .75, 2)      # deep floors seen at grazing angles: more error-prone
    opening['column_constraints'].append({'columns_dxdz': [[x, z]], 'type': 'top_equals', 'top_solid_y': 63 + k, 'rule': f'top solid y == {63 + k}; air for y in {64 + k}..63',
        'level_k_rel_plain': k, 'confidence': conf, 'corr': r, 'margin': mg, 'n_px': npx, 'dist_blocks': round(dist, 1),
        'why': 'TERRACE texture plane sweep (sweep.py unchanged) on opening frames 39/41/44 + 48/49/51 (this lane\'s poses)'
               + ('; both view groups agree' if crossed else '')})
opening['absolute_position'] = {
    'method': 'frozen geometry/opening-edge boundary model (frames 39/41 train, 44 held out) with the far plane fixed at the SAME height as the '
              'foreground (plain = top of world); one common row bias <= 1 px',
    'best_player_xz': edge['frozen(all poses, luma, frames39+41)']['best_xz'], 'best_rmse_px': edge['frozen(all poses, luma, frames39+41)']['best_rmse'],
    'region_best_plus_0.1px_by_variant': {k: v['region_best_plus_0.1px'] for k, v in edge.items()},
    'named_rows_rmse_px': {k: v['named_rmse'] for k, v in edge.items()},
    'files': ['edge-rows.json (175 baseline rows)', 'edge-hidden-scenarios.json (4,385 hidden-reset pairs)', 'edge-map-train.npy (256x256, [z,x])',
              'edge-robustness.json', 'edge-height-sensitivity.json'],
    'confidence_note': 'model error budget ~0.1-0.2 px (edge extraction, pose, eye height); not calibrated by a positive control at the opening',
}
opening['not_measured'] = ['startup chunk pop-in registration (see NOTES: the visible reveal is a screen-space left-to-right sweep, not the '
                           'rd-132211 build order at the measured pose)',
                           'walk on the plain after frame 56 (motion blur defeats per-frame texture phase; no unwrapped path)',
                           'hole entry ~17.8-19 s and the caves 19-85 s (not registered)']
(P / 'constraints-opening.json').write_text(json.dumps(opening, indent=1))

# ---------------- first reset ----------------
r1x, r1z = 0.4682, 0.8626
TR = json.loads((P / 'track-r1-a.json').read_text())
fall = [r for r in TR if r.get('ok') and 1668 <= r['n'] <= 1676]
p0 = fall[0]['player']
r1 = {'lane': 'CAVES', 'written': '2026-10-15', 'status': 'measured constraints, conditional on the stated model; not a recovery',
      'reference': {'what': 'first-reset player position (shared stationary fit 85.967-86.277 s, generator/first-reset-texture)',
                    'player_fraction_xz': [r1x, r1z], 'origin': 'dx=dz=0 is the block column containing the player centre at R1',
                    'y': 'absolute world y; plain top solid y=63'},
      'trajectory_fall': [{'frame': r['n'], 'dx_from_R1': round(r['player'][0] - p0[0], 3), 'dz_from_R1': round(r['player'][1] - p0[1], 3),
                           'eye_above_plain': round(r['h'], 2), 'fit_corr': round(r['score'], 3), 'yaw_deg': round(r['yaw_deg'], 1),
                           'pitch_deg': round(r['pitch_deg'], 1)} for r in fall],
      'cells': [], 'column_constraints': []}
land_cols = [[1, 0], [1, -1]]
for (x, z), c in zip(land_cols, (0.7, 0.4)):
    for y in (62, 63):
        r1['cells'].append({'dx': x, 'y': y, 'dz': z, 'state': 'air', 'confidence': c, 'why': 'player body in the trench after landing (fall track extrapolated 2-3 frames; walls above eye at 86.7-86.9 s)'})
    r1['cells'].append({'dx': x, 'y': 61, 'dz': z, 'state': 'solid', 'confidence': round(c * .85, 2), 'why': 'trench floor under the feet (feet 62.0)'})
r1['column_constraints'].append({'columns_dxdz': land_cols, 'type': 'max_top_equals', 'max_top_solid_y': 61, 'rule': 'max over these columns of the top solid y == 61 (player resting in the 2-deep trench)',
                                 'confidence': 0.6, 'agrees_with': 'TERRACE sweep-r1 labels -2 at (1,0) and (1,-1) (independent frames 1662-1668)'})
RS = json.loads((P / 'sweep-r1fall.json').read_text())
TL = {(x, z): k for x, z, k, *_ in json.loads((P.parent / 'terrace/sweep-r1.json').read_text())['labels']}
for x, z, k, r, mg, npx in RS['labels']:
    dist = math.hypot(x + .5 - r1x, z + .5 - r1z)
    if dist >= 10: continue
    st = 'new' if (x, z) not in TL else ('agrees_with_TERRACE' if TL[(x, z)] == k else f'DISAGREES_with_TERRACE({TL[(x, z)]})')
    if st.startswith('agrees'): continue          # already in TERRACE's map; counted in the summary only
    r1['column_constraints'].append({'columns_dxdz': [[x, z]], 'type': 'top_equals', 'top_solid_y': 63 + k, 'rule': f'top solid y == {63 + k}; air for y in {64 + k}..63', 'level_k_rel_plain': k,
        'confidence': 0.75 if st == 'new' else 0.3, 'status_vs_TERRACE': st, 'corr': r, 'margin': mg, 'n_px': npx, 'dist_blocks': round(dist, 1),
        'why': 'TERRACE plane sweep on R1 fall frames 1669-1675 (this lane\'s grass track poses)'})
PF = json.loads((P / 'pitfloor.json').read_text())
r1['component_constraints'] = [{
    'component': 'R1MAP footage-obs-r1 open component of 215 columns, centroid (7.2,-18.9) relative to floor(R1) (identified by viewing '
                 'direction only: the player walks from the landing trench toward yaw 9-11 deg and looks into it at 89.5-90.2 s)',
    'rule': 'contains lit, ROCK-textured column tops at level -4 (top solid y=59; strongest) and -5 (y=58) relative to the grass plane the '
            'player stands on (most likely the plain); its open floor is NOT grass-textured',
    'confidence': 0.6,
    'evidence': [{'frame': p['n'], 'grass_fit_corr': round(p['grass_fit']['score'], 3), 'depth_peak': p['depth_below_grass_plane'], 'rock_corr': round(p['peak_corr'], 3),
                  'other_maxima': p['local_maxima']} for p in PF],
    'controls': 'same pixels: grass tile at any depth <= 0.03; rock tile with the grid shifted by half a block <= 0.02; non-integer depths ~0',
    'standing_plane_registration': {'frame': 1744, 'player_frac': [0.780, 0.628],
        'shift_floor_player_minus_floor_R1': [2, -7], 'alternatives': [[1, -8], [1, -7], [2, -8], [-1, -9]],
        'method': 'open/closed surface map of frames 1737-1744 (R1MAP rules, this lane\'s grass poses) matched to R1MAP footage-obs-r1 by '
                  'integer shift; best pooled agreement 0.869 / open IoU 0.741 over 229 columns; single frames 1743/1744 also pick (2,-7), 1741 picks (-1,-9)',
        'standing_level': 'unmeasured (neighbours (2,-5)=0 TERRACE/this lane, (1,-7)=-1 this lane); floor levels -4/-5 assume the plain'}}]
r1['sweep_summary'] = {'near_field_labels': 100, 'overlap_with_TERRACE_sweep_r1': RS['near_overlap'], 'agree': RS['near_agree']}
r1['not_measured'] = ['walk along the trench toward -z (1678-1735) and the drop into the large hole (~1745-1760; the R1MAP 215-column component lies '
                      'in that direction) are not registered', 'caves 91-152 s not registered']
(P / 'constraints-r1.json').write_text(json.dumps(r1, indent=1))
print('opening cells', len(opening['cells']), 'traj', len(traj), '| r1 cells', len(r1['cells']), 'fall pts', len(fall))
