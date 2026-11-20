"""Measure decoded frame cadence and correspondence; no game-tick inference."""
import json, subprocess
from pathlib import Path
import cv2
import numpy as np

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'timing'

def read_capture(path):
    probe=json.loads(subprocess.check_output([
        'ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time,pkt_duration_time',
        '-of','json',str(path)]))
    pts=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']])
    cap=cv2.VideoCapture(str(path)); frames=[]
    while True:
        ok,im=cap.read()
        if not ok: break
        frames.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(160,90),
                                 interpolation=cv2.INTER_AREA))
    cap.release()
    assert len(frames)==len(pts),(path,len(frames),len(pts))
    frames=np.array(frames)
    mad=np.abs(np.diff(frames.astype(np.int16),axis=0)).mean(axis=(1,2))
    diffs,counts=np.unique(np.round(np.diff(pts),6),return_counts=True)
    return pts,frames,{'file':str(path.relative_to(BASE)), 'decoded_frames':len(pts),
        'first_pts':float(pts[0]),'last_pts':float(pts[-1]),
        'frame_interval_histogram':dict(zip(map(str,diffs),map(int,counts))),
        'consecutive_thumbnail_mad_percentiles':np.percentile(mad,[0,1,5,50,95,100]).tolist(),
        'near_identical_frame_pairs':[{'first':int(i),'second':int(i+1),
             'second_pts':float(pts[i+1]),'mean_absolute_difference':float(mad[i])}
             for i in np.where(mad<.25)[0]],
        'pts_seconds':pts.tolist(),'consecutive_thumbnail_mad':mad.tolist()}

def main():
    t0,f0,m0=read_capture(BASE/'footage-sources/wayback-fakeurl.mp4')
    t1,f1,m1=read_capture(BASE/'cave-game-tech-test.mp4')
    (OUT/'original2011-cadence.json').write_text(json.dumps(m0,indent=2)+'\n')
    (OUT/'modern360-cadence.json').write_text(json.dumps(m1,indent=2)+'\n')
    matches=[]
    # Compare spatial gradients to reduce influence of encoding color shifts.
    grad0=np.diff(f0.astype(np.float32),axis=2)
    grad1=np.diff(f1.astype(np.float32),axis=2)
    for i,t in enumerate(t0):
        if i%5 and not (84.8<t<86 or 152.2<t<153.5 or 174<t<178): continue
        candidates=np.where(abs(t1-t)<.5)[0]
        distances=np.mean(abs(grad1[candidates]-grad0[i]),axis=(1,2))
        order=np.argsort(distances)
        j=int(candidates[order[0]])
        runner=float(distances[order[1]]) if len(order)>1 else None
        matches.append({'original_index':i,'original_pts':float(t),'modern_index':j,
            'modern_pts':float(t1[j]),'modern_minus_original_seconds':float(t1[j]-t),
            'gradient_mad':float(distances[order[0]]),'runner_up_mad':runner,
            'original_thumbnail_std':float(np.std(f0[i]))})
    confident=[r for r in matches if r['original_thumbnail_std']>10
               and r['runner_up_mad']>r['gradient_mad']*1.3
               and r['gradient_mad']<3]
    result={'method':'Nearest spatial-gradient image within +/-0.5s; matches are image correspondence, not original recording time',
            'all_matches':matches,'filtered_matches':confident,
            'caution':'Encoding timestamps do not authenticate simulation time, capture cadence or ticks between frames'}
    (OUT/'cross-copy-correspondence.json').write_text(json.dumps(result,indent=2)+'\n')
    summary={'original_decoded_frames':len(t0),'modern_decoded_frames':len(t1),
             'original_frame_intervals':m0['frame_interval_histogram'],
             'modern_frame_intervals':m1['frame_interval_histogram'],
             'filtered_correspondences':len(confident),
             'offset_range_seconds':[min(r['modern_minus_original_seconds'] for r in confident),
                                     max(r['modern_minus_original_seconds'] for r in confident)],
             'original_near_identical_pairs':len(m0['near_identical_frame_pairs']),
             'modern_near_identical_pairs':len(m1['near_identical_frame_pairs'])}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
