"""Preserve every source-editor frame from original-channel2011 media."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'timing/end-source-code'
OUT.mkdir(exist_ok=True)
times=json.loads((BASE/'timing/original2011-cadence.json').read_text())['pts_seconds']
cap=cv2.VideoCapture(str(BASE/'footage-sources/wayback-fakeurl.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES,3470)
records=[];frames=[]
for index in range(3470,3500):
    ok,im=cap.read();assert ok
    name=f'frame-{index}.png';cv2.imwrite(str(OUT/name),im)
    records.append({'frame':index,'pts':times[index],'file':name,
                    'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()})
    thumb=cv2.resize(im,(320,180),interpolation=cv2.INTER_AREA)
    cv2.putText(thumb,f'{index} / {times[index]:.3f}',(5,174),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
    frames.append(thumb)
cap.release()
sheet=np.vstack([np.hstack(frames[i:i+5]) for i in range(0,len(frames),5)])
cv2.imwrite(str(OUT/'contact.jpg'),sheet)
(OUT/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
