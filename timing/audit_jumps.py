"""Inspect strongest decoded inter-frame changes for overlooked reset events."""
import json
from pathlib import Path
import cv2,numpy as np

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'timing'
data=json.loads((OUT/'original2011-cadence.json').read_text())
mad=np.array(data['consecutive_thumbnail_mad']);pts=data['pts_seconds']
selected=[]
for i in np.argsort(mad)[::-1]:
    frame=int(i)+1
    if frame<50:continue
    if all(abs(frame-old)>12 for old in selected):selected.append(frame)
    if len(selected)==24:break
selected=sorted(selected)
cap=cv2.VideoCapture(str(BASE/'footage-sources/wayback-fakeurl.mp4'))
sheet=np.zeros((len(selected)*115,640,3),np.uint8)
rows=[]
for row,frame in enumerate(selected):
    for col,index in enumerate([frame-1,frame]):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();assert ok
        im=cv2.resize(im,(320,90),interpolation=cv2.INTER_AREA)
        sheet[row*115+25:row*115+115,col*320:(col+1)*320]=im
        cv2.putText(sheet,f'{index} | {pts[index]:.3f}s',(col*320+5,row*115+17),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    rows.append({'frame':frame,'pts':pts[frame],'mean_absolute_change':float(mad[frame-1])})
cap.release()
cv2.imwrite(str(OUT/'largest-jumps-contact.jpg'),sheet)
(OUT/'largest-jumps.json').write_text(json.dumps({'method':'Top24 thumbnail MAD peaks,12frame separation; excludesstartupfirst50frames; not an exhaustive reset detector','events':rows},indent=2)+'\n')
print(json.dumps(rows),flush=True)
