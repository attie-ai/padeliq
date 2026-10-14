"""Manual interval review, independent of detection and automatic re-identification."""
from pathlib import Path
import csv
import json
import math
import shutil
import subprocess
import cv2
import numpy as np
from engine import metadata, frame_at, calibration


def load_tracks(directory):
    by_frame, by_id = {}, {}
    with (Path(directory)/'all_tracks.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            frame, tid = int(row['frame']), int(row['tracker_id'])
            box = [float(row[k]) for k in ('x1','y1','x2','y2')]
            by_frame.setdefault(frame, {})[tid] = box
            by_id.setdefault(tid, []).append(frame)
    return by_frame, by_id


def validate_intervals(intervals, by_frame, start, end):
    if not intervals:
        raise ValueError('Confirm at least one interval first.')
    normalized = []
    for item in intervals:
        values = [item[k] for k in ('tracker_id','start_frame','end_frame')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v for v in values):
            raise ValueError('IDs and frames must be whole numbers.')
        tid, first, last = map(int,values)
        if not start <= first <= last <= end:
            raise ValueError('Confirmed interval lies outside this processed segment.')
        if not any(tid in by_frame.get(frame,{}) for frame in range(first,last+1)):
            raise ValueError(f'ID {tid} has no saved boxes inside that interval.')
        normalized.append(dict(tracker_id=tid,start_frame=first,end_frame=last))
    normalized.sort(key=lambda r:r['start_frame'])
    for a,b in zip(normalized,normalized[1:]):
        if b['start_frame'] <= a['end_frame']:
            raise ValueError('Intervals overlap. Trim or remove one; only one identity may be assigned per frame.')
    return normalized


def draw(frame, boxes, selected=None, label='Selected player — reviewed'):
    for tid,box in boxes.items():
        x1,y1,x2,y2=map(int,box)
        color=(40,210,40) if tid==selected else (150,150,150)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        text=f'{label}: ID {tid}' if tid==selected else f'ID {tid}'
        # ASCII text renders consistently with OpenCV's built-in font.
        cv2.putText(frame,text.replace('—','-'),(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    return frame


def preview_images(path, by_frame, tid, first, last):
    images=[]
    for idx in sorted({first,(first+last)//2,last}):
        frame=draw(frame_at(path,idx),by_frame.get(idx,{}),tid,'Candidate')
        images.append((idx,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),tid in by_frame.get(idx,{})))
    return images


def export_review(path, source_directory, output, intervals, progress=None, confirmed=True, bounds=None):
    directory, output=Path(source_directory),Path(output)
    summary=json.loads((directory/'summary.json').read_text())
    stat=Path(path).stat()
    if (summary.get('source_size'),summary.get('source_mtime_ns')) != (stat.st_size,stat.st_mtime_ns):
        raise ValueError('The source video changed. Process it again before review.')
    by_frame,_=load_tracks(directory)
    intervals=validate_intervals(intervals,by_frame,summary['start_frame'],summary['end_frame'])
    first,last=bounds or (summary['start_frame'],summary['end_frame'])
    if not summary['start_frame'] <= first <= last <= summary['end_frame']:
        raise ValueError('Export bounds are outside the processed segment.')
    info=metadata(path)
    transform=calibration(summary['calibration'],info['width'],info['height']) if confirmed and summary.get('calibration') else None
    output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    video=output/'reviewed.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),info['fps'],(info['width'],info['height']))
    if not writer.isOpened():
        cap.release(); raise RuntimeError('Could not create review video.')
    rows=[]
    try:
        for idx in range(first,last+1):
            ok,frame=cap.read()
            if not ok: raise ValueError(f'Could not decode frame {idx}. Export is incomplete.')
            assignment=next((r for r in intervals if r['start_frame']<=idx<=r['end_frame']),None)
            tid=assignment['tracker_id'] if assignment else None
            boxes=by_frame.get(idx,{})
            box=boxes.get(tid)
            status=('human_confirmed' if confirmed else 'candidate_preview') if box is not None else ('missing_detection' if assignment else 'unassigned')
            row=dict(frame=idx,seconds=idx/info['fps'],status=status,tracker_id=tid if tid is not None else '',foot_x='',foot_y='',court_x_m='',court_y_m='')
            if box is not None:
                x,y=(box[0]+box[2])/2,box[3]
                row.update(foot_x=x,foot_y=y)
                if transform is not None:
                    xy=cv2.perspectiveTransform(np.float32([[[x,y]]]),transform)[0,0]
                    if np.isfinite(xy).all() and 0<=xy[0]<=10 and 0<=xy[1]<=20:
                        row.update(court_x_m=float(xy[0]),court_y_m=float(xy[1]))
            draw(frame,boxes,tid,'Selected player - reviewed' if confirmed else 'Candidate - inspect identity')
            if box is None:
                cv2.putText(frame,'No confirmed position' if confirmed else 'No candidate detection',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            writer.write(frame);rows.append(row)
            if progress and (idx-first)%10==0:progress((idx-first+1)/(last-first+1))
    finally:
        cap.release();writer.release()
    with (output/'positions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    playable=False
    if shutil.which('ffmpeg'):
        encoded=output/'reviewed_h264.mp4'
        result=subprocess.run(['ffmpeg','-y','-i',str(video),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)],capture_output=True)
        if result.returncode==0: video=encoded;playable=True
    valid=[r for r in rows if r['court_x_m']!='']
    if valid:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        hist,_,_=np.histogram2d([r['court_x_m'] for r in valid],[r['court_y_m'] for r in valid],bins=[20,40],range=[[0,10],[0,20]])
        fig,ax=plt.subplots(figsize=(5,9))
        im=ax.imshow(hist.T/info['fps'],origin='lower',extent=[0,10,0,20],cmap='magma',aspect='equal')
        ax.axhline(10,color='white');ax.set(xlabel='Court x (m)',ylabel='Court y (m)',title='Estimated occupancy - manually reviewed intervals')
        fig.colorbar(im,ax=ax,label='Seconds per bin');fig.savefig(output/'heatmap.png',dpi=150,bbox_inches='tight');plt.close(fig)
    result=dict(video=video.name,browser_playable=playable,start_frame=first,end_frame=last,processed_frames=len(rows),confirmed_frames=sum(r['status']=='human_confirmed' for r in rows),missing_detection_frames=sum(r['status']=='missing_detection' for r in rows),unassigned_frames=sum(r['status']=='unassigned' for r in rows),heatmap_samples=len(valid),intervals=intervals,review_status='human_confirmed' if confirmed else 'candidate_preview',limits='Manual interval attribution, not automatic recognition. Missing detections and unassigned frames remain blank. Confirmation does not correct bad boxes or calibration. No audio.')
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    (output/('confirmed_intervals.json' if confirmed else 'candidate_intervals.json')).write_text(json.dumps(intervals,indent=2))
    if progress: progress(1.)
    return result
