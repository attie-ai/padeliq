from pathlib import Path
import csv
import json
import subprocess
import shutil
import cv2
import numpy as np
from geometry import iou, plausible


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot decode this video. Try exporting it as H.264 MP4.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(3)), int(cap.get(4))
        if not np.isfinite(fps) or fps <= 0 or n <= 0 or min(w,h) <= 0:
            raise ValueError('Video has invalid duration, dimensions or frame rate.')
        return dict(fps=fps, frames=n, width=w, height=h)
    finally:
        cap.release()


def frame_at(path, index):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Could not read the selected frame.')
        return frame
    finally:
        cap.release()


def calibration(points, width, height):
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (4,2) or not np.isfinite(p).all():
        raise ValueError('Enter four finite court corner coordinates.')
    if (p[:,0]<0).any() or (p[:,0]>=width).any() or (p[:,1]<0).any() or (p[:,1]>=height).any():
        raise ValueError('Court corners must lie inside the image.')
    contour = p.reshape(-1,1,2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < width*height*.005:
        raise ValueError('Corners must form a visible, convex court quadrilateral in perimeter order.')
    return cv2.getPerspectiveTransform(p, np.float32([[0,0],[10,0],[10,20],[0,20]]))


def process(path, output, start, end, selected, model_path, device, conf, points=None, progress=None, gap_seconds=0.5):
    from ultralytics import YOLO
    info = metadata(path)
    if not 0 <= start <= end < info['frames']:
        raise ValueError('Segment frames are outside the video.')
    if len(selected) != 4 or not np.isfinite(selected).all():
        raise ValueError('Invalid player selection.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    transform = calibration(points, info['width'], info['height']) if points is not None else None
    model = YOLO(model_path)  # Fresh tracker for each segment.
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    raw = output/'annotated.mp4'
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), info['fps'], (info['width'],info['height']))
    rows, target, previous, lost = [], None, None, False
    missing_frames, recovered_gaps, loss_reason = 0, 0, ""
    max_missing_frames = max(0, round(info["fps"] * gap_seconds))
    issue_saved = False
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('OpenCV could not create the output video.')
    tracks_file = (output/'all_tracks.csv').open('w', newline='')
    tracks_writer = csv.DictWriter(tracks_file, fieldnames=['frame','seconds','tracker_id','x1','y1','x2','y2'])
    tracks_writer.writeheader()
    try:
        for idx in range(start, end+1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Decoding failed at frame {idx}; partial output is incomplete.')
            result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0], conf=conf, device=device, verbose=False)[0]
            boxes = result.boxes
            candidates = [] if boxes is None or boxes.id is None else list(zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().numpy()))
            for tid, box in candidates:
                tracks_writer.writerow(dict(frame=idx, seconds=idx/info['fps'], tracker_id=tid, **dict(zip(['x1','y1','x2','y2'], map(float,box)))))
            if idx == start:
                ranked = sorted([(iou(selected,b),t,b) for t,b in candidates], key=lambda item:item[0], reverse=True)
                if not ranked or ranked[0][0] < .5 or (len(ranked)>1 and ranked[1][0]>.3):
                    raise ValueError('Selected player could not be matched uniquely. Choose another clear frame.')
                _, target, previous = ranked[0]
            match = next((b for t,b in candidates if t == target), None)
            reason = loss_reason if lost else ''
            active = False
            if not lost:
                if match is None:
                    missing_frames += 1
                    reason = 'selected_tracker_id_missing'
                    if missing_frames > max_missing_frames:
                        lost = True
                        loss_reason = reason = 'selected_tracker_id_missing_timeout'
                elif not plausible(previous, match):
                    lost = True
                    loss_reason = reason = 'implausible_box_motion_or_scale'
                elif any(iou(match,b) > .35 for t,b in candidates if t != target):
                    lost = True
                    loss_reason = reason = 'overlap_with_another_player'
                else:
                    if missing_frames:
                        recovered_gaps += 1
                    missing_frames = 0
                    active = True
            status = 'identity_lost' if lost else ('tracked_unverified' if active else 'temporarily_missing')
            row = dict(frame=idx, seconds=idx/info['fps'], status=status, tracker_id=target,
                       reason=reason, visible_tracker_ids=';'.join(str(t) for t,b in candidates),
                       foot_x='', foot_y='', court_x_m='', court_y_m='')
            for t,b in candidates:
                x1,y1,x2,y2 = map(int,b)
                selected_visible = active and t==target
                color = (40,210,40) if selected_visible else (150,150,150)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame, f'Charlotte? ID {t}' if selected_visible else f'ID {t}', (x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            if active:
                previous = match
                x,y = float((match[0]+match[2])/2),float(match[3])
                row.update(foot_x=x,foot_y=y)
                if transform is not None:
                    xy = cv2.perspectiveTransform(np.float32([[[x,y]]]),transform)[0,0]
                    if np.isfinite(xy).all() and 0<=xy[0]<=10 and 0<=xy[1]<=20:
                        row.update(court_x_m=float(xy[0]),court_y_m=float(xy[1]))
            else:
                label = 'IDENTITY LOST' if lost else 'TEMPORARILY MISSING - positions excluded'
                cv2.putText(frame,label,(15,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            if reason and not issue_saved:
                cv2.imwrite(str(output/'tracking_issue.jpg'), frame)
                issue_saved = True
            writer.write(frame)
            rows.append(row)
            if progress and (idx-start)%10==0:
                progress((idx-start+1)/(end-start+1))
    finally:
        tracks_file.close()
        writer.release()
        cap.release()
    with (output/'positions.csv').open('w',newline='') as f:
        csvwriter = csv.DictWriter(f,fieldnames=list(rows[0]))
        csvwriter.writeheader()
        csvwriter.writerows(rows)
    playable = False
    if shutil.which('ffmpeg'):
        encoded = output/'annotated_h264.mp4'
        result = subprocess.run(['ffmpeg','-y','-i',str(raw),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)],capture_output=True)
        if result.returncode==0:
            raw = encoded
            playable = True
    valid = [r for r in rows if r['court_x_m']!='']
    if transform is not None and valid:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        hist, _, _ = np.histogram2d([r['court_x_m'] for r in valid],[r['court_y_m'] for r in valid],bins=[20,40],range=[[0,10],[0,20]])
        fig, ax = plt.subplots(figsize=(5,9))
        im=ax.imshow(hist.T/info['fps'],origin='lower',extent=[0,10,0,20],cmap='magma',aspect='equal')
        ax.axhline(10,color='white'); ax.set(xlabel='Court x (m)',ylabel='Court y (m)',title='Estimated occupancy — unverified tracking')
        fig.colorbar(im,ax=ax,label='Seconds per bin'); fig.savefig(output/'heatmap.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    summary = dict(all_tracks='all_tracks.csv',fps=info['fps'],source_size=Path(path).stat().st_size,source_mtime_ns=Path(path).stat().st_mtime_ns,video=raw.name,browser_playable=playable,start_frame=start,end_frame=end,processed_frames=len(rows),tracked_frames=sum(r['status']=='tracked_unverified' for r in rows),first_loss_frame=next((r['frame'] for r in rows if r['status']=='identity_lost'),None),first_loss_reason=loss_reason or None,first_uncertain_frame=next((r['frame'] for r in rows if r['reason']),None),temporarily_missing_frames=sum(r['status']=='temporarily_missing' for r in rows),recovered_gaps=recovered_gaps,gap_tolerance_seconds=gap_seconds,heatmap_samples=len(valid),device=device,model=model_path,confidence=conf,calibration=points,limits='Tracker identity is unverified. Foot point is box bottom-center. No automatic re-identification. Fixed-camera planar court only. Output has no audio.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    if progress: progress(1.)
    return summary
