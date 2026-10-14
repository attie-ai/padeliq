from pathlib import Path
import hashlib
import uuid
import zipfile
import io
import json
import cv2
import numpy as np
import streamlit as st
from engine import metadata, frame_at, process, calibration

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'local_data'
DATA.mkdir(exist_ok=True)
st.set_page_config(page_title='PadelTrace',page_icon='🎾',layout='wide')
st.title('🎾 PadelTrace v0.1')
st.caption('Local video analysis • manual identity selection • experimental tracking')
st.info('Select Charlotte yourself using her white top and cap. Review the output: a tracker ID can silently switch people. Brief missing detections pause labeling for up to half a second. Only the same tracker ID can resume; prolonged gaps, suspicious jumps or overlaps require a new selection.')
upload = st.file_uploader('Upload a recording (stored on this Mac)',type=['mp4','mov','m4v','avi'])
local = st.text_input('Or enter an absolute local video path',value='/mnt/data/WhatsApp Video 2026-10-13 at 10.55.19.mp4')
if upload is not None:
    content = upload.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    path = DATA/(digest+Path(upload.name).suffix.lower())
    if not path.exists(): path.write_bytes(content)
else:
    path = Path(local).expanduser()
if not path.is_file():
    st.warning('The referenced ChatGPT video is not on this Mac at that path. Upload it above or enter its Mac path.')
    st.stop()
try:
    info = metadata(path)
except Exception as e:
    st.error(str(e)); st.stop()
# Stable source identity prevents displaying a prior video's run after changing inputs.
source_key = (str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns)
if st.session_state.get('source_key') != source_key:
    for key in ('detection', 'run', 'review_run_key', 'review_intervals', 'combined_result', 'candidate_video'):
        st.session_state.pop(key, None)
    st.session_state.source_key = source_key
st.write(f"{info['width']} × {info['height']} • {info['fps']:.2f} fps • {info['frames']/info['fps']:.1f} seconds")
# Completed runs can be reopened after a restart, for this exact saved source.
saved_runs={}
for candidate in (DATA/'runs').glob('*/summary.json'):
    try:
        import json
        saved=json.loads(candidate.read_text())
        if (saved.get('source_size'),saved.get('source_mtime_ns'))==(path.stat().st_size,path.stat().st_mtime_ns):
            saved_runs[str(candidate.parent)]=saved
    except (ValueError,OSError):
        continue
if saved_runs:
    with st.expander('Resume a saved processing run'):
        prior=st.selectbox('Saved run for this recording',sorted(saved_runs,reverse=True),index=None,
            format_func=lambda key:f"Frames {saved_runs[key]['start_frame']}–{saved_runs[key]['end_frame']} · {Path(key).name[:8]}")
        if st.button('Open saved run',disabled=prior is None):
            st.session_state.run=(prior,saved_runs[prior])
            st.session_state.pop('review_run_key',None)
with st.sidebar:
    st.header('Processing')
    model_path = st.text_input('YOLO weights file or model name',value='yolo11s.pt',help='The first use of a model name downloads weights. Use a local .pt file to work offline.')
    device = st.selectbox('Compute device',['cpu','mps'],help='Start with CPU for compatibility. MPS uses Apple Metal; if it fails, rerun on CPU.')
    conf = st.slider('Person confidence',.1,.8,.25,.05)
    st.caption('All frames in the selected segment are processed in order. Start with a short 10–20 second segment.')
start = int(st.number_input('Selection / segment start frame',min_value=0,max_value=info['frames']-1,value=0,step=1))
end = int(st.number_input('Segment end frame (inclusive)',min_value=start,max_value=info['frames']-1,value=min(info['frames']-1,start+int(info['fps']*15)),step=1))
try:
    frame = frame_at(path,start)
except Exception as e:
    st.error(str(e)); st.stop()
selection_key = (source_key,start,model_path,device,conf)
if st.button('Detect players on this frame'):
    try:
        from ultralytics import YOLO
        with st.spinner('Detecting people locally…'):
            result = YOLO(model_path).predict(frame,classes=[0],device=device,conf=conf,verbose=False)[0]
        st.session_state.detection = (selection_key,result.boxes.xyxy.cpu().numpy().tolist())
    except Exception as e:
        st.error(f'Detection failed: {e}. Check installation/weights, or try CPU.')
boxes = st.session_state.get('detection',(None,[]))[1] if st.session_state.get('detection',(None,[]))[0]==selection_key else []
preview = frame.copy()
for i,b in enumerate(boxes):
    x1,y1,x2,y2=map(int,b)
    cv2.rectangle(preview,(x1,y1),(x2,y2),(0,230,255),2)
    cv2.putText(preview,f'Person {i+1}',(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,230,255),2)
st.image(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB),caption=f'Frame {start}. Coordinates refer to original {info["width"]} × {info["height"]} pixels.',width='stretch')
selected = None
if boxes:
    person = st.selectbox('Which numbered person is Charlotte?',range(1,len(boxes)+1),index=None,placeholder='Select after inspecting the frame',key='person_'+hashlib.sha256(repr(selection_key).encode()).hexdigest())
    if person is not None: selected = boxes[person-1]
else:
    st.caption('Detect people first. If Charlotte is not detected, try another clear frame or a lower confidence threshold.')
points = None
if st.checkbox('Add a court heatmap (fixed camera only)'):
    st.write('Enter the four floor corners in perimeter order: far-left, far-right, near-right, near-left. This maps the full doubles court to 10 × 20 m. Use the visible floor-wall intersections, not the top of the fence. Skip this if corners are hidden, the camera moves, or perspective is unclear.')
    import pandas as pd
    corners = st.data_editor(pd.DataFrame({'corner':['far-left','far-right','near-right','near-left'],'x':[0.,float(info['width']-1),float(info['width']-1),0.],'y':[0.,0.,float(info['height']-1),float(info['height']-1)]}),disabled=['corner'],hide_index=True)
    confirmed = st.checkbox('I replaced the example corners with real court corners and verified the camera stays fixed',value=False)
    if confirmed:
        try:
            points=corners[['x','y']].to_numpy().tolist()
            calibration(points,info['width'],info['height'])
            overlay = preview.copy()
            cv2.polylines(overlay,[np.asarray(points,dtype=np.int32)],True,(255,50,255),3)
            st.image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB),caption='Check the magenta boundary follows the court floor.',width='stretch')
        except Exception as e:
            st.error(str(e)); points=None
    if points is None:
        st.warning('Heatmap disabled until calibration is confirmed and valid. Tracking can still run.')
if st.button('Track this segment',type='primary',disabled=selected is None):
    directory = DATA/'runs'/uuid.uuid4().hex
    bar = st.progress(0.,text='Processing locally…')
    try:
        summary = process(path,directory,start,end,selected,model_path,device,conf,points,bar.progress)
        st.session_state.run = (str(directory),summary)
        st.session_state.pop('review_run_key',None)
    except Exception as e:
        st.error(f'Processing failed: {e}. No completed result was saved for this attempt. Try a short segment on CPU.')
    finally:
        bar.empty()
if 'run' in st.session_state:
    directory,summary = st.session_state.run
    directory = Path(directory)
    st.subheader('Latest completed segment')
    if (directory/'tracking_issue.jpg').exists():
        st.image(str(directory/'tracking_issue.jpg'),caption=f"First tracking issue at frame {summary.get('first_uncertain_frame')}. See the reason fields in the summary and CSV.",width='stretch')
    st.json(summary)
    if summary['first_loss_frame'] is not None:
        st.warning(f"Identity was lost at frame {summary['first_loss_frame']}. Review around that point, use the interval review section below to confirm this player’s later IDs, or select a new start frame. Reason: {summary.get('first_loss_reason', 'not recorded by this older run')}. Heatmap excludes missing and lost intervals.")
    st.caption('This result belongs to the settings recorded above. Review the whole segment for silent identity swaps before trusting its heatmap.')
    if summary['browser_playable']: st.video(str(directory/summary['video']))
    else: st.info('Download the MP4 to play in VLC/QuickTime. Install FFmpeg before processing to enable a browser-compatible H.264 preview.')
    if (directory/'heatmap.png').exists(): st.image(str(directory/'heatmap.png'))
    for name,mime in [(summary['video'],'video/mp4'),('positions.csv','text/csv'),('all_tracks.csv','text/csv'),('summary.json','application/json'),('heatmap.png','image/png'),('tracking_issue.jpg','image/jpeg')]:
        file = directory/name
        if file.exists(): st.download_button('Download '+name,file.read_bytes(),file_name=name,mime=mime)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in directory.iterdir():
            if file.is_file(): z.write(file,file.name)
    st.download_button('Download all results (ZIP)',archive.getvalue(),'padeltrace-results.zip','application/zip')
    st.caption(f'Results are also saved locally at {directory}. Delete local_data to remove uploaded recordings and runs. Large downloads consume memory; use the local files for very long recordings.')

    st.divider()
    st.subheader('Join this player’s changing IDs')
    st.write('1. Choose an ID below. 2. Inspect its video and trim the range if needed. 3. Confirm that stretch belongs to your player. Repeat for later IDs, then build the combined result.')
    if not (directory/'all_tracks.csv').exists():
        st.warning('This older run did not save every player’s boxes. Process the video again once with this updated app, then review the new IDs.')
    else:
        from review import load_tracks, preview_images, export_review
        run_key=str(directory)
        if st.session_state.get('review_run_key')!=run_key:
            st.session_state.review_run_key=run_key
            import json
            draft=directory/'review_intervals.json'
            try:
                st.session_state.review_intervals=json.loads(draft.read_text()) if draft.exists() else []
            except (ValueError,OSError):
                st.session_state.review_intervals=[]
            st.session_state.pop('combined_result',None)
            st.session_state.pop('candidate_video',None)
        @st.cache_data(show_spinner=False)
        def saved_tracks(filename, size, modified):
            return load_tracks(Path(filename).parent)
        trackfile=directory/'all_tracks.csv'
        by_frame,by_id=saved_tracks(str(trackfile),trackfile.stat().st_size,trackfile.stat().st_mtime_ns)
        import pandas as pd
        st.caption('IDs belong to this completed run. Reprocessing can assign different numbers. A preview image alone cannot confirm an entire interval; inspect the interval video for identity swaps.')
        overview=[dict(ID=tid,first_frame=min(frames),last_frame=max(frames),detected_frames=len(frames)) for tid,frames in sorted(by_id.items())]
        st.dataframe(pd.DataFrame(overview),hide_index=True,width='stretch')
        tid=st.selectbox('Choose a tracker ID to review',sorted(by_id),index=None,placeholder='Choose an ID from this run',key='review_id_'+directory.name)
        if tid is not None:
            first,last=min(by_id[tid]),max(by_id[tid])
            if first<last:
                first,last=st.slider('Frame interval to confirm',first,last,(first,last),key=f'range_{directory.name}_{tid}')
            st.write(f'ID {tid} · frames {first}–{last} · source time {first/info["fps"]:.2f}–{last/info["fps"]:.2f} seconds')
            try:
                previews=preview_images(path,by_frame,tid,first,last)
                for col,(idx,img,present) in zip(st.columns(len(previews)),previews):
                    col.image(img,caption=f'Frame {idx}'+(' — candidate box' if present else ' — this ID is missing'),width='stretch')
            except Exception as e: st.error(f'Could not show preview: {e}')
            signature=(run_key,tid,first,last)
            interval=dict(tracker_id=tid,start_frame=first,end_frame=last)
            if st.button('Create video preview of this interval'):
                previewdir=directory/'candidate_previews'/uuid.uuid4().hex
                bar=st.progress(0.,text='Preparing interval preview without running YOLO again…')
                try:
                    result=export_review(path,directory,previewdir,[interval],bar.progress,confirmed=False,bounds=(first,last))
                    st.session_state.candidate_video=(signature,str(previewdir),result)
                except Exception as e: st.error(str(e))
                finally: bar.empty()
            candidate=st.session_state.get('candidate_video')
            if candidate and candidate[0]==signature:
                _,previewdir,result=candidate
                video=Path(previewdir)/result['video']
                if result['browser_playable']:st.video(str(video))
                st.download_button('Download interval preview',video.read_bytes(),file_name=f'id-{tid}-{first}-{last}.mp4',mime='video/mp4')
            checked=st.checkbox('I inspected this exact interval and confirm the visible boxes belong to the same selected player',key='confirm_'+hashlib.sha256(repr(signature).encode()).hexdigest())
            if st.button('Add confirmed interval',disabled=not checked):
                from review import validate_intervals
                try:
                    accepted=validate_intervals(st.session_state.review_intervals+[interval],by_frame,summary['start_frame'],summary['end_frame'])
                    st.session_state.review_intervals=accepted
                    (directory/'review_intervals.json').write_text(json.dumps(accepted,indent=2))
                    st.session_state.pop('combined_result',None)
                    st.rerun()
                except ValueError as e: st.error(str(e))
        accepted=st.session_state.review_intervals
        st.markdown('**Confirmed intervals for your player**')
        if not accepted:st.info('No intervals confirmed yet. Nothing is automatically accepted, including the first ID.')
        else:
            st.dataframe(pd.DataFrame(accepted),hide_index=True,width='stretch')
            remove=st.selectbox('Interval to remove',range(len(accepted)),format_func=lambda i:f"ID {accepted[i]['tracker_id']} · {accepted[i]['start_frame']}–{accepted[i]['end_frame']}",key='remove_'+directory.name)
            if st.button('Remove this interval'):
                st.session_state.review_intervals=[r for i,r in enumerate(accepted) if i!=remove]
                (directory/'review_intervals.json').write_text(json.dumps(st.session_state.review_intervals,indent=2))
                st.session_state.pop('combined_result',None)
                st.rerun()
        st.caption('Unconfirmed frames and missing detections remain blank. Any heatmap uses this run’s original court calibration; adding calibration now requires a new processing run.')
        if st.button('Build combined reviewed result',type='primary',disabled=not accepted):
            output=directory/'reviewed_results'/uuid.uuid4().hex
            bar=st.progress(0.,text='Building combined outputs locally…')
            try:
                result=export_review(path,directory,output,accepted,bar.progress)
                st.session_state.combined_result=(str(output),result)
            except Exception as e:st.error(f'Export failed: {e}')
            finally:bar.empty()
        if 'combined_result' in st.session_state:
            output,result=st.session_state.combined_result
            output=Path(output)
            st.subheader('Combined reviewed result')
            st.json(result)
            if result['browser_playable']:st.video(str(output/result['video']))
            if (output/'heatmap.png').exists():st.image(str(output/'heatmap.png'))
            data=io.BytesIO()
            with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:
                for file in output.iterdir():
                    if file.is_file():z.write(file,file.name)
            st.download_button('Download combined result ZIP',data.getvalue(),'padeltrace-reviewed-results.zip','application/zip')
            for name,mime in [(result['video'],'video/mp4'),('positions.csv','text/csv'),('confirmed_intervals.json','application/json')]:
                st.download_button('Download reviewed '+name,(output/name).read_bytes(),file_name='reviewed-'+name,mime=mime)
