from pathlib import Path
import hashlib
import uuid
import zipfile
import io
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
st.info('Select Charlotte yourself using her white top and cap. Review the output: a tracker ID can silently switch people. If a detection disappears or jumps, Charlotte labeling stops for the rest of that segment.')
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
    for key in ('detection', 'run'):
        st.session_state.pop(key, None)
    st.session_state.source_key = source_key
st.write(f"{info['width']} × {info['height']} • {info['fps']:.2f} fps • {info['frames']/info['fps']:.1f} seconds")
with st.sidebar:
    st.header('Processing')
    model_path = st.text_input('YOLO weights file or model name',value='yolo11n.pt',help='The first use of a model name downloads weights. Use a local .pt file to work offline.')
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
    except Exception as e:
        st.error(f'Processing failed: {e}. No completed result was saved for this attempt. Try a short segment on CPU.')
    finally:
        bar.empty()
if 'run' in st.session_state:
    directory,summary = st.session_state.run
    directory = Path(directory)
    st.subheader('Latest completed segment')
    st.json(summary)
    if summary['first_loss_frame'] is not None:
        st.warning(f"Identity was lost at frame {summary['first_loss_frame']}. Review around that point, choose a later clear start frame, detect players and select Charlotte again. Heatmap excludes the lost interval.")
    st.caption('This result belongs to the settings recorded above. Review the whole segment for silent identity swaps before trusting its heatmap.')
    if summary['browser_playable']: st.video(str(directory/summary['video']))
    else: st.info('Download the MP4 to play in VLC/QuickTime. Install FFmpeg before processing to enable a browser-compatible H.264 preview.')
    if (directory/'heatmap.png').exists(): st.image(str(directory/'heatmap.png'))
    for name,mime in [(summary['video'],'video/mp4'),('positions.csv','text/csv'),('summary.json','application/json'),('heatmap.png','image/png')]:
        file = directory/name
        if file.exists(): st.download_button('Download '+name,file.read_bytes(),file_name=name,mime=mime)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in directory.iterdir(): z.write(file,file.name)
    st.download_button('Download all results (ZIP)',archive.getvalue(),'padeltrace-results.zip','application/zip')
    st.caption(f'Results are also saved locally at {directory}. Delete local_data to remove uploaded recordings and runs. Large downloads consume memory; use the local files for very long recordings.')
