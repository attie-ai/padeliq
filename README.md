# PadelIQ v0.1 — local Mac prototype

Select Charlotte on a video frame, track people with Ultralytics YOLO + ByteTrack, download an annotated segment and position CSV, and optionally estimate court occupancy. Designed for an Apple M3 Mac; CPU is the compatibility default and MPS is selectable.

## Install and run on macOS

Use Python 3.11 or 3.12. If you use Homebrew:

```sh
brew install python@3.11 ffmpeg
```

Open Terminal, change to this project folder (you can type `cd ` and drag the folder into Terminal), then:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Keep Terminal running. Stop with Control-C. To restart, activate the same environment and run the last command from this folder.

The first detection downloads `yolo11n.pt` if it is missing. Installation and that download require internet; subsequent inference runs locally. To prepare offline use, run:

```sh
python -c 'from ultralytics import YOLO; YOLO("yolo11n.pt")'
```

Keep the weights file and select its local path in the sidebar. No video is intentionally uploaded to a service. Streamlit binds to localhost, usage statistics are disabled, and files remain in `local_data/`. Ultralytics is a third-party dependency; this is not a hardened network-isolated deployment. Model loading may fetch missing weights. Use trusted weights only.

## First practical run

1. Upload the original recording, or enter its absolute Mac path. The earlier ChatGPT `/mnt/data/WhatsApp Video 2026-10-13 at 10.55.19.mp4` path is not a Mac path. That file was unavailable in the development workspace, so it has not been tested.
2. Choose a frame where Charlotte's white top and cap are clearly visible. Set an end frame approximately 10–20 seconds later. Nothing before the start frame is analyzed.
3. Click **Detect players on this frame**. Inspect the numbered boxes and explicitly choose Charlotte. If detection misses her, choose another frame or lower confidence. Numbers on this preview are selection numbers, not tracking IDs.
4. Leave heatmap off on the first run. Click **Track this segment** and wait for progress to finish.
5. Review the annotated video. Green boxes say `Charlotte?` because identity is unverified. Gray boxes are other tracked people. Output contains no audio. FFmpeg creates an H.264 preview when installed; otherwise the MP4 is downloadable but browser playback may fail. Try VLC for the fallback file.
6. If identity loss is flagged, the rest of that segment is left unlabeled. Choose a later clear frame, detect and select Charlotte again, then process a new segment. Each run is independent. If you observe a silent swap, discard the affected run and use shorter segments ending before the swap. v0.1 does not repair or merge segments automatically.
7. Save the ZIP or individual downloads. Previous runs remain under `local_data/runs/`; only the latest completed run is shown in the app.

## Court calibration and interpretation

Only enable calibration for a stationary camera with four identifiable court floor corners. Enter original image pixel coordinates, in perimeter order: far-left, far-right, near-right, near-left. The displayed example values are not a valid real-world calibration; replace them before confirming. Verify the magenta polygon follows the court floor.

The app maps that quadrilateral to a 10 × 20 m rectangle. If this is not the full doubles court, do not enable it. A single homography cannot correct camera movement, lens distortion, elevated points, or an incorrect corner choice. The point used is the bottom-center of the detection box, an imperfect foot-contact estimate. Glass, occlusion and truncated boxes can move this point substantially. Out-of-court estimates are excluded.

Heatmap bins represent seconds of observed, unverified tracking inside the calibrated court. Missing detections and frames after identity loss contribute nothing. It is not a complete-match occupancy map and is not a speed, distance, shot quality or coaching assessment. Compare `heatmap_samples`, `tracked_frames` and `processed_frames` in the summary to see how much data contributed.

## Outputs

- `annotated_h264.mp4` when FFmpeg succeeds, otherwise `annotated.mp4`: segment at source frame rate and resolution, without audio.
- `positions.csv`: original frame index and source-relative seconds; status, tracker ID, image foot point and optional court coordinates. Lost intervals have blank coordinates. Tracker ID on lost rows records the original selected ID, not a verified current identity.
- `summary.json`: settings, calibration, frame counts, first loss and explicit limitations.
- `heatmap.png`: only generated when a valid calibration and at least one in-court sample exist.

Downloads and processing use memory and disk. Start with short segments. Upload limit is 2 GB; a local path avoids the upload copy. The UI is synchronous during processing and has no in-app cancellation button. Stop the server with Control-C if needed; failed/interrupted runs can leave partial files, but are not presented as completed results. Delete `local_data/` after stopping the app to remove recordings and results.

## Identity and honest limits

ByteTrack associates detections; it does not recognize Charlotte. White clothing is a human selection cue, not a biometric or persistent identity model. Fresh tracker state is created for each run. Initial selection must uniquely overlap a tracked box. Missing selected IDs, implausible per-frame motion/scale changes and strong overlap with another player trigger permanent loss for that segment. This conservative policy intentionally rejects some recoverable occlusions.

These gates cannot detect every identity swap, especially close crossings. Review all output. There is no automatic re-identification, interpolation across gaps, shot/ball tracking, performance scoring or reliable coaching insight in v0.1. Person detection is trained for general scenes and may miss distant players or track spectators/reflections. Fixed frame-rate handling assumes an approximately constant-frame-rate recording; variable-frame-rate timestamps are approximate. Export to constant-frame-rate H.264 if timing accuracy matters.

## Troubleshooting

- MPS errors or unsupported operations: choose CPU and retry. No M3 throughput benchmark was performed.
- Weights download errors: download trusted Ultralytics weights while online and use the local file path.
- No people: choose a clearer frame, lower confidence, or try `yolo11s.pt` (slower).
- Missing `lap`: rerun the requirements installation in the activated environment.
- Decode errors: export the video as H.264 MP4 and retry; incomplete output is not accepted as a successful run.
- Port busy: use `python -m streamlit run app.py --server.port 8502`.

See TESTING.md for acceptance checks and recorded verification. Dependency ranges allow compatible bug fixes but are not a reproducible lockfile; after successful installation save your exact environment with `python -m pip freeze > installed-requirements.txt`.

Official API references: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/), [Streamlit uploads](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader), [Streamlit downloads](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button). Ultralytics has AGPL-3.0 and commercial licensing options; review its license before redistribution or commercial deployment.
