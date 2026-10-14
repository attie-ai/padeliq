# Verification record — 13 October 2026

Checks performed in an isolated macOS arm64 environment with Python 3.14.7, Streamlit 1.65.0, Ultralytics 8.4.176, OpenCV 4.14.0.94, NumPy 2.5.4 and PyTorch 2.14.1. The installation guide recommends Python 3.11/3.12 for broader compatibility; those specific interpreter environments were not exercised here.

- Dependency installation: successful.
- Python source compilation: passed.
- Eight automated tests: passed. These cover overlap/motion gates, ambiguous initial selection, permanent identity loss after disappearance, overlapping players, invalid calibration/video/frame, actual OpenCV MP4 writing/reading, CSV timing, and heatmap PNG generation.
- Streamlit AppTest smoke: passed for missing source, valid local synthetic video, processing disabled before player selection, and optional calibration pending confirmation. Repeated after updating image sizing to the current Streamlit API.
- Real Ultralytics `YOLO.track` with `persist=True`, `tracker='bytetrack.yaml'`, `classes=[0]`, CPU: passed on two blank synthetic frames using a model built from `yolo11n.yaml` with random weights. This validates the API integration, not pretrained detection quality or player association accuracy.

Not verified: original Charlotte recording (file unavailable), pretrained-model download/detection on that recording, M3/MPS performance, full browser interaction and video playback, and FFmpeg H.264 conversion (FFmpeg was absent). The video pipeline tests used deterministic synthetic detections instead of YOLO outputs. Functional tests cannot establish real-world identity reliability.

A source-code prototype is delivered, not a packaged macOS application. Install and run it using README.md, then follow TESTING.md with the actual recording before interpreting its outputs.

## Tracking diagnostics update

Ten synthetic functional tests passed after adding a 0.5-second missing-detection tolerance and explicit loss diagnostics. Checks include blank coordinates during a short gap, same-ID recovery, refusal to adopt a different ID, timeout, sticky motion/overlap rejection, and video/heatmap output. This does not establish the cause of the reported frame-6002 failure or validate recovery on the user recording; rerun that segment and inspect the new reason and first-issue image.


## Manual interval review update — 14 October 2026

Sixteen automated tests passed, including saved detections after identity loss, combining manually selected IDs, blank missing/unassigned frames, exclusion of preview data from heatmaps, rejection of overlap/invalid ranges/changed sources, and Streamlit confirmation/add/export/remove interactions. Video exports and reviewed heatmaps were checked using a synthetic recording. Real padel identity joining remains subject to the user’s review; no automatic re-identification claim is made.
