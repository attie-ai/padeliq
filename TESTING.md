# Practical testing plan

## Automated checks

From the project directory:

```sh
python -m compileall -q .
python -m unittest discover -s tests -v
```

The lightweight tests exercise selection overlap and motion gates. Runtime tests exercise the complete video pipeline with deterministic synthetic detections; they do not validate YOLO accuracy or ByteTrack association on real padel footage.

## Mac acceptance checks with the original recording

1. Install in a clean Python 3.11/3.12 environment, launch on localhost and upload the recording. Verify reported dimensions, duration and frame rate. Try an invalid path and a corrupt video: expect readable errors.
2. Select a clear frame. Verify boxes are numbered, no player is preselected, changing frame or detection settings invalidates selection, and the chosen box is Charlotte. Test a frame with no detections.
3. Process a short CPU segment without calibration. Confirm progress completes; output duration is `(end-start+1)/fps`, CSV rows equal processed frames, timestamps start at the selected source time, and video remains correctly oriented. Play/download all outputs.
4. Repeat on MPS. Compare boxes and timing, noting numerical differences. If MPS fails, confirm CPU can complete. Record RAM, macOS, Python, package versions and processing seconds per video second.
5. Review a crossing/occlusion segment frame by frame. Brief missing detections should pause coordinates for up to 0.5 seconds; the same ID can resume only after passing the gates. Another ID must never be substituted. Gaps beyond the tolerance, suspicious jumps or overlaps should flag permanent loss; all subsequent coordinates must stay blank even if the same ID returns. Check first_loss_reason and tracking_issue.jpg. Check silent swaps manually. Start a new segment after loss and reselect Charlotte. Confirm previous files remain available locally.
6. Test several segments across the recording, including Charlotte at the far end, glass reflections, players near the net and motion blur. Log wrong-person frames, lost frames, first loss, and uninterrupted reviewed seconds. A claim of reliable full-match identity needs substantially more validation than this prototype.
7. Confirm calibration cannot be used without explicit confirmation. Enter crossed, duplicate or outside-image corners: expect rejection. With real floor corners, verify the overlay and that projected known court points are reasonable. Skip heatmaps for camera pans or hidden corners.
8. Review a calibrated segment. Check only pre-loss, in-court samples contribute. Sum of all bin seconds should equal `heatmap_samples/fps`; labeled duration is not automatically the whole segment duration.
9. After downloading weights, disconnect internet and repeat detection and processing using the local weights path. Confirm results work offline.
10. Change the video source after a run: old source results and player selections should clear. Stop and restart; locate saved outputs, then delete test recordings if desired.

## Acceptance threshold

The prototype is accepted when installation, manual selection, short segment processing, conservative loss behavior, downloadable outputs and optional calibration function on the Mac. Track quality is accepted only for segments a human has reviewed. No automatic accuracy or coaching-quality claim follows from passing functional tests.

## Development verification

The supplied original video was absent at its referenced path. Real-video detection, ByteTrack identity quality and M3/MPS performance remain unverified. See the accompanying verification record for checks actually run in the development environment.


## Manual interval review acceptance

- Process a crossing segment and confirm `all_tracks.csv` includes later player IDs after the original ID is lost.
- Choose an ID, inspect the interval preview, and trim its range. Confirmation must be unchecked for a newly chosen interval; no range is accepted by default.
- Add intervals spanning successive IDs belonging to the same player. Reject overlapping ranges, invalid frames and nonexistent IDs. Remove an accepted interval and verify an earlier combined result is invalidated.
- Export the complete original segment: only accepted detections have coordinates. Missing detections and unassigned frames remain blank, and only confirmed in-court samples contribute to the reviewed heatmap.
- Restart the app, upload the same recording and open the saved run. Confirm draft intervals persist. An older run without all_tracks.csv must request reprocessing. Replacing the source video must prevent export from stale tracking data.
- Review the whole combined video visually. Synthetic export tests do not establish identity accuracy on real footage.
