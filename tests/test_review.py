import csv
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from review import load_tracks, validate_intervals, export_review

class ManualReview(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'source.mp4';self.run=self.root/'run';self.run.mkdir()
        w=cv2.VideoWriter(str(self.source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
        self.assertTrue(w.isOpened())
        for _ in range(6):w.write(np.zeros((120,160,3),np.uint8))
        w.release()
        self.summary=dict(start_frame=0,end_frame=5,source_size=self.source.stat().st_size,source_mtime_ns=self.source.stat().st_mtime_ns,calibration=[[0,0],[159,0],[159,119],[0,119]])
        (self.run/'summary.json').write_text(json.dumps(self.summary))
        with (self.run/'all_tracks.csv').open('w',newline='') as f:
            out=csv.DictWriter(f,fieldnames=['frame','seconds','tracker_id','x1','y1','x2','y2']);out.writeheader()
            for frame,tid in [(0,2),(1,2),(3,110),(4,110),(5,138)]:
                out.writerow(dict(frame=frame,seconds=frame/10,tracker_id=tid,x1=20,y1=20,x2=40,y2=80))
        self.intervals=[dict(tracker_id=2,start_frame=0,end_frame=2),dict(tracker_id=110,start_frame=3,end_frame=4),dict(tracker_id=138,start_frame=5,end_frame=5)]
    def tearDown(self):self.temp.cleanup()
    def test_combines_new_ids_and_leaves_hidden_frames_blank(self):
        result=export_review(self.source,self.run,self.root/'result',self.intervals)
        self.assertEqual(result['confirmed_frames'],5)
        self.assertEqual(result['missing_detection_frames'],1)
        self.assertEqual(result['heatmap_samples'],5)
        with (self.root/'result'/'positions.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual([r['tracker_id'] for r in rows],['2','2','2','110','110','138'])
        self.assertEqual(rows[2]['foot_x'],'')
        self.assertEqual(rows[2]['court_x_m'],'')
        self.assertTrue((self.root/'result'/'heatmap.png').exists())
    def test_unassigned_frames_are_not_invented(self):
        result=export_review(self.source,self.run,self.root/'result',self.intervals[1:])
        self.assertEqual(result['unassigned_frames'],3)
        self.assertEqual(result['confirmed_frames'],3)
    def test_preview_does_not_claim_confirmation_or_build_heatmap(self):
        result=export_review(self.source,self.run,self.root/'preview',self.intervals[1:2],confirmed=False,bounds=(3,4))
        self.assertEqual(result['review_status'],'candidate_preview')
        self.assertEqual(result['confirmed_frames'],0)
        self.assertEqual(result['processed_frames'],2)
        self.assertFalse((self.root/'preview'/'heatmap.png').exists())
    def test_rejects_overlap_unknown_id_and_invalid_ranges(self):
        tracks,_=load_tracks(self.run)
        for intervals in [[],self.intervals+[self.intervals[0]],[dict(tracker_id=999,start_frame=0,end_frame=5)],[dict(tracker_id=2,start_frame=-1,end_frame=1)],[dict(tracker_id=2,start_frame=.5,end_frame=1)]]:
            with self.assertRaises(ValueError):validate_intervals(intervals,tracks,0,5)
    def test_rejects_changed_source(self):
        self.summary['source_size']+=1
        (self.run/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaises(ValueError):export_review(self.source,self.run,self.root/'result',self.intervals)
