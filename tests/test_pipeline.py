"""Functional pipeline tests with real OpenCV video and fake detections."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys
import numpy as np
import cv2
from engine import calibration, process, metadata, frame_at

class Tensor:
    def __init__(self, data): self.data=np.asarray(data)
    def int(self): return self
    def cpu(self): return self
    def numpy(self): return self.data
    def tolist(self): return self.data.tolist()

class FakeModel:
    sequence=[]
    def __init__(self, path): self.index=0
    def track(self, frame, **kwargs):
        assert kwargs['persist'] and kwargs['tracker']=='bytetrack.yaml' and kwargs['classes']==[0]
        ids, boxes=self.sequence[self.index]
        self.index+=1
        return [SimpleNamespace(boxes=SimpleNamespace(id=Tensor(ids),xyxy=Tensor(boxes)))]

class Pipeline(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.video=self.root/'input.mp4'
        writer=cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
        self.assertTrue(writer.isOpened())
        for _ in range(5): writer.write(np.zeros((120,160,3),dtype=np.uint8))
        writer.release()
    def tearDown(self): self.temp.cleanup()
    def run_sequence(self, sequence, points=None, gap_seconds=.5):
        FakeModel.sequence=sequence
        with patch.dict(sys.modules,{'ultralytics':SimpleNamespace(YOLO=FakeModel)}):
            return process(self.video,self.root/'run',1,3,[20,20,40,80],'fake','cpu',.25,points,gap_seconds=gap_seconds)
    def test_output_and_heatmap(self):
        summary=self.run_sequence([([7],[[20,20,40,80]])]*3,[[0,0],[159,0],[159,119],[0,119]])
        self.assertEqual(summary['processed_frames'],3)
        self.assertEqual(summary['heatmap_samples'],3)
        self.assertTrue((self.root/'run'/'heatmap.png').exists())
        self.assertEqual(metadata(self.root/'run'/summary['video'])['frames'],3)
        with (self.root/'run'/'positions.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(rows[0]['frame'],'1')
        self.assertEqual(rows[0]['seconds'],'0.1')
    def test_brief_missing_same_id_recovers_without_filling_gap(self):
        summary=self.run_sequence([([7],[[20,20,40,80]]),([],[]),([7],[[20,20,40,80]])])
        self.assertIsNone(summary['first_loss_frame'])
        self.assertEqual(summary['tracked_frames'],2)
        self.assertEqual(summary['recovered_gaps'],1)
        with (self.root/'run'/'positions.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(rows[1]['foot_x'],'')
        self.assertEqual(rows[-1]['status'],'tracked_unverified')
    def test_missing_timeout_does_not_adopt_other_id(self):
        summary=self.run_sequence([([7],[[20,20,40,80]]),([8],[[20,20,40,80]]),([8],[[20,20,40,80]])],gap_seconds=.1)
        self.assertEqual(summary['first_loss_frame'],3)
        self.assertEqual(summary['first_loss_reason'],'selected_tracker_id_missing_timeout')
        self.assertEqual(summary['tracked_frames'],1)
        with (self.root/'run'/'all_tracks.csv').open() as f: saved=list(csv.DictReader(f))
        self.assertEqual([r['tracker_id'] for r in saved],['7','8','8'])
    def test_suspicious_jump_stays_lost(self):
        summary=self.run_sequence([([7],[[20,20,40,80]]),([7],[[100,20,120,80]]),([7],[[20,20,40,80]])])
        self.assertEqual(summary['first_loss_reason'],'implausible_box_motion_or_scale')
        self.assertEqual(summary['tracked_frames'],1)
    def test_overlap_stops_identity(self):
        summary=self.run_sequence([([7],[[20,20,40,80]]),([7,8],[[20,20,40,80],[22,20,42,80]]),([7],[[20,20,40,80]])])
        self.assertEqual(summary['first_loss_frame'],2)
        self.assertEqual(summary['first_loss_reason'],'overlap_with_another_player')
    def test_ambiguous_selection_rejected(self):
        with self.assertRaises(ValueError):
            self.run_sequence([([7,8],[[20,20,40,80],[22,20,42,80]])]*3)
    def test_bad_calibration(self):
        for points in ([[0,0],[159,119],[159,0],[0,119]], [[0,0]]*4, [[-1,0],[159,0],[159,119],[0,119]]):
            with self.assertRaises(ValueError): calibration(points,160,120)
    def test_bad_video_and_frame(self):
        with self.assertRaises(ValueError): metadata(self.root/'absent.mp4')
        with self.assertRaises(ValueError): frame_at(self.video,100)

if __name__=='__main__': unittest.main()
