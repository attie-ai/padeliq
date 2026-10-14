"""Exercise the human confirmation and stale-export controls in Streamlit."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from streamlit.testing.v1 import AppTest
import test_review

class ReviewUI(unittest.TestCase):
    setUp = test_review.ManualReview.setUp
    tearDown = test_review.ManualReview.tearDown
    # Reuse a synthetic recording and saved tracks, without running YOLO.
    def test_review_requires_confirmation_and_invalidates_combined_result(self):
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root))
        summary=dict(self.summary,video='none.mp4',browser_playable=False,first_loss_frame=2,first_loss_reason='selected_tracker_id_missing_timeout')
        self.run.joinpath('summary.json').write_text(json.dumps(summary))
        app=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
        app.text_input[0].set_value(str(self.source)).run()
        app.session_state['run']=(str(self.run),summary)
        app.run()
        self.assertEqual(len(app.exception),0)
        choose=next(x for x in app.selectbox if x.label=='Choose a tracker ID to review')
        choose.select(110).run()
        add=next(x for x in app.button if x.label=='Add confirmed interval')
        self.assertTrue(add.disabled)
        next(x for x in app.checkbox if x.label.startswith('I inspected this exact')).check().run()
        next(x for x in app.button if x.label=='Add confirmed interval').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state['review_intervals'],[dict(tracker_id=110,start_frame=3,end_frame=4)])
        self.assertEqual(json.loads((self.run/'review_intervals.json').read_text()),app.session_state['review_intervals'])
        del app.session_state['review_run_key']
        app.run()
        self.assertEqual(app.session_state['review_intervals'],[dict(tracker_id=110,start_frame=3,end_frame=4)])
        next(x for x in app.button if x.label=='Build combined reviewed result').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state['combined_result'][1]['confirmed_frames'],2)
        next(x for x in app.button if x.label=='Remove this interval').click().run()
        self.assertEqual(app.session_state['review_intervals'],[])
        self.assertNotIn('combined_result',app.session_state)
