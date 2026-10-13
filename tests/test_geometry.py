import unittest
from geometry import iou, plausible

class IdentityGates(unittest.TestCase):
    def test_unique_initial_overlap(self):
        selected=[10,10,30,80]
        self.assertGreater(iou(selected,[11,11,31,81]),.5)
        self.assertEqual(iou(selected,[100,10,120,80]),0)
    def test_motion_rejection(self):
        self.assertTrue(plausible([0,0,20,80],[5,1,25,81]))
        self.assertFalse(plausible([0,0,20,80],[150,0,170,80]))
        self.assertFalse(plausible([0,0,20,80],[0,0,20,20]))

if __name__=='__main__': unittest.main()
