import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig
from src.detector import ObjectDetector, Detection

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        # Use a small model for testing
        self.config.model.name = "yolo11n.pt"
        self.config.model.confidence_threshold = 0.25
        self.detector = ObjectDetector(self.config)

    def test_initialization(self):
        """Test if detector initializes correctly"""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.conf_threshold, 0.25)

    def test_empty_frame_detection(self):
        """Test detection on a black frame (should find nothing)"""
        # Create a black 640x640 frame
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(frame)
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 0)

    def test_detection_structure(self):
        """Test if Detection dataclass works as expected"""
        det = Detection(
            bbox=[0.0, 0.0, 10.0, 10.0],
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        self.assertEqual(det.class_name, "car")
        self.assertEqual(len(det.bbox), 4)

if __name__ == '__main__':
    unittest.main()
