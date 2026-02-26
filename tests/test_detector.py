"""Tests for object detection module."""

from __future__ import annotations

import numpy as np

from src.config import DetectionConfig
from src.vision.detector import Detection, ObjectDetector


class TestDetection:
    def test_creation(self):
        det = Detection(
            x1=100, y1=100, x2=200, y2=300,
            confidence=0.85, class_id=0, label="person",
        )
        assert det.label == "person"
        assert det.confidence == 0.85

    def test_center(self):
        det = Detection(x1=100, y1=100, x2=200, y2=300, confidence=0.9, class_id=0, label="person")
        assert det.center == (150, 200)

    def test_dimensions(self):
        det = Detection(x1=100, y1=100, x2=300, y2=400, confidence=0.9, class_id=0, label="test")
        assert det.width == 200
        assert det.height == 300

    def test_to_dict(self):
        det = Detection(x1=10, y1=20, x2=30, y2=40, confidence=0.5, class_id=1, label="cat")
        d = det.to_dict()
        assert d["label"] == "cat"
        assert d["x1"] == 10
        assert d["confidence"] == 0.5


class TestObjectDetector:
    def test_init(self):
        config = DetectionConfig()
        detector = ObjectDetector(config)
        assert not detector.is_initialized

    def test_frame_skipping(self):
        config = DetectionConfig(run_every_n_frames=3)
        detector = ObjectDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Without initialization, detect returns empty
        result = detector.detect(frame)
        assert result == []

    def test_cached_detections_returned_on_skip(self):
        config = DetectionConfig(run_every_n_frames=3)
        detector = ObjectDetector(config)

        # Set some cached detections
        cached = [Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0, label="test")]
        detector._cached_detections = cached

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: not divisible by 3, returns cached
        result = detector.detect(frame)
        assert len(result) == 1

    def test_release(self):
        config = DetectionConfig()
        detector = ObjectDetector(config)
        detector.release()
        assert not detector.is_initialized
