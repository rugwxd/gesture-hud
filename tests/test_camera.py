"""Tests for camera capture module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.config import CameraConfig
from src.vision.camera import Camera, Frame


class TestFrame:
    def test_frame_creation(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=image, timestamp=1.0, frame_number=1, width=640, height=480)
        assert frame.width == 640
        assert frame.height == 480
        assert frame.frame_number == 1
        assert frame.shape == (480, 640, 3)


class TestCamera:
    def test_init_defaults(self):
        config = CameraConfig()
        camera = Camera(config)
        assert not camera.is_open

    def test_init_with_source(self):
        config = CameraConfig()
        camera = Camera(config, source="test.mp4")
        assert not camera.is_open

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_open_success(self, mock_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_capture.return_value = mock_cap

        config = CameraConfig()
        camera = Camera(config)
        assert camera.open() is True
        assert camera.is_open

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_open_failure(self, mock_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        config = CameraConfig()
        camera = Camera(config)
        assert camera.open() is False

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_read_frame(self, mock_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_image)
        mock_capture.return_value = mock_cap

        config = CameraConfig(fps=1000)  # High FPS to avoid sleep
        camera = Camera(config)
        camera.open()
        frame = camera.read()

        assert frame is not None
        assert frame.width == 640
        assert frame.height == 480
        assert frame.frame_number == 1

    def test_read_without_open(self):
        config = CameraConfig()
        camera = Camera(config)
        assert camera.read() is None

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_release(self, mock_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_capture.return_value = mock_cap

        config = CameraConfig()
        camera = Camera(config)
        camera.open()
        camera.release()
        assert not camera.is_open

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_context_manager(self, mock_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_capture.return_value = mock_cap

        config = CameraConfig()
        with Camera(config) as cam:
            assert cam.is_open
