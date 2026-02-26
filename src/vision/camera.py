"""Camera capture module with webcam and video file support."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import CameraConfig

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """A captured camera frame with metadata."""

    image: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape


class Camera:
    """Camera capture wrapper supporting webcam and video file input.

    Args:
        config: Camera configuration.
        source: Video file path. If None, uses webcam device_id from config.
    """

    def __init__(self, config: CameraConfig, source: str | None = None) -> None:
        self.config = config
        self._source = source if source is not None else config.device_id
        self._cap: cv2.VideoCapture | None = None
        self._frame_count = 0
        self._fps_target = config.fps
        self._last_frame_time = 0.0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> bool:
        """Open the camera or video source.

        Returns:
            True if opened successfully.
        """
        self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            logger.error("Failed to open camera source: %s", self._source)
            return False

        # Set resolution for webcam (not video files)
        if isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera opened: %dx%d from source %s", actual_w, actual_h, self._source)

        return True

    def read(self) -> Frame | None:
        """Read a single frame from the camera.

        Returns:
            Frame object or None if read failed.
        """
        if not self.is_open:
            return None

        # FPS limiting
        now = time.time()
        min_interval = 1.0 / self._fps_target
        elapsed = now - self._last_frame_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        ret, image = self._cap.read()
        if not ret or image is None:
            return None

        self._frame_count += 1
        self._last_frame_time = time.time()

        # Flip horizontally for mirror effect (webcam only)
        if isinstance(self._source, int):
            image = cv2.flip(image, 1)

        height, width = image.shape[:2]
        return Frame(
            image=image,
            timestamp=self._last_frame_time,
            frame_number=self._frame_count,
            width=width,
            height=height,
        )

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()
