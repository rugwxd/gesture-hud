"""YOLOv8 object detection pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.config import DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "label": self.label,
        }


class ObjectDetector:
    """YOLOv8-based object detector with frame skipping for performance.

    Runs detection on every Nth frame and caches results between runs.
    Uses YOLOv8 nano model for real-time performance.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._model = None
        self._initialized = False
        self._frame_counter = 0
        self._cached_detections: list[Detection] = []

    def initialize(self) -> bool:
        """Load the YOLOv8 model. Lazy init to avoid slow import at startup."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model)
            self._initialized = True
            logger.info("Object detector initialized with model: %s", self.config.model)
            return True
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as exc:
            logger.error("Failed to load model %s: %s", self.config.model, exc)
            return False

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a frame.

        Runs detection only every N frames (configured by run_every_n_frames).
        Returns cached results on skipped frames.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of Detection objects.
        """
        self._frame_counter += 1

        # Skip frames for performance
        if self._frame_counter % self.config.run_every_n_frames != 0:
            return self._cached_detections

        if not self._initialized:
            if not self.initialize():
                return []

        try:
            results = self._model(frame, verbose=False, conf=self.config.confidence_threshold)

            detections: list[Detection] = []
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names.get(cls_id, f"class_{cls_id}")

                    detections.append(
                        Detection(
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                            confidence=conf,
                            class_id=cls_id,
                            label=label,
                        )
                    )

            self._cached_detections = detections
            return detections

        except Exception as exc:
            logger.error("Detection error: %s", exc)
            return self._cached_detections

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def release(self) -> None:
        """Release model resources."""
        self._model = None
        self._initialized = False
