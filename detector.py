"""Document detector (YOLOv8n + OpenCV contour fallback).

Returns a bounding box ``(x, y, w, h)`` for the most likely document in the
frame, or ``None`` if nothing usable was found.  The corner refiner tightens
the box into a quadrilateral.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import DOC_MIN_AREA_RATIO, ENABLE_YOLO, YOLO_CONFIDENCE, YOLO_MODEL_PATH

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


class DocumentDetector:
    """YOLOv8n document localiser with a hard fallback to ``None``."""

    def __init__(
        self,
        *,
        enable_yolo: bool = ENABLE_YOLO,
        weights_path: Optional[Path] = None,
        confidence: float = YOLO_CONFIDENCE,
        min_area_ratio: float = DOC_MIN_AREA_RATIO,
    ) -> None:
        self.enable_yolo = enable_yolo
        self.weights_path = Path(weights_path) if weights_path is not None else YOLO_MODEL_PATH
        self.confidence = confidence
        self.min_area_ratio = min_area_ratio
        self._model = None
        self._init_error: Optional[str] = None

        if self.enable_yolo:
            self._try_load_yolo()

    # ------------------------------------------------------------------ #
    def _try_load_yolo(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            if not self.weights_path.exists():
                self._init_error = f"weights not found: {self.weights_path}"
                logger.warning("YOLO weights missing at %s", self.weights_path)
                return
            self._model = YOLO(str(self.weights_path))
            logger.info("Loaded YOLOv8n from %s", self.weights_path)
        except Exception as exc:  # pragma: no cover - defensive
            self._init_error = str(exc)
            logger.warning("Could not load YOLOv8n: %s", exc)

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        """Return the largest document bbox in ``frame_bgr`` or ``None``."""
        h, w = frame_bgr.shape[:2]
        min_area = self.min_area_ratio * w * h
        bbox = self._detect_yolo(frame_bgr, min_area)
        if bbox is not None:
            return bbox
        return self._detect_contour(frame_bgr, min_area)

    # ------------------------------------------------------------------ #
    def _detect_yolo(self, frame_bgr: np.ndarray, min_area: float) -> Optional[BBox]:
        if self._model is None:
            return None
        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=self.confidence,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("YOLO inference failed: %s", exc)
            return None
        if not results:
            return None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        best_box: Optional[BBox] = None
        best_area = 0.0
        for xyxy in boxes.xyxy.cpu().numpy():
            x0, y0, x1, y1 = (int(v) for v in xyxy)
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(frame_bgr.shape[1], x1)
            y1 = min(frame_bgr.shape[0], y1)
            bw = max(0, x1 - x0)
            bh = max(0, y1 - y0)
            area = bw * bh
            if area >= min_area and area > best_area:
                best_area = area
                best_box = (x0, y0, bw, bh)
        return best_box

    # ------------------------------------------------------------------ #
    def _detect_contour(self, frame_bgr: np.ndarray, min_area: float) -> Optional[BBox]:
        """Coarse document bbox from a Canny + contour pass.

        Used as a fallback when YOLO is missing or misses.  The bbox is
        intentionally generous; the corner refiner tightens it later.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:5]:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area:
                return (x, y, w, h)
        return None
