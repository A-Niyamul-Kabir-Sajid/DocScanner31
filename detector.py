"""Document detector (OpenCV contour pass).

Returns a coarse bounding box ``(x, y, w, h)`` for the most likely document in
the frame, or ``None`` if nothing usable was found.  The bbox is intentionally
generous; :class:`corner_refiner.CornerRefiner` tightens it into a
quadrilateral afterwards.

History: this used to run YOLOv8n first and fall back to the contour pass.  The
YOLO half was removed -- the bundled weights failed to load on the Pi
(``bad marshal data``), so every frame paid for an exception and then used this
contour path anyway.  Dropping it removes the ``ultralytics``/``torch``
dependency (a very large install on a Raspberry Pi) with no change in
behaviour.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from config import DOC_MIN_AREA_RATIO

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


class DocumentDetector:
    """Coarse document localiser built on a Canny + contour pass."""

    def __init__(
        self,
        *,
        min_area_ratio: float = DOC_MIN_AREA_RATIO,
    ) -> None:
        self.min_area_ratio = min_area_ratio

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        """Return the largest document bbox in ``frame_bgr`` or ``None``."""
        h, w = frame_bgr.shape[:2]
        min_area = self.min_area_ratio * w * h
        return self._detect_contour(frame_bgr, min_area)

    # ------------------------------------------------------------------ #
    def _detect_contour(self, frame_bgr: np.ndarray, min_area: float) -> Optional[BBox]:
        """Coarse document bbox from a Canny + contour pass.

        The bbox is intentionally generous; the corner refiner tightens it
        later.
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
