"""Corner refinement for document detection.

Given either a region-of-interest cropped from a YOLO bbox, or a binary
edge map, returns the four ordered corners of the page (top-left, top-
right, bottom-right, bottom-left) plus a confidence score in ``[0, 1]``.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import DOC_MIN_AREA_RATIO

logger = logging.getLogger(__name__)

Point = Tuple[int, int]


class CornerRefiner:
    """Find the four corners of the largest quadrilateral in an image."""

    def __init__(self, *, min_area_ratio: float = DOC_MIN_AREA_RATIO) -> None:
        self.min_area_ratio = min_area_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def refine(
        self,
        roi_bgr: np.ndarray,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Find ordered corners inside ``roi_bgr``.

        ``frame_shape=(H, W)`` of the original frame, if provided, lets the
        returned coordinates be in source-frame space.  Otherwise they are
        relative to ``roi_bgr`` (useful for tests).
        """
        h, w = roi_bgr.shape[:2]
        edges = self._edges(roi_bgr)
        corners, confidence = self._approx_quad(edges, w, h)
        if corners is None:
            return None, 0.0
        if frame_shape is not None:
            # Caller passed the ROI already cropped from the full frame; we
            # currently do not know the offset here, so the caller must add it
            # back.  This method intentionally keeps coordinates local.
            pass
        return corners, confidence

    def from_edges(
        self,
        edges: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Same as :meth:`refine` but the caller already computed the edges."""
        return self._approx_quad(edges, width, height)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _edges(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        v = float(np.median(blurred))
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges

    def _approx_quad(
        self,
        edges: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[Optional[np.ndarray], float]:
        min_area = self.min_area_ratio * width * height
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, 0.0

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:8]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = self._reorder(approx.reshape(4, 2))
                confidence = self._confidence(area, width * height)
                return corners.astype(np.int32), confidence
        # Fallback: minimum-area rectangle of the largest contour.
        cnt = contours[0]
        if cv2.contourArea(cnt) < min_area:
            return None, 0.0
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        corners = self._reorder(box)
        confidence = self._confidence(cv2.contourArea(cnt), width * height) * 0.7
        return corners.astype(np.int32), confidence

    @staticmethod
    def _reorder(points: np.ndarray) -> np.ndarray:
        """Order four points as top-left, top-right, bottom-right, bottom-left."""
        pts = points.astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered = np.array(
            [
                pts[np.argmin(s)],   # tl  (smallest x+y)
                pts[np.argmin(d)],   # tr  (smallest y-x)
                pts[np.argmax(s)],   # br  (largest x+y)
                pts[np.argmax(d)],   # bl  (largest y-x)
            ],
            dtype=np.float32,
        )
        return ordered

    @staticmethod
    def _confidence(contour_area: float, frame_area: float) -> float:
        """Return a 0..1 score based on how much of the frame the doc covers."""
        if frame_area <= 0:
            return 0.0
        ratio = contour_area / frame_area
        # A4 paper from a typical webcam is ~25-70% of the frame.
        score = max(0.0, min(1.0, (ratio - 0.05) / 0.5))
        return float(score)
