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
# Quad is a 4x2 int array: rows are (tl, tr, br, bl), columns are (x, y).
# Kept as a type alias so ``auto_capture_controller`` and ``stability_tracker``
# can keep importing ``Quad`` from this module.
Quad = np.ndarray


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
        roi_offset: Optional[Tuple[int, int]] = (0, 0),
    ) -> Tuple[Optional[np.ndarray], float]:
        """Find ordered corners inside ``roi_bgr``.

        Returns corners in the **source-frame** coordinate system.  Pass
        ``roi_offset=(x, y)`` (the top-left of the ROI inside the source
        frame) so the returned corners are translated back into frame space;
        ``frame_shape=(H, W)`` is used only to clip corners to the frame.
        """
        h, w = roi_bgr.shape[:2]
        edges = self._edges(roi_bgr)
        corners, confidence = self._approx_quad(edges, w, h)
        if corners is None:
            return None, 0.0

        # Translate from ROI-local to source-frame coordinates.
        if roi_offset is not None:
            ox, oy = int(roi_offset[0]), int(roi_offset[1])
            corners = corners.astype(np.float32)
            corners[:, 0] += ox
            corners[:, 1] += oy

        # Optionally clip to the frame so the corners stay inside the image.
        if frame_shape is not None:
            fh, fw = int(frame_shape[0]), int(frame_shape[1])
            corners[:, 0] = np.clip(corners[:, 0], 0, fw - 1)
            corners[:, 1] = np.clip(corners[:, 1], 0, fh - 1)

        return corners.astype(np.int32), float(confidence)

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
        # Use a *floor* on the lower threshold so flat / bright frames (which
        # otherwise produce a near-empty edge map) still get useful edges.
        lower = int(max(20, 0.5 * v))
        upper = int(min(255, max(lower + 30, 1.33 * v)))
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
        # Lower fallback threshold: 2% of the frame. We still produce a quad
        # for the largest contour even if it is small so the user always
        # gets *something* on the wire - the confidence score tells the
        # quality gate how good it was.
        soft_min_area = max(1.0, 0.02 * width * height)
        for cnt in contours[:8]:
            area = cv2.contourArea(cnt)
            if area < soft_min_area:
                continue
            # Collapse noisy contour wiggles into the outer boundary *before*
            # running approxPolyDP.  A doc outline with 1641 raw points will
            # never reduce to 4 corners with polyDP alone because small
            # zig-zags keep stealing vertices.  convexHull gives the true
            # 4-corner polygon we want.
            try:
                hull = cv2.convexHull(cnt)
            except cv2.error:
                hull = cnt
            hull_peri = cv2.arcLength(hull, True)
            # Try progressively larger epsilons so a 10-pt hull can still
            # collapse to a clean quad.
            for eps_factor in (0.02, 0.04, 0.06, 0.08, 0.12, 0.18):
                approx = cv2.approxPolyDP(hull, eps_factor * hull_peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    corners = self._reorder(approx.reshape(4, 2))
                    confidence = self._confidence(area, width * height)
                    return corners.astype(np.int32), confidence
                if len(approx) < 4:
                    break
        # Fallback: minimum-area rectangle of the largest contour.
        cnt = contours[0]
        if cv2.contourArea(cnt) < soft_min_area:
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
