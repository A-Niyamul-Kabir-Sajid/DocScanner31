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

    def refine_inner_page(
        self,
        frame_bgr: np.ndarray,
        outer_corners: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Given the *outer* corners of a book/cover, find the inner page
        rectangle (the white area inside a dark cover/border).

        ``outer_corners`` must be 4 points in (TL, TR, BR, BL) order.
        Returns the inner 4 corners in the same order, plus a confidence.

        Algorithm:
            1. Warp the frame using outer_corners into a flat rectangle.
            2. Build a binary mask of bright pixels (>=BRIGHT_THRESHOLD).
            3. Find the largest bright contour.
            4. approxPolyDP that bright contour down to a quad.
            5. Map that quad back to source-frame coordinates via the
               inverse of the outer warp matrix.

        If no inner quad is found, returns (None, 0.0) so the caller can
        decide to keep the outer quad instead.
        """
        import logging  # local import to avoid touching module-level
        logger = logging.getLogger(__name__)

        outer = outer_corners.astype(np.float32)
        tl, tr, br, bl = outer

        # 1) compute outer warp rectangle size
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        out_w = int(max(width_a, width_b))
        out_h = int(max(height_a, height_b))
        if out_w < 20 or out_h < 20:
            return None, 0.0

        dst = np.array(
            [
                [0, 0],
                [out_w - 1, 0],
                [out_w - 1, out_h - 1],
                [0, out_h - 1],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(outer, dst)
        warped = cv2.warpPerspective(frame_bgr, M, (out_w, out_h))
        if warped is None or warped.size == 0:
            return None, 0.0

        # 2) bright-page mask.  In Lab / grayscale, paper reads > 200.
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
        # Adaptive: top 30% brightest pixels are "paper".
        # This handles off-white paper, sepia paper, yellowed pages.
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = hist.sum()
        if total <= 0:
            return None, 0.0
        cumulative = np.cumsum(hist)
        # Find the gray value where the top 70% of pixels are below it.
        bright_threshold = int(np.searchsorted(cumulative, total * 0.70))
        bright_threshold = max(120, min(bright_threshold, 240))

        mask = (gray >= bright_threshold).astype(np.uint8) * 255

        # 3) close gaps in the bright region so we get a solid rectangle
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)

        bright_contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not bright_contours:
            logger.debug("inner-page: no bright contours found (thr=%d)", bright_threshold)
            return None, 0.0

        bright_contours = sorted(bright_contours, key=cv2.contourArea, reverse=True)
        biggest = bright_contours[0]
        if cv2.contourArea(biggest) < 0.20 * out_w * out_h:
            # bright region is less than 20% of the warp -> not a page
            logger.debug("inner-page: biggest bright area too small: %d", cv2.contourArea(biggest))
            return None, 0.0

        # 4) smooth -> approxPolyDP -> 4-corner convex quad
        try:
            hull = cv2.convexHull(biggest)
        except cv2.error:
            hull = biggest
        peri = cv2.arcLength(hull, True)
        quad_warped = None
        for eps_factor in (0.02, 0.04, 0.06, 0.08, 0.12, 0.18):
            approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad_warped = approx.reshape(4, 2).astype(np.float32)
                break
            if len(approx) < 4:
                break

        if quad_warped is None:
            rect = cv2.minAreaRect(hull)
            quad_warped = cv2.boxPoints(rect).astype(np.float32)

        # 5) map back to source-frame coordinates
        quad_warped = quad_warped.reshape(4, 1, 2)
        M_inv = cv2.getPerspectiveTransform(dst, outer)
        quad_source = cv2.perspectiveTransform(quad_warped, M_inv)
        quad_source = quad_source.reshape(4, 2).astype(np.float32)
        ordered = self._reorder(quad_source)

        # confidence: bright-area ratio within the outer warp
        bright_area = float(cv2.contourArea(biggest))
        conf = self._confidence(bright_area, float(out_w * out_h))
        return ordered.astype(np.int32), float(conf)

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
