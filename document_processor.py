"""Document processing pipeline.

Implements the spec's exact 15-step pipeline per captured page:

    1. Capture original COLOR frame.
    2. Convert to grayscale.
    3. Apply Gaussian Blur.
    4. Apply CLAHE for contrast enhancement.
    5. Apply automatic Canny Edge Detection.
    6. Apply dilation followed by erosion.
    7. Detect document contour.
    8. If YOLO is enabled: detect document first, crop ROI, refine corners with OpenCV.
    9. Reorder corner points.
   10. Apply Perspective Warp.
   11. Crop 20 pixels from borders.
   12. Resize to A4 dimensions.
   13. Optional shadow removal.
   14. Optional sharpening.
   15. Save final COLOR scanned page.

Only processed pages are allowed into the PDF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    A4_HEIGHT_PX,
    A4_WIDTH_PX,
    DOC_BORDER_CROP_PX,
    DOC_MIN_AREA_RATIO,
    ENABLE_YOLO,
    SCAN_MODE,
    SHADOW_REMOVAL,
    SHARPEN,
)

logger = logging.getLogger(__name__)

Point = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectionResult:
    """Result of document detection on a single frame."""

    corners: Optional[np.ndarray]  # 4x2 int array in source-frame coordinates
    confidence: float              # 0.0 .. 1.0
    used_yolo: bool
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) or None


# --------------------------------------------------------------------------- #
# DocumentProcessor
# --------------------------------------------------------------------------- #
class DocumentProcessor:
    """Single-page document processing pipeline.

    The processor is intentionally stateless: feed it a BGR frame, get back a
    processed page (still in BGR color) plus a :class:`DetectionResult` so the
    caller can decide whether the page is good enough to accept.
    """

    def __init__(
        self,
        *,
        scan_mode: str = SCAN_MODE,
        border_crop_px: int = DOC_BORDER_CROP_PX,
        target_width: int = A4_WIDTH_PX,
        target_height: int = A4_HEIGHT_PX,
        min_area_ratio: float = DOC_MIN_AREA_RATIO,
        enable_yolo: bool = ENABLE_YOLO,
        shadow_removal: bool = SHADOW_REMOVAL,
        sharpen: bool = SHARPEN,
        detector=None,
        corner_refiner=None,
    ) -> None:
        self.scan_mode = scan_mode
        self.border_crop_px = border_crop_px
        self.target_width = target_width
        self.target_height = target_height
        self.min_area_ratio = min_area_ratio
        self.enable_yolo = enable_yolo
        self.shadow_removal = shadow_removal
        self.sharpen = sharpen

        # Lazy imports so the camera/scanner still runs without ultralytics.
        if detector is None:
            from detector import DocumentDetector  # noqa: WPS433 (local import)

            detector = DocumentDetector(enable_yolo=self.enable_yolo)
        self.detector = detector

        if corner_refiner is None:
            from corner_refiner import CornerRefiner  # noqa: WPS433

            corner_refiner = CornerRefiner(min_area_ratio=self.min_area_ratio)
        self.corner_refiner = corner_refiner

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        """Run the full pipeline on a BGR frame.

        Returns ``(processed_page_bgr, detection_result)``.  When the
        document cannot be localised the function falls back to a centered
        A4 crop of the original frame so the user always sees *something*
        on screen, but ``DetectionResult.confidence`` will be ``0.0``.
        """
        return self.process_with_debug(frame_bgr)[:2]

    def process_with_debug(self, frame_bgr: np.ndarray):
        """Run the pipeline and return every intermediate frame.

        Returns a tuple of
        ``(processed, detection, gray, edges, contour_overlay,
        biggest_contour_overlay, warped, warped_gray, adaptive)`` so the
        UI can stream an 8-panel debug grid without re-running the steps.
        Each frame is a 3-channel BGR image (single-channel stages are
        converted for uniform handling).  All frames are the same height
        as the input.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("process_with_debug() received an empty frame")

        h, w = frame_bgr.shape[:2]
        gray = self._to_grayscale(frame_bgr)
        blurred = self._gaussian_blur(gray)
        contrast = self._clahe(blurred)
        edges = self._auto_canny(contrast)
        closed = self._dilate_then_erode(edges)

        detection = self._detect_corners(frame_bgr, gray, closed, w, h)
        corners = detection.corners

        # All-contours overlay: every edge contour is drawn in green on
        # the original frame so the user can see what the detector saw.
        contour_overlay = frame_bgr.copy()
        try:
            all_contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(contour_overlay, all_contours, -1, (0, 255, 0), 2)
        except Exception:  # pragma: no cover - defensive
            pass

        # Biggest-contour overlay: only the four corners used for warping.
        biggest_overlay = frame_bgr.copy()
        if corners is not None:
            cv2.polylines(
                biggest_overlay,
                [corners.astype(int).reshape(-1, 1, 2)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3,
            )
            for (cx, cy) in corners.astype(int):
                cv2.circle(biggest_overlay, (int(cx), int(cy)), 8, (0, 0, 255), -1)

        if corners is None:
            processed = self._fallback_center_crop(frame_bgr)
            warped_bgr: Optional[np.ndarray] = None
        else:
            warped = self._perspective_warp(frame_bgr, corners, w, h)
            cropped = self._border_crop(warped)
            sized = self._resize_a4(cropped)
            cleaned = self._remove_shadow(sized) if self.shadow_removal else sized
            sharpened = self._sharpen(cleaned) if self.sharpen else cleaned
            processed = self._apply_scan_mode(sharpened)
            warped_bgr = sized  # color warp before scan-mode tinting

        warped_gray_bgr: Optional[np.ndarray] = None
        adaptive_bgr: Optional[np.ndarray] = None
        if warped_bgr is not None:
            warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
            warped_gray_bgr = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
            adaptive = cv2.adaptiveThreshold(
                warped_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                10,
            )
            adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

        return (
            processed,
            detection,
            gray,
            edges,
            contour_overlay,
            biggest_overlay,
            warped_bgr,
            warped_gray_bgr,
            adaptive_bgr,
        )

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        # Step 2: grayscale.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _gaussian_blur(gray: np.ndarray) -> np.ndarray:
        # Step 3: 5x5 Gaussian blur.  Kernel must be odd; bigger blurs lose
        # thin edges so 5 is the standard for document scanning.
        return cv2.GaussianBlur(gray, (5, 5), 0)

    @staticmethod
    def _clahe(gray: np.ndarray) -> np.ndarray:
        # Step 4: CLAHE for local contrast (tile size 8x8, clip 2.0).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _auto_canny(gray: np.ndarray) -> np.ndarray:
        # Step 5: auto Canny using median-based thresholds with a floor
        # so bright / low-contrast frames still produce useful edges.
        v = float(np.median(gray))
        lower = int(max(30, 0.5 * v))
        upper = int(min(255, max(lower + 40, 1.5 * v)))
        return cv2.Canny(gray, lower, upper)

    @staticmethod
    def _dilate_then_erode(edges: np.ndarray) -> np.ndarray:
        # Step 6: dilate to close gaps, then erode to recover edges.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        return eroded

    def _detect_corners(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        edges: np.ndarray,
        width: int,
        height: int,
    ) -> DetectionResult:
        # Steps 7 + 8: YOLO first (if enabled), then OpenCV contour refinement.
        bbox = None
        confidence = 0.0
        used_yolo = False
        corners: Optional[np.ndarray] = None

        if self.enable_yolo:
            try:
                bbox = self.detector.detect(frame_bgr)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("YOLO detection failed: %s", exc)
                bbox = None
            if bbox is not None:
                used_yolo = True
                x, y, w, h = bbox
                # Guard against out-of-bounds bbox values from the detector.
                x = max(0, min(int(x), width))
                y = max(0, min(int(y), height))
                w = max(1, min(int(w), width - x))
                h = max(1, min(int(h), height - y))
                roi = frame_bgr[y : y + h, x : x + w]
                if roi.size > 0:
                    corners, confidence = self.corner_refiner.refine(
                        roi,
                        frame_shape=(height, width),
                        roi_offset=(x, y),
                    )

        if corners is None:
            # Step 7 fallback: contour detection on the closed edge map.
            corners, confidence = self.corner_refiner.from_edges(edges, width, height)

        return DetectionResult(
            corners=corners,
            confidence=float(confidence),
            used_yolo=used_yolo,
            bbox=bbox,
        )

    @staticmethod
    def _perspective_warp(
        frame_bgr: np.ndarray,
        corners: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        # Step 9 + 10: corners are already in (tl, tr, br, bl) order from
        # CornerRefiner.  Compute the output size so the document keeps its
        # aspect ratio and remains fully visible.
        tl, tr, br, bl = corners.astype(np.float32)
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_w = int(max(width_a, width_b))
        max_h = int(max(height_a, height_b))
        max_w = max(max_w, 1)
        max_h = max(max_h, 1)

        dst = np.array(
            [
                [0, 0],
                [max_w - 1, 0],
                [max_w - 1, max_h - 1],
                [0, max_h - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warped = cv2.warpPerspective(frame_bgr, matrix, (max_w, max_h))
        # Clip to the original frame bounds as a safety net.
        warped = warped[: max(0, height), : max(0, width)]
        return warped

    def _border_crop(self, warped: np.ndarray) -> np.ndarray:
        # Step 11: crop 20 pixels from every border.
        b = self.border_crop_px
        if warped.shape[0] <= 2 * b or warped.shape[1] <= 2 * b:
            return warped
        return warped[b:-b, b:-b]

    def _resize_a4(self, cropped: np.ndarray) -> np.ndarray:
        # Step 12: resize to A4 dimensions (portrait by default).
        return cv2.resize(
            cropped,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _remove_shadow(image: np.ndarray) -> np.ndarray:
        # Step 13 (optional): morphological close on a large kernel and
        # divide to flatten background illumination.
        if image.ndim == 2:
            channels = [image]
        else:
            channels = cv2.split(image)
        cleaned_channels: List[np.ndarray] = []
        for ch in channels:
            dilated = cv2.dilate(ch, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(ch, bg)
            normalised = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            cleaned_channels.append(normalised)
        if image.ndim == 2:
            return cleaned_channels[0]
        return cv2.merge(cleaned_channels)

    @staticmethod
    def _sharpen(image: np.ndarray) -> np.ndarray:
        # Step 14 (optional): unsharp mask.
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    def _apply_scan_mode(self, image: np.ndarray) -> np.ndarray:
        # Step 15: still save COLOR by default; the spec only binarises for bw mode.
        if self.scan_mode == "color":
            return image
        if self.scan_mode == "grayscale":
            if image.ndim == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        if self.scan_mode == "bw":
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                10,
            )
        raise ValueError(f"Unknown scan mode: {self.scan_mode!r}")

    # ------------------------------------------------------------------ #
    # Fallback when detection fails
    # ------------------------------------------------------------------ #
    def _fallback_center_crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Center-crop the frame to a portrait ratio and resize to A4.

        Used when no quadrilateral can be located - the user still sees a
        preview, and the quality gate will (correctly) reject the capture.
        """
        h, w = frame_bgr.shape[:2]
        target_ratio = self.target_height / self.target_width
        if h / max(w, 1) > target_ratio:
            new_w = int(h / target_ratio)
            x0 = max(0, (w - new_w) // 2)
            cropped = frame_bgr[:, x0 : x0 + new_w]
        else:
            new_h = int(w * target_ratio)
            y0 = max(0, (h - new_h) // 2)
            cropped = frame_bgr[y0 : y0 + new_h, :]
        return cv2.resize(
            cropped,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )

    # ------------------------------------------------------------------ #
    # Drawing helpers (used by app.py overlay)
    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_overlay(
        frame_bgr: np.ndarray,
        detection: DetectionResult,
        processed_preview: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw the bbox, corners and confidence on a copy of ``frame_bgr``."""
        canvas = frame_bgr.copy()
        if detection.bbox is not None:
            x, y, w, h = detection.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if detection.corners is not None:
            for (cx, cy) in detection.corners.astype(int):
                cv2.circle(canvas, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.polylines(
                canvas,
                [detection.corners.astype(int).reshape(-1, 1, 2)],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
        label = f"conf {detection.confidence:.2f}"
        if detection.used_yolo:
            label = "YOLO + " + label
        cv2.putText(
            canvas,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if processed_preview is not None:
            h, w = canvas.shape[:2]
            ph, pw = processed_preview.shape[:2]
            scale = min(180 / pw, 180 / ph)
            if scale > 0:
                small = cv2.resize(
                    processed_preview,
                    (max(1, int(pw * scale)), max(1, int(ph * scale))),
                )
                if small.ndim == 2:
                    small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
                x0 = w - small.shape[1] - 10
                y0 = h - small.shape[0] - 10
                canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
                cv2.rectangle(
                    canvas,
                    (x0, y0),
                    (x0 + small.shape[1], y0 + small.shape[0]),
                    (255, 255, 255),
                    1,
                )
        return canvas
