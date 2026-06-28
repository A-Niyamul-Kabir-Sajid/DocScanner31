"""Image quality gate.

Runs cheap per-frame checks on the *processed* page before it's appended to
the current document:

    - Blur (Laplacian variance)
    - Brightness (mean grayscale intensity)
    - Motion (mean absolute difference vs the previous processed frame)
    - Document visibility (contour area as a fraction of the frame)
    - Corner confidence (from :mod:`corner_refiner`)

Each failure returns a short reason string; the caller surfaces it on the
LIVE overlay so the user knows why ``C`` was rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    BLUR_MIN_VARIANCE,
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    CORNER_CONFIDENCE_MIN,
    DOC_MIN_AREA_RATIO,
    MOTION_MAX_PX,
    QUALITY_GATE_ENABLED,
)
from document_processor import DetectionResult


@dataclass(frozen=True)
class QualityReport:
    """Outcome of :class:`QualityGate` evaluation."""

    ok: bool
    reason: str = ""
    blur: float = 0.0
    brightness: float = 0.0
    motion: float = 0.0
    corner_confidence: float = 0.0
    document_ratio: float = 0.0


class QualityGate:
    """Decide whether a processed frame should be accepted."""

    def __init__(
        self,
        *,
        enabled: bool = QUALITY_GATE_ENABLED,
        blur_min: float = BLUR_MIN_VARIANCE,
        brightness_min: float = BRIGHTNESS_MIN,
        brightness_max: float = BRIGHTNESS_MAX,
        motion_max: float = MOTION_MAX_PX,
        corner_confidence_min: float = CORNER_CONFIDENCE_MIN,
        min_area_ratio: float = DOC_MIN_AREA_RATIO,
    ) -> None:
        self.enabled = enabled
        self.blur_min = blur_min
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.motion_max = motion_max
        self.corner_confidence_min = corner_confidence_min
        self.min_area_ratio = min_area_ratio
        self._previous_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        processed_bgr: np.ndarray,
        detection: DetectionResult,
        raw_frame_for_motion: Optional[np.ndarray] = None,
    ) -> QualityReport:
        if not self.enabled:
            return QualityReport(ok=True)

        gray = self._to_gray(processed_bgr)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        ref = raw_frame_for_motion if raw_frame_for_motion is not None else processed_bgr
        motion = self._motion(ref)

        document_ratio = 0.0
        if detection.bbox is not None:
            x, y, w, h = detection.bbox
            fh, fw = ref.shape[:2]
            document_ratio = (w * h) / max(1, fh * fw)

        reason = self._first_failure(
            blur=blur,
            brightness=brightness,
            motion=motion,
            document_ratio=document_ratio,
            corner_confidence=detection.confidence,
        )

        if raw_frame_for_motion is not None:
            self._previous_frame = raw_frame_for_motion.copy()

        return QualityReport(
            ok=reason == "",
            reason=reason,
            blur=blur,
            brightness=brightness,
            motion=motion,
            corner_confidence=float(detection.confidence),
            document_ratio=float(document_ratio),
        )

    # ------------------------------------------------------------------ #
    def _first_failure(
        self,
        *,
        blur: float,
        brightness: float,
        motion: float,
        document_ratio: float,
        corner_confidence: float,
    ) -> str:
        if blur < self.blur_min:
            return f"blurry (var={blur:.1f})"
        if brightness < self.brightness_min:
            return f"too dark ({brightness:.0f})"
        if brightness > self.brightness_max:
            return f"too bright ({brightness:.0f})"
        if motion > self.motion_max:
            return f"motion ({motion:.1f}px)"
        if document_ratio < self.min_area_ratio:
            return f"doc too small ({document_ratio:.2f})"
        if corner_confidence < self.corner_confidence_min:
            return f"corners weak ({corner_confidence:.2f})"
        return ""

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _motion(self, frame: np.ndarray) -> float:
        if self._previous_frame is None:
            return 0.0
        prev = self._match_shape(self._previous_frame, frame)
        diff = cv2.absdiff(self._to_gray(prev), self._to_gray(frame))
        return float(np.mean(diff))

    @staticmethod
    def _match_shape(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        if prev.shape == cur.shape:
            return prev
        h, w = cur.shape[:2]
        return cv2.resize(prev, (w, h), interpolation=cv2.INTER_AREA)
