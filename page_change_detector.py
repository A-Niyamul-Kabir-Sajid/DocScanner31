"""Detect when a new page has been placed under the camera.

Three independent signals are fused:

* **Motion plateau** -- frame-to-frame MAD spikes while the user grabs a
  page, then settles below ``PAGE_CHANGE_MOTION_REST_PX`` for
  ``PAGE_CHANGE_REST_FRAMES`` consecutive frames.
* **Quad jump** -- the same ``StabilityTracker`` quad seen while stable,
  then its replacement quad distance exceeds ``PAGE_CHANGE_QUAD_JUMP_PX``.
* **phash delta** -- perceptual hash (via Pillow+imagehash) of the warped
  A4 page differs from the last captured one by more than
  ``PAGE_CHANGE_HASH_DISTANCE`` bits.

The detector owns a small "armed" flag: it only fires after it has
observed at least one stable page, then a move, then a settle.  A new
event always re-arms immediately so back-to-back identical blanks won't
double-fire.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from config import (
    AUTO_PAGE_CHANGE_BUMP,
    PAGE_CHANGE_ENABLED,
    PAGE_CHANGE_HASH_DISTANCE,
    PAGE_CHANGE_MOTION_REST_PX,
    PAGE_CHANGE_MOTION_TRIGGER_PX,
    PAGE_CHANGE_QUAD_JUMP_PX,
    PAGE_CHANGE_REST_FRAMES,
)
from corner_refiner import Quad

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PageChangeEvent:
    """Emitted by :class:`PageChangeDetector` when a new page is detected."""

    timestamp: float            # ``time.monotonic()`` of the event
    confidence: float           # 0..1
    hash_distance: int          # phash hamming distance vs last captured page
    quad_distance: float        # px between last stable quad and current
    motion_at_peak: float       # peak MAD observed during the move


# --------------------------------------------------------------------------- #
# PageChangeDetector
# --------------------------------------------------------------------------- #
class PageChangeDetector:
    """Watch the live pipeline and emit :class:`PageChangeEvent`s."""

    _PHASE_IDLE = "IDLE"
    _PHASE_ARMED = "ARMED"
    _PHASE_MOVING = "MOVING"
    _PHASE_SETTLING = "SETTLING"

    def __init__(
        self,
        *,
        enabled: bool = PAGE_CHANGE_ENABLED,
        motion_trigger_px: float = PAGE_CHANGE_MOTION_TRIGGER_PX,
        motion_rest_px: float = PAGE_CHANGE_MOTION_REST_PX,
        rest_frames: int = PAGE_CHANGE_REST_FRAMES,
        quad_jump_px: float = PAGE_CHANGE_QUAD_JUMP_PX,
        hash_distance: int = PAGE_CHANGE_HASH_DISTANCE,
        auto_bump: bool = AUTO_PAGE_CHANGE_BUMP,
        hash_size: int = 8,
    ) -> None:
        self.enabled = enabled
        self.motion_trigger_px = motion_trigger_px
        self.motion_rest_px = motion_rest_px
        self.rest_frames = rest_frames
        self.quad_jump_px = quad_jump_px
        self.hash_distance = hash_distance
        self.auto_bump = auto_bump
        self._hash_size = hash_size

        # Internal state.
        self._baseline_quad: Optional[Quad] = None      # last stable quad
        self._baseline_hash = None                      # last captured phash
        self._phase: str = self._PHASE_IDLE
        self._peak_motion: float = 0.0
        self._rest_streak: int = 0
        self.last_event: Optional[PageChangeEvent] = None
        self.recent_motion: List[float] = []  # rolling window (last 16)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Wipe baselines (called by ``ScanSession.start_new_document``)."""
        self._baseline_quad = None
        self._baseline_hash = None
        self._phase = self._PHASE_IDLE
        self._peak_motion = 0.0
        self._rest_streak = 0
        self.last_event = None
        self.recent_motion = []

    def update_baseline_after_capture(
        self, processed_bgr: np.ndarray, quad: Optional[Quad]
    ) -> None:
        """Call after a successful capture so the detector knows the current page."""
        if not self.enabled or processed_bgr is None:
            return
        try:
            self._baseline_hash = _phash(processed_bgr, self._hash_size)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("phash baseline failed: %s", exc)
        if quad is not None:
            self._baseline_quad = quad
        # After a capture we are ARMED: any move will be treated as a swap.
        self._phase = self._PHASE_ARMED
        self._rest_streak = 0
        self._peak_motion = 0.0

    def observe(
        self,
        *,
        quad: Optional[Quad],
        processed_bgr: np.ndarray,
        motion_px: float,
    ) -> Optional[PageChangeEvent]:
        """Feed one frame's signals in.  Returns a ``PageChangeEvent`` on swap."""
        if not self.enabled:
            return None

        # Keep a rolling motion window for observability.
        self.recent_motion.append(float(motion_px))
        if len(self.recent_motion) > 16:
            self.recent_motion.pop(0)

        # ---- Arming (first time we see a calm frame) ----
        if self._phase == self._PHASE_IDLE:
            if quad is not None and motion_px < self.motion_rest_px:
                self._baseline_quad = quad
                self._phase = self._PHASE_ARMED
            return None

        # ---- Movement detected ----
        if motion_px > self.motion_trigger_px:
            self._peak_motion = max(self._peak_motion, motion_px)
            self._rest_streak = 0
            if self._phase == self._PHASE_ARMED:
                self._phase = self._PHASE_MOVING
            return None

        # ---- Settling -- need N consecutive calm frames to confirm ----
        if self._phase in (self._PHASE_MOVING, self._PHASE_SETTLING):
            if motion_px <= self.motion_rest_px:
                self._rest_streak += 1
            else:
                self._rest_streak = 0
                self._peak_motion = max(self._peak_motion, motion_px)
                self._phase = self._PHASE_MOVING

            if self._rest_streak < self.rest_frames:
                self._phase = self._PHASE_SETTLING
                return None

            # Settled long enough -- try to confirm.
            event = self._confirm_and_fire(quad, processed_bgr)
            if event is not None:
                return event

            # Hash/quad didn't confirm -- return to ARMED so a future
            # genuine move still gets another chance.
            self._phase = self._PHASE_ARMED
            self._rest_streak = 0
            self._peak_motion = 0.0
            return None

        # ARMED + calm: keep the baseline fresh.
        if motion_px <= self.motion_rest_px and quad is not None:
            self._baseline_quad = quad
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _confirm_and_fire(
        self,
        quad: Optional[Quad],
        processed_bgr: np.ndarray,
    ) -> Optional[PageChangeEvent]:
        quad_dist = 0.0
        if self._baseline_quad is not None and quad is not None:
            quad_dist = _quad_distance(self._baseline_quad, quad)

        try:
            cur_hash = _phash(processed_bgr, self._hash_size)
        except Exception:  # pragma: no cover - defensive
            cur_hash = None

        hash_dist = -1
        if self._baseline_hash is not None and cur_hash is not None:
            # ImageHash supports subtraction as Hamming distance.
            hash_dist = int(self._baseline_hash - cur_hash)

        hash_ok = hash_dist < 0 or hash_dist >= self.hash_distance
        quad_ok = self._baseline_quad is None or quad is None or quad_dist >= self.quad_jump_px

        if not hash_ok and not quad_ok:
            logger.debug(
                "page-change not confirmed (hash=%s quad=%.1fpx)",
                hash_dist, quad_dist,
            )
            return None

        confidence = _confidence(
            hash_dist, quad_dist, self._peak_motion,
            self.hash_distance, self.quad_jump_px,
        )
        event = PageChangeEvent(
            timestamp=time.monotonic(),
            confidence=confidence,
            hash_distance=max(0, hash_dist),
            quad_distance=quad_dist,
            motion_at_peak=self._peak_motion,
        )

        # Update baselines so a SECOND straight-away swap still fires.
        self._baseline_hash = cur_hash or self._baseline_hash
        if quad is not None:
            self._baseline_quad = quad
        self._peak_motion = 0.0
        self._rest_streak = 0
        self._phase = self._PHASE_ARMED
        self.last_event = event
        logger.info(
            "page-change CONFIRMED  conf=%.2f  hash=%d  quad=%.1fpx  peak_motion=%.1fpx",
            confidence, max(0, hash_dist), quad_dist, event.motion_at_peak,
        )
        return event


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _quad_distance(a: Quad, b: Quad) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        return float("inf")
    return float(np.linalg.norm(arr_a - arr_b, axis=1).mean())


def _phash(bgr: np.ndarray, hash_size: int = 8):
    """Compute an imagehash.ImageHash without forcing a hard dep at import time."""
    import imagehash
    from PIL import Image

    if bgr is None or bgr.size == 0:
        raise ValueError("empty frame for phash")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return imagehash.phash(pil, hash_size=hash_size)


def _confidence(
    hash_dist: int,
    quad_dist: float,
    peak_motion: float,
    hash_threshold: int,
    quad_threshold: float,
) -> float:
    """Cheap 0..1 confidence weighted by signal strength."""
    hash_score = 0.0
    if hash_dist >= 0:
        hash_score = min(1.0, hash_dist / max(1, hash_threshold * 2))
    quad_score = min(1.0, quad_dist / max(1.0, quad_threshold * 2))
    motion_score = min(1.0, peak_motion / 60.0)
    # Hash is the strongest signal; weight it.
    return max(0.0, min(1.0, 0.5 * hash_score + 0.3 * quad_score + 0.2 * motion_score))
