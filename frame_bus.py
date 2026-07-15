"""Thread-safe live-frame bus shared between the camera loop and the web UI.

The single camera reader lives in the main :class:`ScanSession` loop
(``app.py``).  Each LIVE render already runs
``DocumentProcessor.process_with_debug`` and therefore holds every pipeline
stage (original / gray / edges / warped page / ...).  Rather than open the
camera a second time from the Flask thread (which would fight the main loop
for the device), the loop **publishes** those stages here and the web
endpoints **consume** the latest one on demand.

Design notes
------------
* ``publish_many`` only swaps ndarray *references* under a lock.
  ``process_with_debug`` returns a fresh array for every stage on every frame
  and never mutates them in place, so a consumer that grabs the reference
  under the lock can safely JPEG-encode it after releasing the lock.
* ``encode_jpeg`` keeps a tiny per-(stage, width) time-throttled cache so that
  N concurrent MJPEG viewers of the same stage share a single ``cv2.imencode``
  call instead of each paying for their own encode on every tick.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Mapping, Optional

import cv2
import numpy as np

# Canonical stage keys the web UI streams.  Kept here so both the publisher
# (app.py) and the consumer (flask_server.py) agree on the vocabulary.
STAGE_KEYS = (
    "original",       # raw camera frame
    "gray",           # grayscale
    "binary",         # auto-Canny edge map
    "contours",       # all-contours overlay
    "present",        # final processed page (what C saves)
    "selected",       # biggest-contour overlay (chosen quad + corners)
    "page_color",     # perspective-warped colour page
    "page_gray",      # warped page, grayscale
    "page_bw",        # warped page, adaptive threshold (black & white)
    "last_captured",  # most recently captured page (static until next capture)
)


def _placeholder(text: str = "waiting for a page", w: int = 640, h: int = 360) -> np.ndarray:
    """A black BGR tile with a centred caption, used when a stage is missing."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (20, 25, 45)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    cv2.putText(
        canvas,
        text,
        ((w - tw) // 2, (h + th) // 2),
        font,
        0.7,
        (138, 147, 184),
        2,
        cv2.LINE_AA,
    )
    return canvas


class LiveFrameBus:
    """Holds the latest BGR frame for each pipeline stage, thread-safely."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: Dict[str, np.ndarray] = {}
        # (stage, max_w) -> (monotonic_ts, jpeg_bytes)
        self._cache: Dict[tuple, tuple] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def publish(self, stage: str, image: Optional[np.ndarray]) -> None:
        """Publish a single stage.  ``None`` is ignored (keeps the last frame)."""
        if image is None:
            return
        with self._lock:
            self._frames[stage] = image

    def publish_many(self, mapping: Mapping[str, Optional[np.ndarray]]) -> None:
        """Publish several stages at once.  ``None`` values are skipped."""
        with self._lock:
            for stage, image in mapping.items():
                if image is not None:
                    self._frames[stage] = image

    # ------------------------------------------------------------------ #
    def get(self, stage: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._frames.get(stage)

    def has_any(self) -> bool:
        with self._lock:
            return bool(self._frames)

    # ------------------------------------------------------------------ #
    def encode_jpeg(
        self,
        stage: str,
        *,
        quality: int = 70,
        max_w: Optional[int] = None,
        min_interval: float = 0.08,
    ) -> bytes:
        """Return JPEG bytes for ``stage``, downscaled to ``max_w`` if given.

        A short time-throttled cache keyed on ``(stage, max_w)`` means many
        simultaneous viewers of the same stream re-use one encode rather than
        each hammering ``cv2.imencode`` on every request tick.  A black
        placeholder is encoded when the stage has never been published.
        """
        key = (stage, max_w)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None and (now - cached[0]) < min_interval:
                return cached[1]

        frame = self.get(stage)
        if frame is None:
            frame = _placeholder()

        if max_w is not None and frame.shape[1] > max_w:
            scale = max_w / float(frame.shape[1])
            frame = cv2.resize(
                frame,
                (max_w, max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        data = buf.tobytes() if ok else b""
        with self._cache_lock:
            self._cache[key] = (now, data)
        return data
