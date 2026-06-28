"""
camera.py
=========

Camera abstraction for the Smart Document Scanner.

Two backends are supported via the ``backend`` argument of ``Camera``:

1. ``"opencv"`` (default) — uses ``cv2.VideoCapture``. Works on Windows with
   DroidCam / IP Webcam and on Linux/Raspberry Pi with a standard UVC USB cam.
2. ``"picamera2"`` — uses the Raspberry Pi Camera Module 2/3.  Only importable
   on a Pi running libcamera + picamera2.

Keeping the camera behind this thin wrapper means ``app.py`` doesn't need to
know which hardware is attached.  When porting to the Pi, just construct
``Camera(backend="picamera2", size=(4608, 2592))`` and the rest of the code
stays identical.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """A thin, swappable wrapper around ``cv2.VideoCapture`` / ``picamera2``.

    Parameters
    ----------
    source : int | str
        Either a numeric index for a local webcam (``0``, ``1``, ...) or the
        URL of a network camera such as ``http://192.168.1.10:4747/video``.
    width, height : int
        Desired capture resolution.  Note: phone cams may not honor this.
    backend : str
        ``"opencv"`` (default) or ``"picamera2"`` (Raspberry Pi only).
    """

    def __init__(
        self,
        source=0,
        width: int = 1280,
        height: int = 720,
        backend: str = "opencv",
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.backend = backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._pi_cam = None  # only set when backend == "picamera2"

        if backend == "picamera2":
            self._open_picamera2()
        else:
            self._open_opencv()

    # ------------------------------------------------------------------ #
    # Backend openers
    # ------------------------------------------------------------------ #
    def _open_opencv(self) -> None:
        """Open a UVC / network stream via OpenCV."""
        # Accept either a numeric index (e.g. "0", "1") or a URL.
        # Numeric strings are converted to int so cv2.VideoCapture uses the
        # DirectShow/MSMF index path on Windows; URLs stay as strings for the
        # FFMPEG/HTTP backend.
        src = self.source
        url = None
        try:
            src = int(src)
        except (TypeError, ValueError):
            # Anything that isn't a plain integer is treated as a URL/stream.
            url = str(src)
        logger.info("Opening OpenCV camera source=%r", src)

        if url is not None:
            # Network streams (DroidCam / IP Webcam) ship as
            # ``multipart/x-mixed-replace`` MJPEG.  On Windows, the default
            # backend often refuses to open the URL; explicitly routing
            # through FFMPEG fixes this on the opencv-python 4.13 wheel.
            self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source {self.source!r}. "
                "Check the index/URL and that DroidCam / IP Webcam is running."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _open_picamera2(self) -> None:
        """Open the Raspberry Pi Camera Module via picamera2."""
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:  # pragma: no cover - Pi-only branch
            raise RuntimeError(
                "picamera2 is not installed. Run 'pip install picamera2' "
                "on the Raspberry Pi OS (Bookworm or newer)."
            ) from exc

        logger.info("Opening Raspberry Pi camera at %dx%d", self.width, self.height)
        self._pi_cam = Picamera2()
        config = self._pi_cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self._pi_cam.configure(config)
        self._pi_cam.start()

    # ------------------------------------------------------------------ #
    # Frame acquisition
    # ------------------------------------------------------------------ #
    def read(self) -> Tuple[bool, np.ndarray]:
        """Return ``(ok, frame)`` mirroring ``VideoCapture.read()``.

        For picamera2 the captured array is already RGB, so we convert to
        BGR to keep the rest of the OpenCV pipeline unchanged.
        """
        if self.backend == "picamera2":
            assert self._pi_cam is not None
            frame_rgb = self._pi_cam.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr

        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("Camera frame read failed")
        return ok, frame

    def release(self) -> None:
        """Release hardware resources.  Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._pi_cam is not None:
            try:
                self._pi_cam.stop()
                self._pi_cam.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.exception("Error while closing picamera2")
            self._pi_cam = None

    # ------------------------------------------------------------------ #
    # Context manager sugar
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()