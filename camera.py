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
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Fast TCP probe for HTTP camera URLs
# --------------------------------------------------------------------------- #
def _probe_http_url(url: str, timeout: float = 1.0) -> bool:
    """Return True if the host:port in *url* is reachable right now.

    This is a **cheap, non-blocking** check that avoids the 10-30 s
    connect timeout baked into ``cv2.VideoCapture``'s FFMPEG backend.
    We only verify that the TCP socket accepts a connection; we do NOT
    send an HTTP request or read any data.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception:  # pragma: no cover - malformed URL
        return True  # let OpenCV try and report its own error

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


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
        # Track the last open attempt so the app layer can poll for recovery.
        # ``is_open`` is the canonical "is the camera usable right now?" flag.
        self.is_open: bool = False
        self.last_open_error: Optional[str] = None

        if backend == "picamera2":
            self._open_picamera2()
        else:
            # OpenCV: do NOT raise on failure - the caller (ScanSession)
            # shows a "camera not found" overlay and retries every few
            # seconds. Raising here would terminate the app.
            self._open_opencv(raise_on_failure=False)

    # ------------------------------------------------------------------ #
    # Backend openers
    # ------------------------------------------------------------------ #
    def _open_opencv(self, raise_on_failure: bool = True) -> None:
        """Open a UVC / network stream via OpenCV.

        When ``raise_on_failure`` is False the method logs and stores
        the error in ``self.last_open_error`` instead of raising, so the
        app layer can keep running and try again later.
        """
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

        # If we previously held a handle, drop it before opening a new one.
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._cap = None

        # Fast TCP probe for HTTP(S) URLs. ``cv2.VideoCapture`` will happily
        # block for tens of seconds waiting for a DroidCam / IP Webcam
        # server that doesn't exist, which makes the app look frozen.
        # A 1-second connect timeout is plenty for a phone on the same
        # LAN; if the device is genuinely offline we want to learn that
        # *fast* so we can show the "camera not found" overlay and retry.
        if url is not None:
            if not _probe_http_url(url, timeout=1.0):
                self.last_open_error = (
                    f"Could not reach camera URL {url!r} "
                    "(connection refused / timeout / DNS failure)."
                )
                logger.warning(self.last_open_error)
                self.is_open = False
                if raise_on_failure:
                    raise RuntimeError(self.last_open_error)
                return

        try:
            if url is not None:
                # Network streams (DroidCam / IP Webcam) ship as
                # ``multipart/x-mixed-replace`` MJPEG.  On Windows, the default
                # backend often refuses to open the URL; explicitly routing
                # through FFMPEG fixes this on the opencv-python 4.13 wheel.
                self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            else:
                self._cap = cv2.VideoCapture(src)
        except Exception as exc:  # pragma: no cover - cv2 rarely throws
            self.is_open = False
            self.last_open_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Camera open raised: %s", self.last_open_error)
            if raise_on_failure:
                raise
            return

        opened = bool(self._cap.isOpened())
        if not opened:
            self.last_open_error = (
                f"Could not open camera source {self.source!r}. "
                "Check the index/URL and that DroidCam / IP Webcam is running."
            )
            logger.warning(self.last_open_error)
            self.is_open = False
            if raise_on_failure:
                raise RuntimeError(self.last_open_error)
            return

        # Success: apply requested geometry.
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:  # pragma: no cover - some cams refuse set()
            pass
        self.is_open = True
        self.last_open_error = None

    # ------------------------------------------------------------------ #
    # Recovery
    # ------------------------------------------------------------------ #
    def try_reopen(self) -> bool:
        """Attempt to (re)open the underlying capture device.

        Returns True if the camera is now usable, False otherwise.  Safe to
        call on every frame; the actual ``cv2.VideoCapture`` call is cheap
        when the source is reachable.
        """
        if self.backend == "picamera2":
            # picamera2 currently raises on failure; we don't auto-recover
            # it here. The Pi boot path is rarely the failure case the user
            # is hitting (their crash is on Windows + DroidCam URL).
            return self._pi_cam is not None
        self._open_opencv(raise_on_failure=False)
        return self.is_open

    # ------------------------------------------------------------------ #
    # Backend openers
    # ------------------------------------------------------------------ #
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

        When the camera is offline (``is_open`` is False or the underlying
        handle was never created) we return ``(False, empty_frame)`` instead
        of raising so the GUI loop can render its "camera not found" overlay
        without crashing.
        """
        if self.backend == "picamera2":
            if self._pi_cam is None:
                empty = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                return False, empty
            frame_rgb = self._pi_cam.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr

        if not self.is_open or self._cap is None:
            empty = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return False, empty

        try:
            ok, frame = self._cap.read()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Camera read raised: %s", exc)
            self.is_open = False
            empty = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return False, empty
        if not ok:
            # Network streams sometimes return a frame of zeros on a
            # transient hiccup. Mark the camera unhealthy so the app can
            # schedule a retry without crashing the rest of the pipeline.
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