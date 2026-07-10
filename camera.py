"""
camera.py
=========

Camera abstraction for the Smart Document Scanner.

Two real backends are supported, plus an ``"auto"`` mode that picks the right
one for the host platform:

1. ``"opencv"`` — uses ``cv2.VideoCapture``. Works on Windows with DroidCam /
   IP Webcam, and on Linux/Pi with a standard UVC USB cam or a libcamera
   pipeline that exposes a ``/dev/video*`` node.
2. ``"picamera2"`` — uses the Raspberry Pi Camera Module via libcamera.
   Required for native control of focus / exposure / AWB on the Pi Camera
   Module (including the IMX519 with AK7375 autofocus motor).  Only importable
   on a Pi running libcamera + picamera2.
3. ``"auto"`` (default) — :func:`detect_raspberry_pi` decides which of the
   above to use.  Windows / macOS desktop development boxes land on OpenCV;
   a Pi 4/5 with the V4L2 stack lands on picamera2.  URL sources (DroidCam /
   IP Webcam) always downgrade to OpenCV regardless of platform.

Keeping the camera behind this thin wrapper means ``app.py`` doesn't need to
know which hardware is attached.  When porting to the Pi, the same code path
that worked on the dev laptop now picks the IMX519 stack automatically; you
can still force a backend with ``Camera(backend="opencv", ...)`` or
``Camera(backend="picamera2", ...)`` for testing.

Focus control
-------------

For a document scanner fixed focus outperforms continuous autofocus (no focus
hunting, no blur spikes mid-page).  When ``autofocus=False`` and the picamera2
backend is active, ``Camera`` issues a single ``set_controls`` call setting
``AfMode = Manual`` and ``LensPosition = lens_position`` (in dioptres).  The
AK7375 lens motor on the IMX519 honours this through the Raspberry Pi camera
stack (the same path you used manually with ``v4l2-ctl --set-ctrl=focus_absolute``).
On OpenCV the kwargs are accepted but inert — USB webcams and phone-stream
URLs expose no standard focus control through ``cv2.VideoCapture``.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Platform detection
# --------------------------------------------------------------------------- #
_RPI_MODEL_PATH = "/proc/device-tree/model"
# Cached on first call; the answer doesn't change for the lifetime of the
# process and probing ``/proc/device-tree`` on every Camera(...) instantiation
# would be wasteful in the LIVE tick.
_RPI_DETECTED: Optional[bool] = None


def detect_raspberry_pi() -> bool:
    """Return True if we appear to be running on a Raspberry Pi.

    Three signals are checked (any one is enough):

    * ``platform.uname().machine`` reports an aarch64/armv7l kernel — useful
      on Pi OS Bookworm where /proc/device-tree may be masked by containers.
    * ``/proc/device-tree/model`` exists and contains the string
      "Raspberry Pi" (case-insensitive).  This is the canonical check.
    * ``/dev/v4l-subdev*`` is present, which only happens when libcamera has
      loaded a sensor driver (IMX219, IMX477, IMX519, ...).

    The function never raises: any OSError / FileNotFoundError is swallowed
    and treated as "not a Pi" so a missing procfs on a desktop box can't
    prevent the OpenCV fallback from running.
    """
    global _RPI_DETECTED
    if _RPI_DETECTED is not None:
        return _RPI_DETECTED

    try:
        # 1. Kernel architecture heuristic.
        try:
            machine = (os.uname().machine or "").lower()
        except (AttributeError, OSError):
            machine = ""
        if machine in {"aarch64", "armv7l", "armv6l"}:
            logger.info("Raspberry Pi detected via uname.machine=%r", machine)
            _RPI_DETECTED = True
            return True

        # 2. Device-tree model string.
        try:
            with open(_RPI_MODEL_PATH, "rb") as fh:
                model_blob = fh.read().decode("utf-8", errors="ignore").lower()
        except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
            model_blob = ""
        if "raspberry pi" in model_blob:
            logger.info(
                "Raspberry Pi detected via %s (%r)",
                _RPI_MODEL_PATH, model_blob.strip("\x00").strip(),
            )
            _RPI_DETECTED = True
            return True

        # 3. libcamera subdev presence — only Pi camera stacks create these.
        try:
            for entry in os.listdir("/dev"):
                if entry.startswith("v4l-subdev"):
                    logger.info(
                        "Raspberry Pi camera stack detected via /dev/%s", entry,
                    )
                    _RPI_DETECTED = True
                    return True
        except (FileNotFoundError, PermissionError, OSError):
            pass
    except Exception:  # pragma: no cover - belt-and-braces
        logger.exception("Pi detection failed; falling back to OpenCV")

    _RPI_DETECTED = False
    return False


def _looks_like_url(source: object) -> bool:
    """Heuristic: is ``source`` a network stream rather than a local index?"""
    if not isinstance(source, str):
        return False
    parsed = urlparse(source)
    return bool(parsed.scheme and parsed.netloc)


def select_backend(source: object, requested: str) -> str:
    """Resolve the ``backend`` argument to one of the real backends.

    * ``"opencv"`` / ``"picamera2"`` are returned verbatim.
    * ``"auto"`` maps to ``"picamera2"`` on a Pi, ``"opencv"`` elsewhere.
    * URL sources (DroidCam / IP Webcam) downgrade to OpenCV even when the
      user explicitly asked for picamera2 — the Pi camera stack cannot serve
      an HTTP MJPEG feed, and silently failing would mask the real cause.
    """
    requested = (requested or "auto").lower()
    if requested not in {"opencv", "picamera2", "auto"}:
        logger.warning("Unknown backend %r; falling back to OpenCV", requested)
        requested = "opencv"

    if _looks_like_url(source) and requested == "picamera2":
        logger.info(
            "Camera source %r is a URL; overriding picamera2 -> opencv", source,
        )
        return "opencv"

    if requested == "auto":
        return "picamera2" if detect_raspberry_pi() else "opencv"
    return requested


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
        ``"opencv"`` / ``"picamera2"`` / ``"auto"`` (default).
    autofocus : bool | None
        When using picamera2: ``True`` requests continuous autofocus,
        ``False`` switches to manual focus at ``lens_position``, ``None``
        defers to the global default (``config.DEFAULT_AUTOFOCUS``).  Ignored
        on the OpenCV backend (USB cams and phone URLs expose no standard
        focus control through ``cv2.VideoCapture``).
    lens_position : float | None
        Manual focus distance in **dioptres** (1/metres).  Only honoured on
        the picamera2 backend when ``autofocus=False``.  Sensible desk-scan
        values land between 1.5 (≈65 cm) and 4.0 (≈25 cm); the default of
        2.2 (≈45 cm) matches a typical over-the-desk scanner mount.
    """

    def __init__(
        self,
        source=0,
        width: int = 1280,
        height: int = 720,
        backend: str = "auto",
        autofocus: Optional[bool] = None,
        lens_position: Optional[float] = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        # Resolve the requested backend to a concrete string.  URLs always
        # collapse to "opencv"; "auto" picks by platform.
        self.backend = select_backend(source, backend)
        self._requested_backend = backend
        # Focus knobs.  Pull the runtime defaults lazily so tests can patch
        # ``config.DEFAULT_*`` without importing this module first.
        try:
            from config import DEFAULT_AUTOFOCUS, DEFAULT_LENS_POSITION_DIOPTRES
            self._default_autofocus = DEFAULT_AUTOFOCUS
            self._default_lens_position = DEFAULT_LENS_POSITION_DIOPTRES
        except Exception:  # pragma: no cover - config is required at runtime
            self._default_autofocus = False
            self._default_lens_position = 2.2
        self.autofocus = self._default_autofocus if autofocus is None else bool(autofocus)
        self.lens_position = (
            self._default_lens_position if lens_position is None else float(lens_position)
        )
        self._cap: Optional[cv2.VideoCapture] = None
        self._pi_cam = None  # only set when backend == "picamera2"
        # Track the last open attempt so the app layer can poll for recovery.
        # ``is_open`` is the canonical "is the camera usable right now?" flag.
        self.is_open: bool = False
        self.last_open_error: Optional[str] = None
        # Latches whether the most recent frame read succeeded.  ``try_reopen``
        # uses this to decide whether to rebuild the picamera2 pipeline; a
        # transient read failure should NOT trigger a full reconfigure.
        self._last_read_ok: bool = False

        if self.backend == "picamera2":
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
            return self._reopen_picamera2()
        self._open_opencv(raise_on_failure=False)
        return self.is_open

    def _reopen_picamera2(self) -> bool:
        """Tear down and recreate the picamera2 pipeline on failure.

        ``picamera2`` lacks a cheap "are you still alive?" probe, so when a
        read raises (or the user explicitly closed the handle) we close
        the current pipeline and instantiate a fresh ``Picamera2()``.  The
        underlying libcamera context is reused but the sensor pipeline is
        rebuilt, which is the only way to clear a stuck sensor state.
        """
        # If we already believe the camera is open AND a recent read
        # succeeded we have nothing to do.  ``_last_read_ok`` tracks that
        # so a transient ``capture_array`` failure isn't escalated to a
        # full reconfigure on every single frame.
        if (
            self._pi_cam is not None
            and self.is_open
            and self._last_read_ok
        ):
            return True

        if self._pi_cam is not None:
            try:
                self._pi_cam.stop()
            except Exception:  # pragma: no cover - best-effort
                logger.debug("picamera2 stop() raised during reopen; ignoring")
            try:
                self._pi_cam.close()
            except Exception:  # pragma: no cover - best-effort
                logger.debug("picamera2 close() raised during reopen; ignoring")
            self._pi_cam = None
            self.is_open = False
            self._last_read_ok = False

        # Re-open from scratch using the same parameters we stored at
        # construction time.  _open_picamera2 also reapplies the focus
        # controls, so a focused mount keeps its focus across reconnects.
        try:
            self._open_picamera2()
            self.is_open = True
            self._last_read_ok = True
            self.last_open_error = None
            return True
        except Exception as exc:  # pragma: no cover - driver-specific
            self.is_open = False
            self._pi_cam = None
            self._last_read_ok = False
            self.last_open_error = f"picamera2 reopen failed: {exc}"
            logger.warning(self.last_open_error)
            return False

    # ------------------------------------------------------------------ #
    # Backend openers
    # ------------------------------------------------------------------ #
    def _open_picamera2(self) -> None:
        """Open the Raspberry Pi Camera Module via picamera2."""
        try:
            from picamera2 import Picamera2  # type: ignore
            from libcamera import controls  # type: ignore
        except ImportError as exc:  # pragma: no cover - Pi-only branch
            raise RuntimeError(
                "picamera2 is not installed. Run 'pip install picamera2' "
                "on the Raspberry Pi OS (Bookworm or newer)."
            ) from exc

        logger.info(
            "Opening Raspberry Pi camera at %dx%d (autofocus=%s, "
            "lens_position=%s dioptres)",
            self.width, self.height, self.autofocus, self.lens_position,
        )
        self._pi_cam = Picamera2()
        config = self._pi_cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self._pi_cam.configure(config)
        self._pi_cam.start()

        # Apply focus controls AFTER start(): libcamera rejects most
        # ``set_controls`` calls on a configured-but-not-started pipeline.
        # We *try* the focus controls and only warn on failure so the app
        # still runs on a fixed-focus camera module (V1/V2/HQ) that doesn't
        # have an ``AfMode`` enum entry at all.
        focus_controls = {}
        if self.autofocus:
            focus_controls["AfMode"] = controls.AfModeEnum.Continuous
        else:
            focus_controls["AfMode"] = controls.AfModeEnum.Manual
            focus_controls["LensPosition"] = float(self.lens_position)
        try:
            self._pi_cam.set_controls(focus_controls)
            logger.info("Applied picamera2 focus controls: %s", focus_controls)
        except Exception as exc:  # pragma: no cover - driver-specific
            # Camera without an autofocus motor (V1, V2, HQ without AF lens)
            # raises here.  We log and continue so the LIVE overlay still
            # renders — focus is irrelevant in that case anyway.
            logger.warning(
                "Picamera2 focus controls %s rejected by driver: %s. "
                "Falling back to whatever the sensor defaults to.",
                focus_controls, exc,
            )
        # Mark the pipeline as live.  Without this the LIVE loop spins
        # forever in the "camera not found" overlay because ``read()``
        # never had a reason to flip ``is_open``.  A freshly-started
        # pipeline IS open; treat it as such.
        self.is_open = True
        self._last_read_ok = True
        self.last_open_error = None
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
                self._last_read_ok = False
                return False, empty
            try:
                frame_rgb = self._pi_cam.capture_array()
            except Exception as exc:  # pragma: no cover - driver-specific
                logger.warning("picamera2 capture_array failed: %s", exc)
                self.is_open = False
                self._last_read_ok = False
                self.last_open_error = f"capture_array: {exc}"
                empty = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                return False, empty
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._last_read_ok = True
            # Only stamp is_open=True after the first successful frame;
            # that way a configured-but-not-started pipeline (or a sensor
            # that's wedged after start()) gets one retry before we mark
            # it offline.  Cheap insurance against the "stop()/close() but
            # is_open still True" race that used to plague the Pi path.
            if not self.is_open:
                self.is_open = True
                self.last_open_error = None
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
        # Drop the recovery latch so a subsequent ``try_reopen`` knows it
        # has to do real work rather than short-circuiting on stale state.
        self.is_open = False
        self._last_read_ok = False

    # ------------------------------------------------------------------ #
    # Context manager sugar
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()