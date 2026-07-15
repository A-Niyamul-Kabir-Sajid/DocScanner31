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
import time
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
    full_fov : bool
        Pi-camera-only: when ``True`` (default) the wrapper asks libcamera
        for the **full sensor area** via ``ScalerCrop = (0, 0, 1, 1)`` so the
        LIVE preview matches ``rpicam-hello --width W --height H`` instead
        of being cropped (digitally zoomed) to the requested aspect ratio.
        Set to ``False`` to recover the old behaviour where picamera2 picks
        the tightest matching crop.  Ignored on the OpenCV backend; UVC
        webcams / phone streams don't apply this auto-crop in the first
        place so the flag is a no-op there.
    """

    def __init__(
        self,
        source=0,
        width: int = 1280,
        height: int = 720,
        backend: str = "auto",
        autofocus: Optional[bool] = None,
        lens_position: Optional[float] = None,
        rotate: int = 0,
        full_fov: bool = True,
        autofocus_on_capture: bool = False,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        # Resolve the requested backend to a concrete string.  URLs always
        # collapse to "opencv"; "auto" picks by platform.
        self.backend = select_backend(source, backend)
        self._requested_backend = backend
        # Rotation applied to every frame leaving ``read()``.  Useful when
        # the camera module is physically mounted portrait-side or you just
        # want the LIVE preview / scanner output in portrait orientation
        # without rotating the sensor mount.  Allowed: 0, 90, 180, 270.
        # Anything else is clamped to 0 with a warning so a typo in the
        # CLI / config doesn't silently produce sideways scans.
        try:
            rot = int(rotate)
        except (TypeError, ValueError):
            rot = 0
        if rot not in (0, 90, 180, 270):
            logger.warning(
                "Unsupported camera rotate=%r; expected 0/90/180/270. "
                "Falling back to no rotation.", rotate,
            )
            rot = 0
        self.rotate = rot
        # Pi-camera-only: disable libcamera's automatic sensor crop so the
        # preview shows the same field of view as ``rpicam-hello --width W
        # --height H``.  OpenCV ignores this flag.
        self.full_fov = bool(full_fov)
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
        # Single-shot AF trigger before each capture on picamera2.  See
        # ``trigger_autofocus`` for the lock-poll loop.  Honoured only on
        # the picamera2 backend; the OpenCV backend treats it as a no-op.
        self.autofocus_on_capture = bool(autofocus_on_capture)
        self._cap: Optional[cv2.VideoCapture] = None
        self._pi_cam = None  # only set when backend == "picamera2"
        # ``_af_locked`` / ``_af_in_flight`` track single-shot AF on picamera2
        # so concurrent capture calls don't issue overlapping triggers.
        self._af_in_flight: bool = False
        self._af_locked: Optional[bool] = None
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
    def _select_picamera2_config(self) -> dict:
        """Build a ``picamera2`` configuration honouring ``self.full_fov``.

        When ``full_fov=True`` (default) we ask picamera2 to use the
        **largest** sensor mode the driver exposes, then let libcamera
        downscale the main stream to the requested ``(width, height)``.
        This is what ``rpicam-hello --width W --height H`` does internally,
        and it's the only way to keep the full field of view on a 4:3 sensor
        when the user asks for a 16:9 output like 1280x720.

        When ``full_fov=False`` we fall back to the legacy behaviour:
        picamera2 picks a sensor mode whose native aspect ratio already
        matches the requested output, which crops the sensor (and so
        zooms in digitally).

        Older picamera2 builds (pre-0.3) don't accept the ``sensor=``
        kwarg; in that case we degrade gracefully to the default mode
        selection instead of raising.
        """
        main_size = (int(self.width), int(self.height))
        main_stream = {"size": main_size, "format": "RGB888"}

        if not self.full_fov:
            return self._pi_cam.create_video_configuration(main=main_stream)

        try:
            sensor_modes = self._pi_cam.sensor_modes or []
        except Exception as exc:  # pragma: no cover - driver-specific
            logger.warning(
                "picamera2.sensor_modes probe failed (%s); using the "
                "default mode selection (preview may look zoomed).", exc,
            )
            return self._pi_cam.create_video_configuration(main=main_stream)

        if not sensor_modes:
            return self._pi_cam.create_video_configuration(main=main_stream)

        # Pick the largest output size - that sensor mode uses the entire
        # pixel array, so the FOV matches the lens's full coverage.
        largest = max(
            sensor_modes,
            key=lambda m: int(m["size"][0]) * int(m["size"][1]),
        )
        sensor_output_size = (int(largest["size"][0]), int(largest["size"][1]))
        logger.info(
            "Picamera2 full-FOV: forcing sensor mode %s, scaling main "
            "stream to %dx%d.", sensor_output_size, *main_size,
        )
        try:
            return self._pi_cam.create_video_configuration(
                main=main_stream,
                sensor={"output_size": sensor_output_size},
            )
        except TypeError:
            # Pre-0.3 picamera2 doesn't expose ``sensor=``. Fall back; the
            # user will see the legacy zoomed preview but the app still
            # runs.
            logger.warning(
                "picamera2.create_video_configuration() rejected the "
                "'sensor' kwarg on this picamera2 version; falling back "
                "to default mode selection (preview may look zoomed)."
            )
            return self._pi_cam.create_video_configuration(main=main_stream)

    # ------------------------------------------------------------------ #
    # Focus-application hardening for the IMX519 + AK7375 stack
    # ------------------------------------------------------------------ #
    # When ``set_controls(AfMode=Manual, LensPosition=X)`` lands, the next
    # frame that the sensor exposes reflects the new lens position.  On
    # the IMX519 running ~30 fps that's roughly 200-400 ms wall time
    # before ``capture_request().get_metadata()["LensPosition"]`` reports
    # X.  Reading metadata sooner gives you the *previous* position even
    # when the driver did accept the new value, which looks like a stuck
    # lens to the caller.  These two knobs tune the wait/retry policy.
    _FOCUS_SETTLE_DELAY_S = 0.30
    _FOCUS_SETTLE_ATTEMPTS = 4

    def set_manual_focus(self, dioptres: float) -> float:
        """Apply a new manual ``LensPosition`` and verify it stuck.

        Public entry point used by runtime focus hotkeys (``[``/``]``/``{``/``}``)
        so they get the same race protection as the startup path.  Returns
        the value the driver reports in metadata after the settle loop,
        or ``float('nan')`` if the camera isn't open / has no picamera2
        handle.  The caller can compare against the requested value to
        surface "the motor didn't move" in the HUD.
        """
        if self._pi_cam is None:
            return float("nan")
        try:
            from libcamera import controls as _controls  # type: ignore
        except Exception:
            return float("nan")
        target = float(dioptres)
        self.lens_position = target
        self.autofocus = False
        try:
            self._pi_cam.set_controls(
                {
                    "AfMode": _controls.AfModeEnum.Manual,
                    "LensPosition": target,
                }
            )
        except Exception as exc:  # pragma: no cover - driver-specific
            logger.warning("Manual focus set_controls raised: %s", exc)
            return float("nan")
        # Reuse the same settle/retry/auto-wake logic as startup so
        # runtime nudges behave identically to ``--lens-position``.
        try:
            self._settle_manual_focus(_controls)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("set_manual_focus settle raised: %s", exc)
        # Sample the metadata once more so the caller can show
        # "applied=2.70 dpt" in the HUD.  We sleep the same settle delay
        # so the frame we sample was exposed *after* the final retry.
        time.sleep(self._FOCUS_SETTLE_DELAY_S)
        try:
            req = self._pi_cam.capture_request()
            try:
                meta = req.get_metadata() or {}
            finally:
                req.release()
            actual = meta.get("LensPosition")
        except Exception:  # pragma: no cover - driver-specific
            return float("nan")
        if actual is None:
            return float("nan")
        try:
            return float(actual)
        except (TypeError, ValueError):
            return float("nan")

    def get_applied_lens_position(self) -> float:
        """Return the last ``LensPosition`` the driver reported in metadata.

        Distinct from ``self.lens_position`` (which is the value WE stored):
        on a fixed-focus module ``self.lens_position`` will be 2.7 even
        though the driver happily echoes back whatever you wrote without
        moving the lens.  This helper returns ``float('nan')`` when the
        camera isn't open or the metadata doesn't carry the field.
        """
        if self._pi_cam is None:
            return float("nan")
        try:
            req = self._pi_cam.capture_request()
            try:
                meta = req.get_metadata() or {}
            finally:
                req.release()
            actual = meta.get("LensPosition")
        except Exception:  # pragma: no cover - driver-specific
            return float("nan")
        if actual is None:
            return float("nan")
        try:
            return float(actual)
        except (TypeError, ValueError):
            return float("nan")

    def _settle_manual_focus(self, controls) -> None:
        """Re-apply ``LensPosition`` until the driver actually accepts it.

        The IMX519 + AK7375 autofocus actuator on Raspberry Pi OS Bookworm
        has two well-known races we have to work around:

        1. First-frame race: the very first
           ``set_controls(AfMode=Manual, LensPosition=...)`` call after
           ``Picamera2.start()`` returns success but the I²C command to the
           actuator never lands - the lens stays at the module's power-on
           default (typically infinity / far).

        2. Metadata-read race: ``capture_request().get_metadata()["LensPosition"]``
           reflects the controls stamped on a frame that was *already in
           flight* when we called ``set_controls``.  We have to wait at
           least one exposure (~200 ms on the IMX519 at 30 fps) before the
           next frame's metadata reports the new position.  Reading
           metadata immediately gives you back the OLD value, which is
           exactly the wrong signal for "did the lens move?".

        We re-issue controls up to ``FOCUS_SETTLE_ATTEMPTS`` times, sleeping
        ``FOCUS_SETTLE_DELAY_S`` seconds between each issuance so the next
        captured frame is exposed under the new controls.  If verification
        still fails we fall back to a one-shot ``AfMode=Auto`` cycle so
        libcamera talks to the actuator at least once before we re-apply
        Manual.
        """
        if self._pi_cam is None:
            return
        target = float(self.lens_position)
        attempts = int(getattr(self, "_FOCUS_SETTLE_ATTEMPTS", 4))
        delay_s = float(getattr(self, "_FOCUS_SETTLE_DELAY_S", 0.30))
        try:
            for attempt in range(attempts):
                # Give the actuator wall time to actually move before we
                # sample metadata.  Without this sleep we read metadata
                # from a frame whose exposure started before
                # ``set_controls`` returned, which still reports the
                # previous position even when the driver DID accept the
                # new value.
                time.sleep(delay_s)
                req = self._pi_cam.capture_request()
                try:
                    meta = req.get_metadata()
                finally:
                    req.release()
                actual = meta.get("LensPosition") if meta else None
                if actual is not None and abs(float(actual) - target) < 0.05:
                    logger.info(
                        "LensPosition confirmed at %.2f dpt on attempt %d.",
                        float(actual), attempt + 1,
                    )
                    return
                logger.warning(
                    "LensPosition request was %.2f dpt but driver reports "
                    "%.2f dpt (attempt %d/%d) - re-applying.",
                    target, actual if actual is not None else float("nan"),
                    attempt + 1, attempts,
                )
                self._pi_cam.set_controls(
                    {
                        "AfMode": controls.AfModeEnum.Manual,
                        "LensPosition": target,
                    }
                )
        except Exception as exc:  # pragma: no cover - driver-specific
            logger.warning("LensPosition verification raised: %s", exc)

        # Last resort: kick a one-shot AF pass so the actuator wakes up,
        # then re-issue our Manual value.  Without this, a camera that
        # booted up in an unknown focus state will sit at infinity
        # forever.  We then run the same settle-and-verify loop we used
        # above, because the metadata-read race means a single
        # ``capture_request`` straight after ``set_controls`` will echo
        # the OLD position and look like the lens didn't move.
        try:
            logger.info("Falling back to AfMode=Auto one-shot to wake the "
                        "lens actuator, then re-applying Manual %.2f.",
                        target)
            self._pi_cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
            # Wait for the auto pass to finish (poll metadata briefly).
            for _ in range(20):
                req = self._pi_cam.capture_request()
                try:
                    meta = req.get_metadata() or {}
                finally:
                    req.release()
                if meta.get("AfStatus") in (2, 3):  # Focused / Cannot focus
                    break
                time.sleep(0.05)
            self._pi_cam.set_controls(
                {
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": target,
                }
            )
            # Run the same settle loop after the Auto-wake so we don't
            # log "driver reports 1.00" when the actuator just hadn't
            # finished moving yet.
            for post_attempt in range(attempts):
                time.sleep(delay_s)
                req = self._pi_cam.capture_request()
                try:
                    meta = req.get_metadata() or {}
                finally:
                    req.release()
                actual = meta.get("LensPosition")
                if actual is not None and abs(float(actual) - target) < 0.05:
                    logger.info(
                        "After Auto-wake: lens confirmed at %.2f dpt "
                        "(post-wake attempt %d/%d).",
                        float(actual), post_attempt + 1, attempts,
                    )
                    return
                logger.warning(
                    "After Auto-wake: requested %.2f dpt, driver reports "
                    "%.2f (post-wake attempt %d/%d).",
                    target, actual if actual is not None else float("nan"),
                    post_attempt + 1, attempts,
                )
            logger.error(
                "Auto-wake fallback could not confirm %.2f dpt after %d "
                "settle attempts - the lens actuator may not be present "
                "(fixed-focus module?) or is wedged.",
                target, attempts,
            )
        except Exception as exc:  # pragma: no cover - driver-specific
            logger.warning("Auto-wake fallback raised: %s", exc)

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
        # ``full_fov=True`` (default) picks the LARGEST sensor mode and lets
        # libcamera scale the main stream down to (width, height).  This is
        # what ``rpicam-hello --width W --height H`` does internally, and it
        # preserves the full sensor field of view.
        #
        # Without this, ``create_video_configuration(main={"size": (1280,
        # 720)})`` on a 4:3 IMX519 selects a 16:9 sensor mode that already
        # crops the sensor vertically - the resulting preview is visibly
        # more "zoomed in" than the same resolution via rpicam-hello.
        config = self._select_picamera2_config()
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

        # Verify the lens actually moved.  On the IMX519 + AK7375 stack the
        # very first ``set_controls`` after ``start()`` is frequently a
        # no-op: the lens actuator needs a frame to wake up, and libcamera
        # silently clamps ``LensPosition`` to the previous value rather
        # than raising.  We capture a request, read back the metadata, and
        # if the driver didn't honour our value we re-issue the controls
        # (still no luck → switch to ``AfMode=Auto`` so libcamera kicks the
        # actuator at least once before we re-apply Manual).
        if not self.autofocus:
            try:
                self._settle_manual_focus(controls)
            except Exception as exc:  # pragma: no cover - driver-specific
                logger.warning("Manual-focus settle raised: %s", exc)
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
    def _apply_rotate(self, frame: np.ndarray) -> np.ndarray:
        """Rotate ``frame`` by ``self.rotate`` degrees clockwise.

        Called on every successful frame leaving ``read()`` so the rest of
        the pipeline (detector, scanner, LIVE overlay, PDF writer) sees a
        consistently-oriented image regardless of how the sensor is
        mounted.  Implemented with ``cv2.rotate`` because picamera2's
        hardware transform doesn't support 90° rotations.

        ``rotate == 0`` returns the frame untouched (fast path).
        """
        if frame is None or self.rotate == 0:
            return frame
        if self.rotate == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotate == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotate == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def trigger_autofocus(self, timeout_s: float = 2.0) -> bool:
        """Run a single-shot autofocus cycle on picamera2.

        Behaviour
        ---------
        * Only honoured when ``self.backend == "picamera2"``.  On the
          OpenCV backend this method returns ``False`` immediately so
          callers can call it unconditionally.
        * Issues ``set_controls({"AfMode": Auto, "AfTrigger": 0})``,
          which is libcamera's "kick a one-shot AF pass" recipe.
        * Polls the ``AfStatus`` metadata for up to ``timeout_s``
          seconds, returning as soon as the driver reports a lock
          (``AfStatus == 2`` -- Focused) or the timeout elapses.
        * Re-applies the AF mode the user originally asked for
          (``Continuous`` or ``Manual``) after the trigger so the LIVE
          preview returns to its baseline behaviour.
        * A second concurrent call while one is in flight is a no-op
          and returns the in-flight result.

        Returns ``True`` if focus locked within ``timeout_s``, else
        ``False``.  The caller should still proceed with the capture
        even on ``False`` -- a timed-out AF cycle usually still
        produces a usable frame, just not guaranteed sharp.
        """
        if self.backend != "picamera2":
            return False
        if self._pi_cam is None or not self.is_open:
            return False
        if self._af_in_flight:
            # Another thread / call is already polling; piggyback on it.
            return bool(self._af_locked)

        try:
            from libcamera import controls as _controls  # type: ignore
        except Exception:
            logger.debug("libcamera not importable; skipping trigger_autofocus.")
            return False

        self._af_in_flight = True
        self._af_locked = None
        # Remember the AF mode the user wanted so we can restore it.
        restore_mode = (
            _controls.AfModeEnum.Continuous
            if self.autofocus
            else _controls.AfModeEnum.Manual
        )
        try:
            self._pi_cam.set_controls({"AfMode": _controls.AfModeEnum.Auto})
            # libcamera ignores AfTrigger on the IMX519 but it costs nothing
            # to send it; some firmware revisions honour it.
            try:
                self._pi_cam.set_controls({"AfTrigger": 0})
            except Exception:
                pass

            deadline = time.monotonic() + max(0.1, float(timeout_s))
            while time.monotonic() < deadline:
                # ``capture_metadata`` is non-blocking and returns the
                # most recent frame's controls dict.
                meta = {}
                try:
                    meta = self._pi_cam.capture_metadata() or {}
                except Exception as exc:  # pragma: no cover - driver-specific
                    logger.debug("capture_metadata raised during AF: %s", exc)
                    break
                # libcamera returns enum int values; 1=Idle, 2=Focused,
                # 3=Scanning, 4=Failed. Anything != Focused means we
                # keep waiting (Scanning) or give up (Failed).
                status = meta.get("AfStatus")
                if status == 2:
                    self._af_locked = True
                    break
                if status in (1, 4):
                    if status == 4:
                        # Failed - no point spinning the wheel further.
                        self._af_locked = False
                        break
                time.sleep(0.03)
            else:
                # Loop exhausted without a lock.
                self._af_locked = False
        except Exception as exc:  # pragma: no cover - driver-specific
            logger.warning("trigger_autofocus failed: %s", exc)
            self._af_locked = False
        finally:
            # Restore the user's preferred AF mode so the LIVE preview
            # goes back to either continuous-AF or the manual lens
            # position they configured.
            try:
                restore = {"AfMode": restore_mode}
                if not self.autofocus:
                    restore["LensPosition"] = float(self.lens_position)
                self._pi_cam.set_controls(restore)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to restore AF mode after trigger: %s", exc)
            self._af_in_flight = False

        logger.debug(
            "trigger_autofocus finished; locked=%s (timeout=%ss)",
            self._af_locked, timeout_s,
        )
        return bool(self._af_locked)

    def autofocus_and_lock(self, timeout_s: float = 3.0) -> float:
        """Run one-shot AF, then **lock** the lens where AF converged.

        Unlike :meth:`trigger_autofocus` (which restores the previous AF mode
        afterwards), this reads the ``LensPosition`` the autofocus pass settled
        on and pins the lens there in ``AfMode=Manual`` so it stays put — ideal
        for a document scanner where the page sits at a fixed distance and
        continuous-AF hunting would only introduce blur.

        Side effects: updates ``self.lens_position`` to the locked value and
        sets ``self.autofocus = False`` (via :meth:`set_manual_focus`).

        Returns the locked ``LensPosition`` in dioptres, or ``float('nan')`` on
        the OpenCV backend / a closed camera / when libcamera is unavailable.
        """
        if self.backend != "picamera2" or self._pi_cam is None or not self.is_open:
            return float("nan")
        if self._af_in_flight:
            # Another AF pass is already running; don't stack a second.
            return float(self.lens_position)
        try:
            from libcamera import controls as _controls  # type: ignore
        except Exception:
            logger.debug("libcamera not importable; skipping autofocus_and_lock.")
            return float("nan")

        self._af_in_flight = True
        self._af_locked = None
        locked_pos = float("nan")
        last_lens = float("nan")
        try:
            self._pi_cam.set_controls({"AfMode": _controls.AfModeEnum.Auto})
            try:
                self._pi_cam.set_controls({"AfTrigger": 0})
            except Exception:
                pass

            deadline = time.monotonic() + max(0.1, float(timeout_s))
            while time.monotonic() < deadline:
                try:
                    meta = self._pi_cam.capture_metadata() or {}
                except Exception as exc:  # pragma: no cover - driver-specific
                    logger.debug("capture_metadata raised during AF-lock: %s", exc)
                    break
                lp = meta.get("LensPosition")
                if lp is not None:
                    try:
                        last_lens = float(lp)
                    except (TypeError, ValueError):
                        pass
                status = meta.get("AfStatus")
                if status == 2:            # Focused
                    self._af_locked = True
                    locked_pos = last_lens
                    break
                if status == 4:            # Failed
                    self._af_locked = False
                    break
                time.sleep(0.03)
            else:
                self._af_locked = False
        except Exception as exc:  # pragma: no cover - driver-specific
            logger.warning("autofocus_and_lock failed: %s", exc)
            self._af_locked = False
        finally:
            self._af_in_flight = False

        # Pin the lens.  Prefer the converged position; if AF never reported a
        # LensPosition, fall back to the last one we saw (or the current stored
        # value) so we at least stop continuous hunting and hold *something*.
        target = locked_pos
        if not (target == target):          # NaN check
            target = last_lens if (last_lens == last_lens) else float(self.lens_position)
        applied = self.set_manual_focus(target)
        result = applied if (applied == applied) else target
        logger.info(
            "autofocus_and_lock: locked=%s at %.2f dpt (driver reports %.2f)",
            self._af_locked, target,
            applied if (applied == applied) else float("nan"),
        )
        return result

    def read(self) -> Tuple[bool, np.ndarray]:
        """Return ``(ok, frame)`` mirroring ``VideoCapture.read()``.

        For picamera2 the captured array is already RGB, so we convert to
        BGR to keep the rest of the OpenCV pipeline unchanged.  Frames are
        then rotated by ``self.rotate`` degrees clockwise so portrait-mode
        camera mounts produce portrait-oriented output transparently.

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
            frame_bgr = self._apply_rotate(frame_bgr)
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
        return ok, self._apply_rotate(frame)

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