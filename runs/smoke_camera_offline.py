"""Smoke test for the camera-offline recovery path.

We point the camera at a deliberately unreachable URL (the loopback on a
port nothing is listening on) and assert that:

* ``Camera(...)`` does NOT raise.
* ``is_open`` is False and ``last_open_error`` is populated.
* ``ScanSession.camera_status()`` reports offline + a sane retry countdown.
* The LIVE render path returns the dedicated "camera not found" canvas
  instead of a crash traceback.

Run with the project venv:

    .venv/Scripts/python.exe runs/smoke_camera_offline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the project root is importable when invoked as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from camera import Camera
from app import ScanSession


def expect(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  ok  - {msg}")


def main() -> None:
    bad_url = "http://127.0.0.1:1/video"  # nothing listens here

    # --- 1. Camera constructor should not raise on a dead source. --- #
    cam = Camera(source=bad_url, width=640, height=480)
    expect(cam.is_open is False, "Camera marked offline for unreachable URL")
    expect(cam.last_open_error is not None, "last_open_error populated")
    expect(cam._cap is None, "no leaked VideoCapture handle")

    # --- 2. read() must return (False, empty) instead of raising. --- #
    ok, frame = cam.read()
    expect(ok is False, "read() reports failure when offline")
    expect(isinstance(frame, np.ndarray), "read() returns a numpy frame")
    expect(frame.shape == (480, 640, 3), "placeholder frame uses configured geometry")

    # --- 3. ScanSession integration: status + render. --- #
    session = ScanSession(camera_source=bad_url, camera_width=640, camera_height=480)
    # Force the lazy camera property to materialise; this is what
    # would have crashed before the fix.
    cam2 = session.camera
    expect(cam2.is_open is False, "Session camera is offline")

    status = session.camera_status()
    expect(status["online"] is False, "camera_status reports offline")
    expect(status["source"] == bad_url, "camera_status echoes source")
    expect("error" in status and status["error"], "camera_status carries error")

    # Tick the watchdog: first call schedules the next retry ~3s out.
    session._ensure_camera_alive()
    expect(session._next_camera_retry_at > 0.0, "retry deadline scheduled")

    # Render the offline banner (this used to crash).
    canvas = session._render_live(None)
    expect(canvas.shape == (480, 640, 3), "offline canvas uses configured geometry")
    expect(canvas.dtype == np.uint8, "offline canvas dtype is uint8")

    # The banner should mention "CAMERA NOT FOUND" somewhere in the
    # pixels we painted via cv2.putText. We sample a row at the title
    # position and confirm it's not all-zero (i.e. text was drawn).
    band = canvas[35:55, 15:300, :]
    expect(int(band.max()) > 0, "offline banner has non-zero text pixels")

    # --- 4. Fast-forward time and confirm a reopen attempt fires. --- #
    # Patch ``time.monotonic`` so we don't have to sleep 3 real seconds.
    import app as app_mod
    base = time.monotonic()
    app_mod.time.monotonic = lambda: base + 100.0  # far past the deadline
    try:
        session._ensure_camera_alive()
    finally:
        app_mod.time.monotonic = time.monotonic
    # We expect a fresh deadline was queued (still offline, retry scheduled).
    expect(session._next_camera_retry_at > base + 50.0,
           "next retry deadline was pushed forward after attempt")

    print("\nALL CAMERA-OFFLINE SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()