"""Smoke test for the picamera2 recovery path added to camera.py.

We can't actually instantiate ``Picamera2`` on a Windows / macOS dev box, so
this probe patches ``sys.modules`` with a stub and exercises:

1. ``Camera.try_reopen()`` returns True after a fresh ``Picamera2()``.
2. ``is_open`` flips to ``True`` after a successful ``read()``.
3. After a failed ``capture_array()``, ``try_reopen()`` tears down the
   failed pipeline and brings up a fresh one (verified by a counter).
4. The first open path no longer leaves ``is_open = False`` for picamera2.

Run from the project root with the venv active:

    python runs/probe_picamera_reopen.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# Allow running this file directly (it lives in runs/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Stub picamera2 + libcamera so ``camera._open_picamera2()`` can import.
# --------------------------------------------------------------------------- #
class _StubCam:
    instances = 0

    def __init__(self):
        type(self).instances += 1
        self._explode_next = False
        self._started = False

    def create_video_configuration(self, **kw):
        return {"fake": True, **kw}

    def configure(self, config):
        pass

    def start(self):
        self._started = True

    def set_controls(self, controls):
        pass

    def capture_array(self):
        if self._explode_next:
            self._explode_next = False
            raise RuntimeError("simulated sensor stall")
        import numpy as np
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def stop(self):
        self._started = False

    def close(self):
        pass


class _AfModeEnum:
    Manual = 0
    Continuous = 1


class _Controls:
    AfModeEnum = _AfModeEnum


picamera2_mod = types.ModuleType("picamera2")
picamera2_mod.Picamera2 = _StubCam
sys.modules["picamera2"] = picamera2_mod

libcamera_mod = types.ModuleType("libcamera")
libcamera_controls_mod = types.ModuleType("libcamera.controls")
libcamera_controls_mod.AfModeEnum = _AfModeEnum
libcamera_mod.controls = libcamera_controls_mod
sys.modules["libcamera"] = libcamera_mod
sys.modules["libcamera.controls"] = libcamera_controls_mod


# --------------------------------------------------------------------------- #
# Now safely import the real Camera class and exercise it.
# --------------------------------------------------------------------------- #
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from camera import Camera  # noqa: E402


def banner(msg: str) -> None:
    print("\n=== " + msg + " ===")


def main() -> None:
    _StubCam.instances = 0

    banner("Open picamera2 backend")
    cam = Camera(source=0, width=1280, height=720, backend="picamera2")
    assert cam.is_open, "is_open should be True after a fresh open"
    assert _StubCam.instances == 1, f"expected one Picamera2 ctor, got {_StubCam.instances}"
    print("OK  is_open=True after _open_picamera2()")

    banner("read() returns a frame and latches _last_read_ok")
    ok, frame = cam.read()
    assert ok and isinstance(frame, np.ndarray) and frame.shape == (720, 1280, 3), (
        f"read() returned unexpected payload: ok={ok} shape={frame.shape if frame is not None else None}"
    )
    assert cam._last_read_ok, "_last_read_ok should be True after a successful read"
    print("OK  read() ok=True shape=(720,1280,3) _last_read_ok=True")

    banner("Simulated capture_array failure flips is_open=False")
    cam._pi_cam._explode_next = True
    ok, frame = cam.read()
    assert not ok, "read() should report failure when capture_array raises"
    assert not cam.is_open, "is_open should flip False after a raise"
    print("OK  read() ok=False, is_open=False after simulated stall")

    banner("try_reopen() rebuilds the picamera2 pipeline")
    cam._pi_cam._explode_next = False  # the fresh instance must not stall
    reopened = cam.try_reopen()
    assert reopened, "try_reopen() should return True after a successful rebuild"
    assert cam.is_open, "is_open must be True after try_reopen()"
    assert _StubCam.instances == 2, (
        f"expected two Picamera2 ctor calls after recovery, got {_StubCam.instances}"
    )
    print(f"OK  try_reopen() returned True, Picamera2 instances={_StubCam.instances}")

    banner("Post-recovery read() works again")
    ok, frame = cam.read()
    assert ok, "read() must succeed after try_reopen()"
    assert cam._last_read_ok, "_last_read_ok must be True"
    print("OK  read() ok=True after recovery")

    banner("try_reopen() is a no-op when the camera is healthy")
    before = _StubCam.instances
    assert cam.try_reopen(), "healthy try_reopen should return True"
    assert _StubCam.instances == before, "healthy try_reopen must not rebuild"
    print("OK  healthy try_reopen did not rebuild the pipeline")

    banner("release() drops is_open so a future reopen has to work")
    cam.release()
    assert not cam.is_open
    assert not cam._last_read_ok
    print("OK  release() cleared is_open and _last_read_ok")

    banner("Post-release try_reopen() rebuilds again")
    reopened = cam.try_reopen()
    assert reopened, "try_reopen() after release() should succeed"
    assert cam.is_open
    assert _StubCam.instances == 3, (
        f"expected three Picamera2 ctor calls after release+reopen, got {_StubCam.instances}"
    )
    print(f"OK  try_reopen() after release() returned True, instances={_StubCam.instances}")

    cam.release()
    print("\nALL PICAMERA2 RECOVERY PROBES PASSED")


if __name__ == "__main__":
    main()
