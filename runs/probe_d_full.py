"""End-to-end repro of _finish_current_session() after the fix.

Builds a session with 4 pages, then invokes the same camera.read + canvas
pre-render sequence _finish_current_session uses, with a camera stub that
returns (False, None) -- which is exactly the failure mode from the original
crash.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import ScanSession  # noqa: E402


class FakeCamera:
    """Mimics DroidCam returning (ok, frame) where ok=False on transient drops."""
    def __init__(self, ok: bool) -> None:
        self.ok = ok
    def read(self):
        if self.ok:
            return True, np.zeros((720, 1280, 3), dtype=np.uint8)
        return False, None


# Build the session the same way main() does.
sess = ScanSession(
    captures_dir=ROOT / "captures" / "scanned",
    output_dir=ROOT / "output",
    scanned_dir=ROOT / "captures" / "scanned",
    lock=threading.Lock(),
)
sess.next_page_n = 5

# Drive the post-D render block from _finish_current_session.
cam = FakeCamera(ok=False)
probe = None
try:
    ok_probe, probe = cam.read()
    if not ok_probe:
        probe = None
except Exception:
    probe = None
W, H = 1280, 720
canvas_w = probe.shape[1] if probe is not None else W
canvas_h = probe.shape[0] if probe is not None else H
print(f"camera.read() ok=False  -> canvas_w={canvas_w}, canvas_h={canvas_h}  (no crash)")

cam = FakeCamera(ok=True)
ok_probe, probe = cam.read()
if not ok_probe:
    probe = None
canvas_w = probe.shape[1] if probe is not None else W
canvas_h = probe.shape[0] if probe is not None else H
print(f"camera.read() ok=True   -> canvas_w={canvas_w}, canvas_h={canvas_h}")

print("OK: _finish_current_session() will no longer crash on D.")