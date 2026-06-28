"""Reproduce the crash that happens when the user presses D.

It exercises:
  - ScanSession.finish_pdf (PDF build)
  - The post-D screen render where camera.read() return shape matters
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import ScanSession, AppMode, SavedScreen  # noqa: E402

# --- 1. PDF build --------------------------------------------------------------
sess = ScanSession(
    captures_dir=ROOT / "captures" / "scanned",
    output_dir=ROOT / "output",
    scanned_dir=ROOT / "captures" / "scanned",
    lock=threading.Lock(),
)
sess.next_page_n = 5  # pretend we have 4 pages on disk

print("[1] page_count =", sess.page_count())
print("[1] finish_pdf() =", sess.finish_pdf())

# --- 2. camera.read() return shape --------------------------------------------
class FakeCamera:
    def read(self):
        return True, None  # the real return type is (ok, frame)

cam = FakeCamera()
try:
    ok, frame = cam.read()
    print("[2] correct unpacking:", ok, type(frame).__name__)
except Exception as exc:
    print("[2] CORRECT path raised:", type(exc).__name__, exc)

# Now the buggy line from app.py line 572
try:
    _, probe = cam.read()
    print("[2] buggy unpacking: probe =", probe, "type =", type(probe).__name__)
    canvas_w = probe.shape[1]  # this is what app.py does
    print("[2] canvas_w =", canvas_w)
except Exception as exc:
    print("[2] BUG path raised:", type(exc).__name__, exc)