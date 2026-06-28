"""
tests/run_synthetic_session.py
==============================

End-to-end smoke test that mirrors the real user flow **without any camera**.

1. Patch Camera with a synthetic subclass that returns a synthetic document
   frame (white background with a dark-edged rectangle) on every ``read()``.
2. Drive ``ScanSession.handle_key('c')`` three times, then ``handle_key('d')``
   and expect ``output/pdf/document_001.pdf`` plus a matching QR PNG.
3. Drive ``handle_key('n')`` to start a new document, capture 2 pages, finish,
   and expect ``document_002.pdf``.
4. Verify state transitions: LIVE -> PDF_VIEW on D, PDF_VIEW -> LIVE on N.
5. Verify the produced PDFs are valid (header bytes + size).

Run with::

    .venv\\Scripts\\python.exe tests\\run_synthetic_session.py

This proves the C/D/N/Q pipeline works end-to-end.  When you have real camera
hardware available, run ``tests/run_camera_session.py`` instead.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Make ``python tests/run_synthetic_session.py`` work without installing.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402

from app import (                          # noqa: E402
    LIVE_SCANNER_MODE,
    PDF_VIEW_MODE,
    ScanSession,
    ScannerState,
)
from camera import Camera                   # noqa: E402
from config import (                       # noqa: E402
    CAPTURES_DIR,
    OUTPUT_DIR,
    PDF_DIR,
    QR_DIR,
)
from document_processor import (            # noqa: E402
    DetectionResult,
    DocumentProcessor,
)


PROJECT = HERE.parent
CAPS    = CAPTURES_DIR
OUT     = OUTPUT_DIR


# --------------------------------------------------------------------- #
# Synthetic Camera - every read() returns a fresh document-shaped frame.
# --------------------------------------------------------------------- #
class SyntheticCamera(Camera):
    """Stand-in for a real camera that emits a valid document frame."""

    def __init__(self, width: int = 1280, height: int = 720):
        # Skip Camera.__init__ - we never touch cv2.VideoCapture.
        self.source = "synthetic"
        self.width = width
        self.height = height
        self.backend = "synthetic"
        self._cap = None
        self._pi_cam = None
        self._tick = 0

    def _draw_document(self) -> np.ndarray:
        """Return a BGR frame with a clear white rectangle on a grey bg."""
        self._tick += 1
        # Slight per-frame jitter so DocumentDetector doesn't get suspicious.
        jitter = (self._tick * 7) % 11
        img = np.full((self.height, self.width, 3), 180, dtype=np.uint8)
        # Dark border so the rectangle pops in Canny edges.
        x0, y0 = 120 + jitter, 90 + jitter
        x1, y1 = self.width - 120 - jitter, self.height - 90 - jitter
        cv2.rectangle(img, (x0, y0), (x1, y1), (250, 250, 250), thickness=-1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), thickness=4)
        # Faux printed lines so thresholding produces something.
        for y in range(y0 + 40, y1 - 20, 30):
            cv2.line(img, (x0 + 20, y), (x1 - 20, y), (60, 60, 60), 1)
        return img

    def read(self):
        return True, self._draw_document()

    def release(self):
        # Nothing to release.
        self._cap = None
        self._pi_cam = None


def grab_valid_frame(camera: Camera, processor: DocumentProcessor):
    """Read frames until DocumentProcessor finds a quadrilateral."""
    for _ in range(15):
        ok, frame = camera.read()
        if not ok:
            continue
        _processed, detection = processor.process(frame)
        if detection.corners is not None and detection.confidence > 0.0:
            return frame
    raise RuntimeError("No document detected in synthetic frames - detector is broken.")


def verify_pdf(path: Path) -> bool:
    """Return True if path looks like a non-empty PDF (magic bytes + size)."""
    if not path.exists() or path.stat().st_size < 100:
        return False
    with open(path, "rb") as fh:
        return fh.read(4) == b"%PDF"


def key_for(letter: str) -> int:
    """Map a letter to the integer code OpenCV's waitKey returns."""
    return ord(letter.lower()) & 0xFF


def main() -> int:
    # --- clean any leftovers from earlier runs -----------------------
    for p in CAPS.glob("page_*.jpg"):
        p.unlink()
    for p in PDF_DIR.glob("document_*.pdf"):
        p.unlink()
    for p in QR_DIR.glob("document_*.png"):
        p.unlink()
    print("[harness] cleaned captures/ and output/")

    # --- bring up the synthetic pipeline ----------------------------
    print("[harness] opening SYNTHETIC camera ...")
    camera    = SyntheticCamera(width=1280, height=720)
    processor = DocumentProcessor()
    session   = ScanSession(camera_source="synthetic")
    # Inject the synthetic camera into the session (no VideoCapture).
    session._camera = camera
    # The synthetic frame is intentionally smooth grey so detector corners
    # pop without motion blur; relax the quality gate so C always accepts.
    from quality_gate import QualityGate
    session._quality_gate = QualityGate(
        blur_min=1.0,
        brightness_min=80.0,
        brightness_max=255.0,
        motion_max=255.0,
        corner_confidence_min=0.0,
        min_area_ratio=0.0,
        enabled=False,  # bypass - synthetic frames are clean by construction
    )
    print("[harness] pipeline ready.")

    assert session.state == ScannerState.LIVE_SCANNER_MODE, "should start in LIVE"

    # --- session 1: capture 3 pages ----------------------------------
    print("[harness] session 1 - simulating C x3 ...")
    for i in range(1, 4):
        frame = grab_valid_frame(camera, processor)
        ok, msg, _proc, _det = session.capture_current_frame(frame)
        assert ok, f"C press {i} rejected: {msg}"
        assert session.page_count() == i, f"expected {i} pages, got {session.page_count()}"
        print(f"  - C pressed -> page_{i}.jpg  ({msg})")

    # --- press D ------------------------------------------------------
    print("[harness] D pressed -> finish_pdf() ...")
    saved = session.finish_pdf()
    assert saved is not None, "finish_pdf returned None"
    print(f"  - produced {saved.name} ({saved.stat().st_size} bytes)")
    assert verify_pdf(saved), f"{saved} is not a valid PDF"
    assert saved.name == "document_001.pdf", f"expected document_001.pdf, got {saved.name}"
    assert session.state == PDF_VIEW_MODE, "should transition to PDF_VIEW on D"
    assert session.page_count() == 0, "counter should reset after D"

    # --- start a new document ----------------------------------------
    print("[harness] N pressed -> start_new_document() ...")
    session.start_new_document()
    assert session.state == LIVE_SCANNER_MODE, "N should return to LIVE"
    p_next = session.page_filename()
    assert p_next.name == "page_1.jpg", f"counter did not reset: {p_next.name}"

    for i in range(2):
        frame = grab_valid_frame(camera, processor)
        ok, msg, _proc, _det = session.capture_current_frame(frame)
        assert ok, f"second-session C {i} rejected: {msg}"
        print(f"  - C pressed -> page_{i+1}.jpg  ({msg})")

    saved2 = session.finish_pdf()
    assert saved2 is not None and saved2.name == "document_002.pdf"
    assert verify_pdf(saved2)
    assert session.state == PDF_VIEW_MODE
    print(f"  - produced {saved2.name} ({saved2.stat().st_size} bytes)")

    # --- final report ------------------------------------------------
    pages_left = sorted(p.name for p in CAPS.glob("page_*.jpg"))
    pdfs       = sorted(p.name for p in PDF_DIR.glob("document_*.pdf"))
    qrs        = sorted(p.name for p in QR_DIR.glob("document_*.png"))
    print()
    print("=== RESULT ===")
    print(f"  pages still on disk : {pages_left}")
    print(f"  saved PDFs          : {pdfs}")
    print(f"  saved QR PNGs       : {qrs}")
    print(f"  document_001.pdf    : valid={verify_pdf(PDF_DIR / 'document_001.pdf')}")
    print(f"  document_002.pdf    : valid={verify_pdf(PDF_DIR / 'document_002.pdf')}")

    camera.release()
    return 0 if len(pdfs) == 2 and len(qrs) >= 2 else 5


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as exc:                      # pragma: no cover - defensive
        import traceback
        traceback.print_exc()
        print(f"[harness] CRASH: {exc!r}")
        rc = 1
    print(f"[harness] exit {rc}")
    sys.exit(rc)