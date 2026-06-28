"""
tests/run_camera_session.py
===========================

End-to-end smoke test that mirrors the real user flow:

1. Open Camera(--source 1).
2. Read 3 frames, run each through DocumentScanner, save to page_1/2/3.jpg.
3. Call ScanSession.finish_pdf()  → expect output/scan_1.pdf.
4. Capture 2 more pages, finish again  → expect output/scan_2.pdf.
5. Verify page counter resets to page_1.jpg after each D.
6. Verify the produced PDFs are valid (header bytes).

Run with:

    .venv\\Scripts\\python.exe tests\\run_camera_session.py

It does NOT touch the OpenCV window — we read frames silently so it can run
headless.  If the camera fails to open, the script exits cleanly with a
diagnostic so the user knows it's a hardware/permission issue, not a code bug.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

# Make ``python tests/run_camera_session.py`` work without installing.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2  # noqa: E402

from app import ScanSession                # noqa: E402
from camera import Camera                  # noqa: E402
from detector import DocumentDetector      # noqa: E402
from scanner import DocumentScanner        # noqa: E402


PROJECT = HERE.parent
CAPS    = PROJECT / "captures"
OUT     = PROJECT / "output"


def grab_valid_frame(camera: Camera, detector: DocumentDetector, max_tries: int = 30) -> tuple:
    """Read frames until DocumentDetector returns a valid document quad.

    Returns (frame, quad).  Raises RuntimeError if no document is ever found.
    """
    for _ in range(max_tries):
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.05)
            continue
        detection = detector.detect(frame)
        if detection is not None and detection.quad is not None:
            return frame, detection.quad
        time.sleep(0.05)
    raise RuntimeError(
        "No document detected in %d frames — hold a page in front of the camera."
        % max_tries
    )


def verify_pdf(path: Path) -> bool:
    """Return True if path looks like a non-empty PDF (magic bytes + size)."""
    if not path.exists() or path.stat().st_size < 100:
        return False
    with open(path, "rb") as fh:
        return fh.read(4) == b"%PDF"


def main() -> int:
    # --- clean any leftovers from earlier runs -----------------------
    for p in CAPS.glob("page_*.jpg"):
        p.unlink()
    for p in OUT.glob("scan_*.pdf"):
        p.unlink()

    print("[harness] opening camera source=1 ...")
    try:
        camera = Camera("1", 1280, 720, backend="opencv")
    except RuntimeError as exc:
        print(f"[harness] FAIL: {exc}")
        return 2
    print("[harness] camera opened OK.")

    detector = DocumentDetector()
    scanner  = DocumentScanner(apply_threshold=True)
    session  = ScanSession(captures_dir=CAPS, output_dir=OUT)

    # --- session 1: capture 3 pages ----------------------------------
    print("[harness] session 1 — capturing 3 pages (simulated C x3) ...")
    for i in range(1, 4):
        try:
            frame, quad = grab_valid_frame(camera, detector)
        except RuntimeError as exc:
            print(f"[harness] FAIL: {exc}")
            camera.release()
            return 3
        warped = scanner.scan(frame, quad)
        path = session.page_filename()
        cv2.imwrite(str(path), warped)
        print(f"  - saved {path.name}  (session count = {session.page_count()})")

    assert session.page_count() == 3, "expected 3 pages in session 1"

    # --- press D ------------------------------------------------------
    print("[harness] pressing D → finish_pdf ...")
    saved = session.finish_pdf()
    if saved is None:
        print("[harness] FAIL: finish_pdf returned None")
        camera.release()
        return 4
    print(f"  - produced {saved.name} ({saved.stat().st_size} bytes)")
    assert verify_pdf(saved), f"{saved} is not a valid PDF"
    assert saved.name == "scan_1.pdf", f"expected scan_1.pdf, got {saved.name}"
    assert session.page_count() == 0, "counter should reset after D"

    # --- session 2: capture 2 pages ----------------------------------
    print("[harness] session 2 — capturing 2 pages ...")
    p_next = session.page_filename()
    assert p_next.name == "page_1.jpg", f"counter did not reset: {p_next.name}"
    for i in range(2):
        frame, quad = grab_valid_frame(camera, detector)
        warped = scanner.scan(frame, quad)
        path = session.page_filename()
        cv2.imwrite(str(path), warped)
        print(f"  - saved {path.name}  (session count = {session.page_count()})")

    saved2 = session.finish_pdf()
    assert saved2 is not None and saved2.name == "scan_2.pdf"
    assert verify_pdf(saved2)
    print(f"  - produced {saved2.name} ({saved2.stat().st_size} bytes)")

    # --- final report ------------------------------------------------
    pages_left = sorted(p.name for p in CAPS.glob("page_*.jpg"))
    pdfs = sorted(p.name for p in OUT.glob("scan_*.pdf"))
    print()
    print("=== RESULT ===")
    print(f"  pages still on disk : {pages_left}")
    print(f"  saved PDFs          : {pdfs}")
    print(f"  scan_1.pdf valid    : {verify_pdf(OUT / 'scan_1.pdf')}")
    print(f"  scan_2.pdf valid    : {verify_pdf(OUT / 'scan_2.pdf')}")

    camera.release()
    return 0 if len(pdfs) == 2 else 5


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as exc:                      # pragma: no cover - defensive
        print(f"[harness] CRASH: {exc!r}")
        rc = 1
    print(f"[harness] exit {rc}")
    sys.exit(rc)
