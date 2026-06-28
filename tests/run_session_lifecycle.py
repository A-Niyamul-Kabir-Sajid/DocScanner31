"""
tests/run_session_lifecycle.py
==============================

End-to-end smoke test that exercises the *full* C / D / N / Q loop without
a camera.  It reuses :class:`SyntheticCamera` from ``run_synthetic_session``
so the detector always finds a quad, plus the on-window PDF-saved canvas
renderer so we can assert its canvas is well-formed without launching the
OpenCV window itself.

Sequence verified
-----------------
1. ``handle_key('c')`` x3 -> ``page_1..3.jpg`` saved.
2. ``handle_key('d')`` -> ``document_001.pdf`` written, page counter resets,
   state moves to ``PDF_VIEW_MODE``.  The PDF_VIEW canvas is rendered and
   asserted to be non-blank and the correct shape.
3. ``handle_key('n')`` -> page counter resets to ``page_1.jpg`` and state
   returns to ``LIVE_SCANNER_MODE`` (no PDF produced).
4. ``handle_key('c')`` x2 -> ``page_1..2.jpg`` written.
5. ``handle_key('d')`` -> ``document_002.pdf`` written.
6. ``handle_key('c')`` x1, then ``handle_key('q')`` -> modal opens;
   ``handle_key('y')`` -> auto-finish emits ``document_003.pdf`` and
   ``quit_requested`` flips to True.

Run with::

    .venv\\Scripts\\python.exe tests\\run_session_lifecycle.py

Exits 0 on success, non-zero on the first failed assertion.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Make ``python tests/run_session_lifecycle.py`` work without installing.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402

from app import ScanSession, ScannerState   # noqa: E402
from camera import Camera                   # noqa: E402
from config import PDF_DIR, QR_DIR, SCANNED_DIR  # noqa: E402
from document_processor import DocumentProcessor  # noqa: E402

# Reuse the synthetic frame generator from the sibling harness so we don't
# duplicate the "what does a document look like" code.
from run_synthetic_session import (         # noqa: E402
    SyntheticCamera,
    grab_valid_frame,
    verify_pdf,
)


PROJECT = HERE.parent


def reset_artifacts() -> None:
    """Wipe captures/, output/pdf/document_*.pdf and output/qr/document_*.png."""
    if SCANNED_DIR.exists():
        for p in SCANNED_DIR.glob("page_*.jpg"):
            try:
                p.unlink()
            except OSError:
                pass
    if PDF_DIR.exists():
        for p in PDF_DIR.glob("document_*.pdf"):
            try:
                p.unlink()
            except OSError:
                pass
    if QR_DIR.exists():
        for p in QR_DIR.glob("document_*.png"):
            try:
                p.unlink()
            except OSError:
                pass
    print("[lifecycle] cleaned captures/, output/pdf/, output/qr/")


def build_session() -> tuple:
    """Return (camera, processor, session) with the synthetic pipeline wired up."""
    camera = SyntheticCamera(width=1280, height=720)
    processor = DocumentProcessor()
    session = ScanSession(camera_source="synthetic")

    # Inject the synthetic camera so the session never touches VideoCapture.
    session._camera = camera
    # The synthetic frame is intentionally smooth grey so the detector corners
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
    return camera, processor, session


def press_letter(letter: str) -> int:
    """Map a letter to the integer code OpenCV's waitKey would return."""
    return ord(letter.lower()) & 0xFF


def main() -> int:
    reset_artifacts()
    camera, processor, session = build_session()
    print("[lifecycle] pipeline ready.")

    assert session.state == ScannerState.LIVE_SCANNER_MODE, "must start in LIVE"

    # --- session 1: C x3 -> D -----------------------------------------
    print("[lifecycle] session 1 - C x3 ...")
    for i in range(1, 4):
        frame = grab_valid_frame(camera, processor)
        session.handle_key(press_letter("c"))
        # capture_current_frame() reads its own frame from the camera; the
        # SyntheticCamera always returns a valid one so we don't have to
        # feed it explicitly.  The first-hand call above used a *processor*
        # only to confirm the frame is detectable - not to actually capture.
        assert session.page_count() == i, (
            f"expected {i} pages after C, got {session.page_count()}"
        )
        print(f"  - C pressed -> pages={session.page_count()}  msg={session.last_message!r}")

    print("[lifecycle] D pressed -> finish_pdf() ...")
    session.handle_key(press_letter("d"))
    saved1 = session.last_pdf_path
    assert saved1 is not None, "D did not produce a PDF path"
    assert verify_pdf(saved1), f"{saved1} is not a valid PDF"
    assert saved1.name == "document_001.pdf", f"expected document_001.pdf, got {saved1.name}"
    assert session.state == ScannerState.PDF_VIEW_MODE, "should be PDF_VIEW after D"
    assert session.page_count() == 0, "counter should reset after D"
    print(f"  - produced {saved1.name} ({saved1.stat().st_size} bytes)")

    # Render the PDF-saved canvas and assert it's well-formed (replaces the
    # legacy _build_pdf_saved_canvas helper with the new render() path).
    canvas = session.render()
    assert isinstance(canvas, np.ndarray), "PDF_VIEW canvas must be an ndarray"
    assert canvas.ndim == 3 and canvas.shape[2] == 3, "canvas must be BGR"
    h, w = canvas.shape[:2]
    assert (h, w) == (camera.height, camera.width), (
        f"unexpected canvas shape {canvas.shape[:2]} vs camera {(camera.height, camera.width)}"
    )
    non_black = int(np.sum(canvas.sum(axis=2) > 30))
    assert non_black > 1000, "PDF_VIEW canvas looks blank - HUD/text not drawn"
    print(f"  - PDF_VIEW canvas shape={canvas.shape} non-black px={non_black}")

    # --- session 2: N -> C x2 -> D -----------------------------------
    print("[lifecycle] N pressed -> start_new_document() ...")
    session.handle_key(press_letter("n"))
    assert session.state == ScannerState.LIVE_SCANNER_MODE, "N must return to LIVE"
    assert session.page_filename().name == "page_1.jpg", (
        f"counter not reset: {session.page_filename().name}"
    )
    # PDF list should still contain exactly the first document - no new PDF
    # was produced by the N keypress.
    pre_pdfs = sorted(p.name for p in PDF_DIR.glob("document_*.pdf"))
    assert pre_pdfs == ["document_001.pdf"], f"N unexpectedly produced PDFs: {pre_pdfs}"

    for i in range(1, 3):
        _ = grab_valid_frame(camera, processor)
        session.handle_key(press_letter("c"))
        assert session.page_count() == i, (
            f"session-2 expected {i} pages, got {session.page_count()}"
        )
        print(f"  - C pressed -> pages={session.page_count()}")

    session.handle_key(press_letter("d"))
    saved2 = session.last_pdf_path
    assert saved2 is not None and saved2.name == "document_002.pdf", (
        f"expected document_002.pdf, got {None if saved2 is None else saved2.name}"
    )
    assert verify_pdf(saved2)
    assert session.state == ScannerState.PDF_VIEW_MODE
    print(f"  - produced {saved2.name} ({saved2.stat().st_size} bytes)")

    # --- session 3: C x1 -> Q (modal) -> Y ----------------------------
    print("[lifecycle] session 3 - C x1 then Q ...")
    # Force back to LIVE so we can capture a fresh page.
    session.handle_key(press_letter("n"))
    assert session.state == ScannerState.LIVE_SCANNER_MODE

    _ = grab_valid_frame(camera, processor)
    session.handle_key(press_letter("c"))
    assert session.page_count() == 1, "session-3 should have 1 page before Q"

    session.handle_key(press_letter("q"))
    assert session.show_exit_modal, "Q should open the Exit modal"
    assert "quit" in session.last_message.lower(), (
        f"modal last_message unexpected: {session.last_message!r}"
    )
    print(f"  - modal open   msg={session.last_message!r}")

    # Modal canvas should still be well-formed even in LIVE.
    live_canvas = session.render()
    assert isinstance(live_canvas, np.ndarray) and live_canvas.shape == (camera.height, camera.width, 3)

    session.handle_key(press_letter("y"))
    assert not session.show_exit_modal, "Y should dismiss the modal"
    assert session.quit_requested, "Y should set quit_requested"
    # Q with pages must auto-finish -> document_003.pdf.
    saved3 = session.last_pdf_path
    assert saved3 is not None and saved3.name == "document_003.pdf", (
        f"Y should have auto-finished -> document_003.pdf, got "
        f"{None if saved3 is None else saved3.name}"
    )
    assert verify_pdf(saved3)
    print(f"  - auto-saved {saved3.name} ({saved3.stat().st_size} bytes)")

    # --- final report ------------------------------------------------
    pages_left = sorted(p.name for p in SCANNED_DIR.glob("page_*.jpg"))
    pdfs = sorted(p.name for p in PDF_DIR.glob("document_*.pdf"))
    qrs = sorted(p.name for p in QR_DIR.glob("document_*.png"))
    print()
    print("=== LIFECYCLE RESULT ===")
    print(f"  pages still on disk : {pages_left}")
    print(f"  saved PDFs          : {pdfs}")
    print(f"  saved QR PNGs       : {qrs}")
    print(f"  document_001.pdf    : valid={verify_pdf(PDF_DIR / 'document_001.pdf')}")
    print(f"  document_002.pdf    : valid={verify_pdf(PDF_DIR / 'document_002.pdf')}")
    print(f"  document_003.pdf    : valid={verify_pdf(PDF_DIR / 'document_003.pdf')}")
    print(f"  final state         : {session.state.name}")
    print(f"  quit_requested      : {session.quit_requested}")

    camera.release()
    ok = (
        pdfs == ["document_001.pdf", "document_002.pdf", "document_003.pdf"]
        and len(qrs) >= 3
        and session.quit_requested
        and session.state == ScannerState.PDF_VIEW_MODE
    )
    return 0 if ok else 6


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    print(f"[lifecycle] exit {rc}")
    sys.exit(rc)
