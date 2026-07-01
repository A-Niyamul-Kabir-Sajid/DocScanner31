"""Smart Document Scanner - top-level application + LIVE/PDF_VIEW FSM.

The scanner has two states:

* ``LIVE_SCANNER_MODE`` - the camera is live and the user can press
  ``C`` to capture, ``D`` to finish the current document, ``N`` is
  ignored and ``Q`` asks to quit.

* ``PDF_VIEW_MODE`` - the camera frame is **off**; the OpenCV window
  shows the just-saved PDF's filename, page count, QR preview, and a
  download URL.  From here the user presses ``N`` to start a *new*
  document, or ``Q`` to quit.

Key map (case-insensitive):

    LIVE_SCANNER_MODE
        C - capture current frame (quality-gate enforced)
        D - flush pages -> PDF + QR + Flask server -> PDF_VIEW_MODE
        N - rejected (must D first)
        Q - if any pages captured: auto D, then modal Exit? Y/N
            else: modal Exit? Y/N

    PDF_VIEW_MODE
        N - reset pages, bump doc counter, return to LIVE_SCANNER_MODE
        Q - modal Exit? Y/N
        C/D - ignored

Modal Exit dialog:
    Y - quit the application
    N - dismiss the dialog and stay in the current state
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from camera import Camera
from config import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_WEB_HOST,
    DEFAULT_WEB_PORT,
    DOCUMENT_COUNTER_START,
    DOCUMENT_PREFIX,
    OUTPUT_DIR,
    PAGE_PREFIX,
    PDF_DIR,
    QR_DIR,
    RAW_DIR,
    RAW_PREFIX,
    SCAN_MODE,
    SCANNED_DIR,
    WINDOW_TITLE,
)
from document_processor import DetectionResult, DocumentProcessor
from flask_server import FlaskServer
from image_grid import stack_images
from pdf_builder import PDFBuilder, document_filename
from quality_gate import QualityGate, QualityReport
from qr_generator import QRGenerator

# --------------------------------------------------------------------------- #
# 8-panel debug-grid layout (matches the Murtaza-style pipeline view).
# --------------------------------------------------------------------------- #
DEBUG_GRID_LABELS: List[List[str]] = [
    ["Original", "Gray", "Threshold", "Contours"],
    ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"],
]
DEBUG_GRID_SCALE: float = 0.5

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# FSM states
# --------------------------------------------------------------------------- #
class ScannerState(str, Enum):
    LIVE_SCANNER_MODE = "LIVE_SCANNER_MODE"
    PDF_VIEW_MODE = "PDF_VIEW_MODE"


# Legacy aliases so older tests keep working.
LIVE_SCANNER_MODE = ScannerState.LIVE_SCANNER_MODE
PDF_VIEW_MODE = ScannerState.PDF_VIEW_MODE


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple,
    *,
    color=(255, 255, 255),
    bg: Optional[tuple] = None,
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """Draw text with an optional dark background pill for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    if bg is not None:
        cv2.rectangle(
            canvas,
            (x - 4, y - th - 4),
            (x + tw + 4, y + baseline + 4),
            bg,
            thickness=-1,
        )
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Grid-builder helpers (used by _render_live)
# --------------------------------------------------------------------------- #
def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Promote a single-channel stage to 3-channel BGR for stacking."""
    if img is None:
        raise ValueError("_to_bgr received None")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _make_thumb(img: Optional[np.ndarray], size: int = 160) -> Optional[np.ndarray]:
    """Return a centred square thumbnail (or ``None`` if input is unusable)."""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    scale = size / max(h, w, 1)
    if scale <= 0:
        return None
    resized = cv2.resize(
        img,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def grid_label_bottom(panels: Sequence[Sequence[object]], scale: float) -> int:
    """Y-pixel of the bottom of the first (top) row's label header strip."""
    if not panels or not panels[0]:
        return 0
    first = next((t for t in panels[0] if t is not None), None)
    if first is None:
        return 0
    tile_h = int(first.shape[0] * scale)
    return tile_h + 30  # 30 = label header strip height (matches stack_images)


# --------------------------------------------------------------------------- #
@dataclass
class ScanSession:
    """Owns the LIVE / PDF_VIEW state machine and all collaborators.

    Construction does **not** open the camera - call :meth:`run` for that.
    Tests construct a :class:`ScanSession` directly and call
    :meth:`finish_pdf`, :meth:`start_new_document`, etc. without spinning up
    the camera loop.
    """

    captures_dir: Path = SCANNED_DIR
    output_dir: Path = OUTPUT_DIR
    camera_source: object = 0
    camera_backend: str = "opencv"
    camera_width: int = DEFAULT_CAMERA_WIDTH
    camera_height: int = DEFAULT_CAMERA_HEIGHT
    web_host: str = DEFAULT_WEB_HOST
    web_port: int = DEFAULT_WEB_PORT
    scan_mode: str = SCAN_MODE

    # Bookkeeping populated by __post_init__.
    scanned_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    pdf_dir: Path = field(init=False)
    qr_dir: Path = field(init=False)
    pages: List[np.ndarray] = field(default_factory=list, init=False)
    doc_counter: int = field(default=DOCUMENT_COUNTER_START, init=False)
    state: ScannerState = field(default=ScannerState.LIVE_SCANNER_MODE, init=False)
    last_message: str = field(default="", init=False)
    last_quality: Optional[QualityReport] = field(default=None, init=False)
    last_pdf_path: Optional[Path] = field(default=None, init=False)
    last_qr_path: Optional[Path] = field(default=None, init=False)
    show_exit_modal: bool = field(default=False, init=False)
    quit_requested: bool = field(default=False, init=False)

    # Collaborators (created on demand so tests can replace them).
    _camera: Optional[Camera] = field(default=None, init=False)
    _processor: Optional[DocumentProcessor] = field(default=None, init=False)
    _quality_gate: Optional[QualityGate] = field(default=None, init=False)
    _pdf_builder: Optional[PDFBuilder] = field(default=None, init=False)
    _qr_generator: Optional[QRGenerator] = field(default=None, init=False)
    _flask_server: Optional[FlaskServer] = field(default=None, init=False)
    _on_finish_callbacks: List[Callable[[Path], None]] = field(
        default_factory=list, init=False
    )

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.captures_dir = Path(self.captures_dir)
        self.output_dir = Path(self.output_dir)
        self.scanned_dir = self.captures_dir
        self.raw_dir = self.captures_dir.parent / "raw"
        self.pdf_dir = self.output_dir / "pdf"
        self.qr_dir = self.output_dir / "qr"
        for d in (self.captures_dir, self.scanned_dir, self.raw_dir, self.output_dir, self.pdf_dir, self.qr_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Collaborator accessors (lazy so tests can monkey-patch).
    # ------------------------------------------------------------------ #
    @property
    def camera(self) -> Camera:
        if self._camera is None:
            self._camera = Camera(
                source=self.camera_source,
                width=self.camera_width,
                height=self.camera_height,
                backend=self.camera_backend,
            )
        return self._camera

    @property
    def processor(self) -> DocumentProcessor:
        if (
            self._processor is None
            or getattr(self._processor, "scan_mode", None) != self.scan_mode
        ):
            self._processor = DocumentProcessor(scan_mode=self.scan_mode)
        return self._processor

    @property
    def quality_gate(self) -> QualityGate:
        if self._quality_gate is None:
            self._quality_gate = QualityGate()
        return self._quality_gate

    @property
    def pdf_builder(self) -> PDFBuilder:
        if self._pdf_builder is None:
            # Build the PDF from the *raw* camera frames so the PDF embeds the
            # whole image the user actually saw. The cropped/scanned files are
            # still kept alongside for reference.
            self._pdf_builder = PDFBuilder(pages_dir=self.raw_dir, output_dir=self.pdf_dir)
        return self._pdf_builder

    @property
    def qr_generator(self) -> QRGenerator:
        if self._qr_generator is None:
            self._qr_generator = QRGenerator(output_dir=self.qr_dir, host=self.web_host, port=self.web_port)
        return self._qr_generator

    @property
    def flask_server(self) -> FlaskServer:
        if self._flask_server is None:
            self._flask_server = FlaskServer(self, host=self.web_host, port=self.web_port)
        return self._flask_server

    # ------------------------------------------------------------------ #
    # Public FSM API
    # ------------------------------------------------------------------ #
    def request_quit(self) -> None:
        """Signal the main loop to exit (called by Flask /quit)."""
        self.quit_requested = True

    def on_finish(self, callback: Callable[[Path], None]) -> None:
        """Register a callback fired after a successful PDF build."""
        self._on_finish_callbacks.append(callback)

    def page_count(self) -> int:
        return len(self.pages)

    def page_filename(self, index: Optional[int] = None) -> Path:
        """Return the path the *next* (or specific) page should be written to."""
        idx = (index if index is not None else self.page_count() + 1)
        return self.scanned_dir / f"{PAGE_PREFIX}{idx}.jpg"

    def raw_filename(self, index: Optional[int] = None) -> Path:
        """Return the path the *next* raw (unprocessed) frame should be written to.

        Raw frames are written regardless of the quality-gate outcome so the
        user can always inspect what the camera actually saw.
        """
        idx = (index if index is not None else self.page_count() + 1)
        return self.raw_dir / f"{RAW_PREFIX}{idx}.jpg"

    # ------------------------------------------------------------------ #
    # Capture pipeline
    # ------------------------------------------------------------------ #
    def capture_current_frame(self, frame: Optional[np.ndarray] = None) -> tuple:
        """Run the LIVE pipeline on ``frame`` (or read one from the camera).

        Returns ``(ok, message, processed_bgr, detection)``.
        """
        if frame is None:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                return False, "camera read failed", None, None  # type: ignore[return-value]

        # Always archive the raw frame first — it is the ground truth of what
        # the camera saw and is invaluable for tuning the quality gate.
        raw_path = self.raw_filename()
        try:
            cv2.imwrite(str(raw_path), frame)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not write raw frame to %s: %s", raw_path, exc)

        processed, detection = self.processor.process(frame)
        report = self.quality_gate.evaluate(processed, detection, raw_frame_for_motion=frame)
        self.last_quality = report
        if not report.ok:
            # Tag the raw file with the rejection reason so it's easy to triage.
            try:
                reason_path = raw_path.with_suffix(".reason.txt")
                reason_path.write_text(
                    f"rejected: {report.reason}\n"
                    f"blur={report.blur:.2f}\n"
                    f"brightness={report.brightness:.2f}\n"
                    f"motion={report.motion:.2f}\n"
                    f"corner_confidence={report.corner_confidence:.3f}\n",
                    encoding="utf-8",
                )
            except OSError:  # pragma: no cover - defensive
                pass
            return False, f"rejected: {report.reason}", processed, detection

        # Save the page to disk (the canonical PDF source) and keep an in-mem copy.
        path = self.page_filename()
        cv2.imwrite(str(path), processed)
        self.pages.append(processed)
        return True, f"page {self.page_count()} captured", processed, detection

    # ------------------------------------------------------------------ #
    def finish_pdf(self) -> Optional[Path]:
        """Flush ``self.pages`` into a numbered PDF and a matching QR PNG."""
        if not self.pages:
            self.last_message = "no pages to save"
            return None

        doc_id = self.doc_counter
        filename = document_filename(doc_id)
        # Persist any in-memory pages that aren't already on disk.
        for i, page in enumerate(self.pages, start=1):
            target = self.page_filename(index=i)
            if not target.exists():
                cv2.imwrite(str(target), page)

        pdf_path = self.pdf_builder.build_from_paths(self.list_pages(), filename)
        if pdf_path is None:
            self.last_message = "PDF build failed"
            return None

        self.last_pdf_path = pdf_path
        try:
            self.last_qr_path = self.qr_generator.make_for_document(doc_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("QR generation failed: %s", exc)
            self.last_qr_path = None

        # Start the Flask server exactly once, on the first D.
        try:
            self.flask_server.ensure_running()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Flask server failed to start: %s", exc)

        for cb in list(self._on_finish_callbacks):
            try:
                cb(pdf_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("on_finish callback raised: %s", exc)

        self.last_message = f"saved {pdf_path.name} ({self.page_count()} pages)"
        logger.info(self.last_message)

        # Reset for the next document.
        self.pages = []
        self.state = ScannerState.PDF_VIEW_MODE
        return pdf_path

    # ------------------------------------------------------------------ #
    def start_new_document(self) -> None:
        """Close the PDF_VIEW dialog and start a fresh document."""
        self.doc_counter += 1
        self.pages = []
        self.last_pdf_path = None
        self.last_qr_path = None
        self.last_message = f"new document {self.doc_counter}"
        # Wipe the in-session captures folder so page_NNN.jpg restarts at 1.
        for f in self.scanned_dir.glob(f"{PAGE_PREFIX}*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass
        self.state = ScannerState.LIVE_SCANNER_MODE

    # ------------------------------------------------------------------ #
    def list_pages(self) -> List[Path]:
        if not self.scanned_dir.exists():
            return []
        pages = [
            p for p in self.scanned_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and p.stem.startswith(PAGE_PREFIX)
        ]
        pages.sort(key=lambda p: int(p.stem[len(PAGE_PREFIX):]))
        return pages

    # ------------------------------------------------------------------ #
    # Key dispatch
    # ------------------------------------------------------------------ #
    def handle_key(self, key: int) -> None:
        """Dispatch an OpenCV waitKey code to the FSM."""
        ch = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 else ""

        # Modal Exit? Y/N takes priority over normal keys.
        if self.show_exit_modal:
            if ch in ("y", "n"):
                self.show_exit_modal = False
                if ch == "y":
                    self.quit_requested = True
                    # Auto-save on Q if pages exist.
                    if self.page_count() > 0 and self.last_pdf_path is None:
                        self.finish_pdf()
                    return
                # "n" - just close the modal, stay in the current state.
                self.last_message = "exit cancelled"
                return
            # ignore other keys while modal is up
            return

        if self.state == ScannerState.LIVE_SCANNER_MODE:
            self._handle_live_key(ch)
        else:
            self._handle_pdf_view_key(ch)

    # ------------------------------------------------------------------ #
    def _handle_live_key(self, ch: str) -> None:
        if ch == "c":
            ok, msg, _proc, _det = self.capture_current_frame()
            self.last_message = msg
            return
        if ch == "d":
            if self.page_count() == 0:
                self.last_message = "press C first - nothing to save"
                return
            saved = self.finish_pdf()
            if saved is not None:
                self.last_message = f"finished -> {saved.name}"
            return
        if ch == "n":
            # N is only valid in PDF_VIEW.  Stay in LIVE and tell the user.
            self.last_message = "press D first to finish this document"
            return
        if ch == "q":
            self.show_exit_modal = True
            self.last_message = (
                "save & quit?" if self.page_count() > 0 else "quit? (no pages captured)"
            )
            return
        if ch in ("m", "M"):
            order = ("color", "grayscale", "bw")
            current = self.scan_mode if self.scan_mode in order else order[0]
            self.scan_mode = order[(order.index(current) + 1) % len(order)]
            self._processor = None  # force rebuild on next frame
            self.last_message = f"scan mode -> {self.scan_mode}"
            return

    # ------------------------------------------------------------------ #
    def _handle_pdf_view_key(self, ch: str) -> None:
        if ch == "n":
            self.start_new_document()
            return
        if ch == "q":
            self.show_exit_modal = True
            self.last_message = "quit?"
            return
        if ch in ("c", "d"):
            self.last_message = "press N for a new document"
            return

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Compose the OpenCV window canvas for the current state."""
        if self.state == ScannerState.LIVE_SCANNER_MODE:
            return self._render_live(frame)
        return self._render_pdf_view()

    # ------------------------------------------------------------------ #
    def _render_live(self, frame: Optional[np.ndarray]) -> np.ndarray:
        if frame is None:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        # Run the processor so the overlay reflects current detection state.
        try:
            (
                processed,
                detection,
                gray,
                edges,
                contour_overlay,
                biggest_overlay,
                warped_bgr,
                warped_gray_bgr,
                adaptive_bgr,
            ) = self.processor.process_with_debug(frame)
        except Exception:
            return self._render_live_fallback(frame)

        # Build the 8-panel debug grid.  Stages that are unavailable
        # (no quad found => no warp) are filled with black tiles so the
        # layout stays stable and the user can see what's missing.
        blank = np.zeros_like(frame)
        panels: List[List[Optional[np.ndarray]]] = [
            [frame, _to_bgr(gray), _to_bgr(edges), contour_overlay],
            [
                biggest_overlay,
                warped_bgr if warped_bgr is not None else blank,
                warped_gray_bgr if warped_gray_bgr is not None else blank,
                adaptive_bgr if adaptive_bgr is not None else blank,
            ],
        ]
        try:
            canvas = stack_images(panels, DEBUG_GRID_SCALE, DEBUG_GRID_LABELS)
        except Exception:
            return self._render_live_fallback(frame)

        # HUD (drawn on top of the stacked grid - the canvas is wider/taller
        # than the raw frame so coordinates are relative to the grid).
        _draw_text(
            canvas,
            f"state: LIVE  pages: {self.page_count()}  mode: {self.scan_mode}  conf {detection.confidence:.2f}",
            (10, 30),
            color=(0, 255, 0),
            bg=(0, 0, 0),
        )
        _draw_text(
            canvas,
            "[C] capture   [D] finish PDF   [M] cycle mode   [N] (after D)   [Q] quit",
            (10, 60),
            color=(255, 255, 255),
            bg=(0, 0, 0),
        )
        if self.last_message:
            _draw_text(canvas, self.last_message, (10, 90),
                       color=(0, 255, 255), bg=(0, 0, 0))

        # Live quality readout - always visible.  Turns orange when the
        # gate is currently rejecting (so the user knows why C did nothing),
        # green when the next C will succeed.
        readout_y = grid_label_bottom(panels, DEBUG_GRID_SCALE) + 20
        if self.last_quality is not None:
            q = self.last_quality
            color = (0, 255, 0) if q.ok else (0, 140, 255)
            _draw_text(
                canvas,
                f"quality: {q.reason or 'ok'} "
                f"(blur={q.blur:.0f} bright={q.brightness:.0f} "
                f"motion={q.motion:.1f}px)",
                (10, readout_y),
                color=color,
                bg=(0, 0, 0),
            )
        else:
            _draw_text(canvas, "quality: -- (press C to sample)", (10, readout_y),
                       color=(180, 180, 180), bg=(0, 0, 0))

        # Thumb of the final processed page in the bottom-right of the grid,
        # so the user can see what would be saved on C.
        thumb = _make_thumb(processed, size=160)
        if thumb is not None and canvas.shape[1] > thumb.shape[1] + 20:
            tx = canvas.shape[1] - thumb.shape[1] - 20
            ty = canvas.shape[0] - thumb.shape[0] - 20
            canvas[ty : ty + thumb.shape[0], tx : tx + thumb.shape[1]] = thumb
            cv2.rectangle(
                canvas,
                (tx - 2, ty - 2),
                (tx + thumb.shape[1] + 2, ty + thumb.shape[0] + 2),
                (255, 255, 255),
                1,
            )
            _draw_text(canvas, "would save", (tx, ty - 8),
                       color=(255, 255, 255), bg=(0, 0, 0), scale=0.5)

        if self.show_exit_modal:
            self._draw_exit_modal(canvas)

        return canvas

    # ------------------------------------------------------------------ #
    def _render_live_fallback(self, frame: np.ndarray) -> np.ndarray:
        """Single-tile fallback when ``process_with_debug`` or stacking fails."""
        try:
            processed, detection = self.processor.process(frame)
            canvas = DocumentProcessor.draw_overlay(frame, detection, processed_preview=processed)
        except Exception:
            canvas = frame.copy()
            _draw_text(canvas, "pipeline error - showing raw frame",
                       (10, 30), color=(0, 0, 255), bg=(0, 0, 0))

        _draw_text(canvas, f"state: LIVE  pages: {self.page_count()}  mode: {self.scan_mode}",
                   (10, 60), color=(0, 255, 0), bg=(0, 0, 0))
        _draw_text(canvas,
                   "[C] capture   [D] finish PDF   [M] cycle mode   [N] (after D)   [Q] quit",
                   (10, 90), color=(255, 255, 255), bg=(0, 0, 0))
        if self.last_message:
            _draw_text(canvas, self.last_message, (10, 120),
                       color=(0, 255, 255), bg=(0, 0, 0))
        if self.show_exit_modal:
            self._draw_exit_modal(canvas)
        return canvas

    # ------------------------------------------------------------------ #
    def _render_pdf_view(self) -> np.ndarray:
        # NOTE: the camera frame is intentionally NOT drawn here - per spec.
        h, w = self.camera_height, self.camera_width
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (32, 32, 32)

        _draw_text(canvas, "state: PDF_VIEW", (10, 30),
                   color=(0, 255, 255), bg=(0, 0, 0))
        _draw_text(canvas, "Document saved. Scan the QR or open the URL below.",
                   (10, 60), color=(255, 255, 255), bg=(0, 0, 0))

        y = 110
        if self.last_pdf_path is not None:
            n_pages = max(self.page_count_for_pdf(self.last_pdf_path), 0)
            _draw_text(canvas, f"PDF: {self.last_pdf_path.name}  ({n_pages} pages)",
                       (10, y), color=(255, 255, 255), bg=(0, 0, 0))
            y += 30

            host = self.flask_server.host or self.web_host
            port = self.flask_server.port or self.web_port
            url = f"http://{host}:{port}/{self.last_pdf_path.name}"
            _draw_text(canvas, f"URL: {url}", (10, y), color=(180, 255, 180), bg=(0, 0, 0))
            y += 30

        if self.last_qr_path is not None and self.last_qr_path.exists():
            qr_img = cv2.imread(str(self.last_qr_path), cv2.IMREAD_COLOR)
            if qr_img is not None:
                size = 220
                qr_img = cv2.resize(qr_img, (size, size), interpolation=cv2.INTER_AREA)
                x0 = w - size - 20
                y0 = h - size - 60
                canvas[y0:y0+size, x0:x0+size] = qr_img
                cv2.rectangle(canvas, (x0-2, y0-2), (x0+size+2, y0+size+2), (255, 255, 255), 2)

        _draw_text(canvas, "[N] new document   [Q] quit", (10, h - 30),
                   color=(255, 255, 255), bg=(0, 0, 0))

        if self.last_message:
            _draw_text(canvas, self.last_message, (10, h - 60),
                       color=(0, 255, 255), bg=(0, 0, 0))

        if self.show_exit_modal:
            self._draw_exit_modal(canvas)

        return canvas

    # ------------------------------------------------------------------ #
    def _draw_exit_modal(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        box_w, box_h = 460, 200
        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), thickness=-1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 2)

        title = "Exit Application?"
        if self.page_count() > 0 and self.last_pdf_path is None:
            title = "Save current document & Exit?"

        _draw_text(canvas, title, (x0 + 20, y0 + 50), color=(0, 255, 255), scale=0.8)
        _draw_text(canvas, "Press Y to confirm", (x0 + 20, y0 + 90), color=(255, 255, 255))
        _draw_text(canvas, "Press N to cancel",  (x0 + 20, y0 + 120), color=(255, 255, 255))

    # ------------------------------------------------------------------ #
    @staticmethod
    def page_count_for_pdf(pdf_path: Path) -> int:
        """Best-effort page count for a PDF written by PIL.

        ``PIL`` doesn't expose page counts directly, so we count ``/Type /Page``
        tokens which works for the simple PDFs we generate.
        """
        try:
            data = pdf_path.read_bytes()
        except OSError:
            return 0
        # crude but sufficient for PIL-produced PDFs
        return max(1, data.count(b"/Type /Page") - data.count(b"/Type /Pages"))

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Open the camera and pump the FSM until quit."""
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        try:
            while not self.quit_requested:
                canvas = self.render()
                cv2.imshow(WINDOW_TITLE, canvas)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC also quits (no modal)
                    if self.page_count() > 0 and self.last_pdf_path is None:
                        self.finish_pdf()
                    break
                if key != 255:
                    self.handle_key(key)
        finally:
            self.shutdown()

    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Release the camera and close the window."""
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:  # pragma: no cover - defensive
                pass
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# Backwards-compatible aliases (legacy tests still import these names).
# --------------------------------------------------------------------------- #
def _build_pdf_saved_canvas(session: ScanSession) -> np.ndarray:
    """Used by ``tests/run_session_lifecycle.py`` - returns the PDF_VIEW canvas."""
    session.state = ScannerState.PDF_VIEW_MODE
    return session._render_pdf_view()


# Re-export the legacy scanner shim so older tests don't break.
from scanner import DocumentScanner  # noqa: E402  (intentional at module bottom)


# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    """Parse CLI flags.  ``--source`` accepts an int index OR a URL string."""
    p = argparse.ArgumentParser(
        prog="document-scanner",
        description="Smart Document Scanner - live camera + PDF/QR pipeline.",
    )
    p.add_argument(
        "--source",
        default="0",
        help=(
            "Camera source. Either an integer index (e.g. 0, 1) for a "
            "local webcam, or a URL for a network stream such as "
            "http://192.168.1.107:4747/video (DroidCam / IP Webcam)."
        ),
    )
    p.add_argument("--backend", default="opencv", choices=["opencv", "picamera2"])
    p.add_argument("--scan-mode", default=SCAN_MODE,
                   choices=["color", "grayscale", "bw"],
                   help="Output style for captured pages (default: %(default)s)")
    p.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    p.add_argument("--no-quality-gate", action="store_true",
                   help="Bypass the QualityGate (useful for tuning the detector).")
    p.add_argument("--blur-min", type=float, default=None,
                   help=("Minimum Laplacian variance to accept a frame. "
                         "Lower this for soft phone-camera streams (default ~80). "
                         "Try 20-40 if 'rejected: blurry' appears too often."))
    p.add_argument("--motion-max", type=float, default=None,
                   help=("Maximum mean pixel-difference between consecutive "
                         "frames. Raise this (e.g. 40-60) if MJPEG shimmer "
                         "trips 'rejected: motion' even when the scene is still."))
    p.add_argument("--brightness-min", type=float, default=None,
                   help="Reject frames with mean grayscale below this.")
    p.add_argument("--brightness-max", type=float, default=None,
                   help="Reject frames with mean grayscale above this.")
    p.add_argument("--host", default=DEFAULT_WEB_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_WEB_PORT)
    p.add_argument("--web", dest="web", action="store_true", default=True,
                   help="Start the Flask gallery (default: on).")
    p.add_argument("--no-web", dest="web", action="store_false",
                   help="Do not start the Flask gallery.")
    return p.parse_args(argv)


def _coerce_source(value: str) -> object:
    """Turn ``"0"`` into ``0`` (index) but keep ``"http://..."`` as a string."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    """Console entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    args = _parse_args(argv)
    session = ScanSession(
        camera_source=_coerce_source(args.source),
        camera_backend=args.backend,
        camera_width=args.width,
        camera_height=args.height,
        web_host=args.host,
        web_port=args.port,
        scan_mode=args.scan_mode,
    )

    if args.no_quality_gate:
        from quality_gate import QualityGate
        session._quality_gate = QualityGate(enabled=False)
        logger.info("QualityGate disabled via --no-quality-gate")
    else:
        # Allow per-axis overrides even when the gate stays enabled.
        overrides = {
            "blur_min": args.blur_min,
            "motion_max": args.motion_max,
            "brightness_min": args.brightness_min,
            "brightness_max": args.brightness_max,
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if overrides:
            from quality_gate import QualityGate
            gate = QualityGate(**overrides)
            session._quality_gate = gate
            logger.info("QualityGate overrides: %s", overrides)

    logger.info(
        "Starting scanner: source=%r backend=%s %dx%d web=%s host=%s port=%d",
        args.source, args.backend, args.width, args.height,
        args.web, args.host, args.port,
    )

    if not args.web:
        # Disable the auto-start in finish_pdf() by short-circuiting the property.
        session._flask_server = type("_NoFlask", (), {"ensure_running": lambda self: None})()

    try:
        session.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user - shutting down.")
    finally:
        session.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))