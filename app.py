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
    ABSOLUTE_BLUR_MIN_VARIANCE,
    DEFAULT_AUTO_CAPTURE_COOLDOWN,
    DEFAULT_AUTO_CAPTURE_ENABLED,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_STABLE_FRAMES,
    DEFAULT_STABILITY_TOLERANCE,
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
from auto_capture_controller import AutoCaptureController, S1_SEEKING_STABLE, S2_WAITING_FOR_CHANGE
from document_processor import DetectionResult, DocumentProcessor
from flask_server import FlaskServer
from sound import SoundPlayer
try:
    from config import DEFAULT_SOUND_ENABLED, DEFAULT_SOUND_VOLUME
except ImportError:  # pragma: no cover - config is required at runtime
    DEFAULT_SOUND_ENABLED = True
    DEFAULT_SOUND_VOLUME = 0.6
from voice import VoicePrompter
try:
    from config import (
        DEFAULT_VOICE_BACKEND,
        DEFAULT_VOICE_ENABLED,
        DEFAULT_VOICE_LANGUAGE,
        DEFAULT_VOICE_RATE_WPM,
    )
except ImportError:  # pragma: no cover - config is required at runtime
    DEFAULT_VOICE_ENABLED = True
    DEFAULT_VOICE_LANGUAGE = "en"
    DEFAULT_VOICE_RATE_WPM = 165
    DEFAULT_VOICE_BACKEND = "auto"
from image_grid import stack_images
from page_change_detector import PageChangeDetector, PageChangeEvent
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
    _page_change_detector: Optional[PageChangeDetector] = field(default=None, init=False)
    _on_finish_callbacks: List[Callable[[Path], None]] = field(
        default_factory=list, init=False
    )

    # Page-change tuning knobs (CLI-overridable).  These mirror the
    # constants in config.py but are duplicated on the dataclass so
    # tests can override them per-instance without monkey-patching
    # module-level globals.
    page_change_enabled: bool = True
    auto_page_change_bump: bool = True
    page_change_hash_distance: int = 10
    page_change_motion_trigger_px: float = 25.0
    page_change_motion_rest_px: float = 6.0
    page_change_rest_frames: int = 6
    page_change_quad_jump_px: float = 35.0
    last_page_change: Optional[PageChangeEvent] = field(default=None, init=False)

    # Auto-capture tuning knobs (CLI-overridable).  When
    # ``auto_capture_enabled`` is True the LIVE loop will capture the
    # current frame as soon as a document quad has been stable for
    # ``auto_capture_stable_frames`` consecutive frames, then wait
    # ``auto_capture_cooldown_s`` seconds before re-arming.
    auto_capture_enabled: bool = DEFAULT_AUTO_CAPTURE_ENABLED
    auto_capture_cooldown_s: float = DEFAULT_AUTO_CAPTURE_COOLDOWN
    auto_capture_stable_frames: int = DEFAULT_STABLE_FRAMES
    auto_capture_tolerance_px: float = DEFAULT_STABILITY_TOLERANCE

    # Audio cues.  ``sound_enabled`` is the master switch (mirrored on
    # ``--sound`` / ``--no-sound``).  ``sound_volume`` is a 0..1 gain
    # applied to every WAV blob at construction time.  ``_sound`` is
    # constructed lazily so tests can monkey-patch it via
    # ``session._sound = stub``.
    sound_enabled: bool = DEFAULT_SOUND_ENABLED
    sound_volume: float = DEFAULT_SOUND_VOLUME
    _sound: Optional[SoundPlayer] = field(default=None, init=False)
    _sound_detect_start_played: bool = field(default=False, init=False)

    # Voice prompts (spoken cues layered on top of the tones).  Same
    # lazy / monkey-patchable contract as ``_sound`` above: tests can
    # drop in a stub via ``session._voice = ...`` and reconfigure via
    # ``session.voice_enabled = False``.
    voice_enabled: bool = DEFAULT_VOICE_ENABLED
    voice_language: str = DEFAULT_VOICE_LANGUAGE
    voice_rate_wpm: int = DEFAULT_VOICE_RATE_WPM
    voice_backend: str = DEFAULT_VOICE_BACKEND
    _voice: Optional[VoicePrompter] = field(default=None, init=False)
    # Internal book-keeping for the HUD pill.
    _auto_capture: Optional[AutoCaptureController] = field(default=None, init=False)
    _auto_capture_phase: str = field(default="off", init=False)
    _auto_capture_progress: tuple = field(default=(0, 0), init=False)

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

    @property
    def page_change_detector(self) -> PageChangeDetector:
        if self._page_change_detector is None:
            self._page_change_detector = PageChangeDetector(
                enabled=self.page_change_enabled,
                motion_trigger_px=self.page_change_motion_trigger_px,
                motion_rest_px=self.page_change_motion_rest_px,
                rest_frames=self.page_change_rest_frames,
                quad_jump_px=self.page_change_quad_jump_px,
                hash_distance=self.page_change_hash_distance,
                auto_bump=self.auto_page_change_bump,
            )
        return self._page_change_detector

    @property
    def auto_capture(self) -> AutoCaptureController:
        """Lazy accessor so tests can monkey-patch ``_auto_capture``."""
        if self._auto_capture is None:
            self._auto_capture = AutoCaptureController(
                enabled=self.auto_capture_enabled,
                s2_no_match_timeout_s=self.auto_capture_cooldown_s,
            )
            # Tighten the inner StabilityTracker thresholds to match the
            # ScanSession knobs (which may be overridden per-instance).
            self._auto_capture.tracker.required_frames = self.auto_capture_stable_frames
            self._auto_capture.tracker.tolerance = self.auto_capture_tolerance_px
        return self._auto_capture

    @property
    def sound(self) -> SoundPlayer:
        """Lazy accessor for the audio cue player.

        Constructed on first access so tests can replace ``_sound``
        without triggering ``__post_init__``.
        """
        if self._sound is None:
            self._sound = SoundPlayer(
                enabled=self.sound_enabled,
                volume=self.sound_volume,
            )
        return self._sound

    def play_sound(self, event: str) -> None:
        """Safe wrapper around :meth:`SoundPlayer.play_event`.

        Never raises.  All exceptions are logged at DEBUG level so a
        failing sound backend can never break the LIVE loop or the
        quality gate.
        """
        if not self.sound_enabled:
            return
        try:
            self.sound.play_event(event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("play_sound(%r) failed: %s", event, exc)

    @property
    def voice(self) -> VoicePrompter:
        """Lazy accessor for the spoken-voice prompter.

        Constructed on first access so tests can replace ``_voice``
        without triggering ``__post_init__`` (mirrors :attr:`sound`).
        """
        if self._voice is None:
            self._voice = VoicePrompter(
                enabled=self.voice_enabled,
                language=self.voice_language,
                rate_wpm=self.voice_rate_wpm,
                backend=self.voice_backend,
                sound_player=self.sound,
            )
        return self._voice

    def speak(self, event: str, **fmt) -> None:
        """Safe wrapper around :meth:`VoicePrompter.speak`.

        Never raises.  All exceptions are logged at DEBUG level so a
        missing TTS backend or a failing subprocess can never break the
        LIVE loop, the quality gate, or any capture path.
        """
        if not self.voice_enabled:
            return
        try:
            self.voice.speak(event, **fmt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("speak(%r) failed: %s", event, exc)

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

        # Hard gate: never persist a frame without detected corners. The
        # previous behavior was to fall back to a center crop, which fed
        # garbage (and visually convincing but blank) pages into the PDF.
        detection_corners = getattr(detection, "corners", None)
        if detection_corners is None:
            self.last_quality = None
            # Still observe the page-change detector with the *raw* quad
            # signal (None) so a real quad later on can establish a baseline.
            try:
                self.page_change_detector.observe(
                    quad=None,
                    processed_bgr=processed,
                    motion_px=0.0,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("page-change observe (no-doc) failed: %s", exc)
            return False, "no document detected - hold page in frame", processed, detection

        report = self.quality_gate.evaluate(processed, detection, raw_frame_for_motion=frame)
        self.last_quality = report

        # Hard sharpness floor that ALWAYS fires, even when the user's
        # session has the soft quality gate disabled.  Catches the
        # "waxy 6.0-variance upscaled" failure mode that was producing
        # blurred pages in the PDF without any reject signal.
        try:
            import cv2 as _cv2  # local import keeps top-of-file untouched
            gray = _cv2.cvtColor(processed, _cv2.COLOR_BGR2GRAY)
            lap_var = float(_cv2.Laplacian(gray, _cv2.CV_64F).var())
        except Exception:
            lap_var = float(report.blur) if report else 0.0
        if lap_var < ABSOLUTE_BLUR_MIN_VARIANCE:
            self.last_quality = report
            try:
                reason_path = raw_path.with_suffix(".reason.txt")
                reason_path.write_text(
                    f"rejected: blur (abs floor) variance={lap_var:.1f}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
            return False, f"rejected: blurry ({lap_var:.0f}<{ABSOLUTE_BLUR_MIN_VARIANCE:.0f})", processed, detection

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
            # Still feed the page-change observer so we don't get stuck
            # in IDLE forever just because the very first frame was rejected.
            try:
                self.page_change_detector.observe(
                    quad=getattr(detection, "corners", None),
                    processed_bgr=processed,
                    motion_px=report.motion,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("page-change observe (rejected frame) failed: %s", exc)
            return False, f"rejected: {report.reason}", processed, detection

        # ------------------------------------------------------------------
        # Page-change detection -- runs ONLY on accepted frames so that the
        # detector's baseline always reflects a real, scannable page.
        # ------------------------------------------------------------------
        page_change_event: Optional[PageChangeEvent] = None
        try:
            page_change_event = self.page_change_detector.observe(
                quad=getattr(detection, "corners", None),
                processed_bgr=processed,
                motion_px=report.motion,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("page-change observe failed: %s", exc)

        if page_change_event is not None:
            self.last_page_change = page_change_event
            swapped = self._on_page_change(page_change_event, processed)
            if swapped is not None:
                # _on_page_change already grabbed the new page via recursion
                # (or failed to and we want to fall through to the normal
                # capture path below as a safety net).
                if swapped:
                    return True, f"auto-bumped to page {self.page_count()}", processed, detection
                # swapped == False means the user disabled auto-bump;
                # capture THIS frame as page N so we keep momentum.

        # Save the page to disk (the canonical PDF source) and keep an in-mem copy.
        path = self.page_filename()
        cv2.imwrite(str(path), processed)
        self.pages.append(processed)

        # Refresh the page-change baseline so the NEXT frame is compared
        # against this just-captured page (not a stale one from minutes ago).
        try:
            self.page_change_detector.update_baseline_after_capture(
                processed, getattr(detection, "corners", None)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("page-change baseline update failed: %s", exc)

        return True, f"page {self.page_count()} captured", processed, detection

    # ------------------------------------------------------------------ #
    def _on_page_change(
        self,
        event: PageChangeEvent,
        processed_bgr: np.ndarray,
    ) -> Optional[bool]:
        """React to a confirmed page swap.

        * If ``auto_page_change_bump`` is True  -> wipe the in-session
          captures, drop in-memory pages, drop a breadcrumb under
          ``raw/auto_change_<ts>.reason.txt``, and recursively capture the
          current frame so the new page is appended.
        * If it's False -> just set ``last_message`` so the LIVE HUD can
          prompt the user to press ``C``.

        Returns ``True`` if a swap was performed, ``False`` if the user
        disabled auto-bump, ``None`` if the recursive capture raised
        (caller will fall through and capture the original frame).
        """
        if not self.auto_page_change_bump:
            self.last_message = (
                f"NEW PAGE detected (conf={event.confidence:.2f}) -- press C to capture"
            )
            logger.info("page-change detected but auto-bump disabled")
            return False

        try:
            # Wipe scanned page_NNN.jpg so page indexing restarts cleanly.
            for f in self.scanned_dir.glob(f"{PAGE_PREFIX}*.jpg"):
                try:
                    f.unlink()
                except OSError:
                    pass
            self.pages = []

            # Drop a breadcrumb so an offline reviewer can tell why the
            # page counter restarted.
            breadcrumb = self.raw_dir / f"auto_change_{int(time.time())}.reason.txt"
            breadcrumb.parent.mkdir(parents=True, exist_ok=True)
            breadcrumb.write_text(
                f"auto page change\n"
                f"confidence={event.confidence:.3f}\n"
                f"hash_distance={event.hash_distance}\n"
                f"quad_distance={event.quad_distance:.2f}\n"
                f"motion_at_peak={event.motion_at_peak:.2f}\n",
                encoding="utf-8",
            )

            self.last_message = (
                f"NEW PAGE (conf={event.confidence:.2f}) -- capturing..."
            )
            logger.info(
                "page-change auto-bump  conf=%.2f hash=%d quad=%.1fpx",
                event.confidence, event.hash_distance, event.quad_distance,
            )
            # The page list was just reset above; announce the page
            # number we're about to capture (1-indexed) instead of the
            # detector confidence which was the previous, confusing
            # value here.
            self.speak("page_change", n=self.page_count() + 1)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("auto-bump cleanup failed: %s", exc)
            self.speak("error", detail=str(exc))

        # Now grab the new page so the user sees it land in the PDF list.
        try:
            ok, msg, _proc, _det = self.capture_current_frame()
            if not ok:
                self.last_message = f"new page detected but capture failed: {msg}"
            return ok
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("recursive capture during auto-bump failed: %s", exc)
            self.speak("error", detail=str(exc))
            return None

    # ------------------------------------------------------------------ #
    def _maybe_auto_capture(self) -> None:
        """Tick the two-state auto-capture FSM on each LIVE render frame.

        States (mirrored by ``AutoCaptureController.state``):

        * **State 1 -- ``S1_SEEKING_STABLE``**  ("S1_seeking" on the HUD)
          Watch the incoming frames; when the same quad has been visible
          for ``auto_capture_stable_frames`` consecutive stable ticks,
          fire a capture and drop into State 2.

        * **State 2 -- ``S2_WAITING_FOR_CHANGE``**
          Stay armed but refuse to fire until 2 continuous seconds have
          passed during which the live quad is *not* similar to the
          four-point contour that was saved at the moment of the last
          capture.  A tick is "not similar" when any of these holds:

              * the live quad is ``None`` (page left the frame);
              * the per-frame motion spiked above
                ``page_change_motion_trigger_px``;
              * the quad drifted beyond ``auto_capture_tolerance_px``
                from ``last_captured_quad``;
              * the dedicated ``PageChangeDetector`` already saw a move
                (this triggers an instant flip).

          The first continuous no-match streak of ``s2_no_match_timeout_s``
          seconds (default 2.0 s) flips the FSM back to State 1, ready
          for the next auto-capture.  A manual ``c`` key-press from
          State 2 captures again, overwrites ``last_captured_quad`` with
          the freshly saved contour, and resets the no-match timer back
          to zero -- the FSM stays in State 2.

          There is no fixed cooldown: a similar quad on any tick resets
          the no-match timer to 0, and the next non-match tick starts
          the count over.
        """
        if not self.auto_capture_enabled:
            self._auto_capture_phase = "off"
            self._auto_capture_progress = (0, 0)
            return

        # Pull a fresh frame and run detection.
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self._auto_capture_phase = "S1_seeking"
            self._auto_capture_progress = (0, 0)
            self.last_message = "AUTO | waiting for camera"
            return

        try:
            processed, detection = self.processor.process(frame)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("auto-capture processor failed: %s", exc)
            self._auto_capture_phase = "S1_seeking"
            return

        quad = getattr(detection, "corners", None)

        # Compute a cheap per-frame motion estimate for the FSM.  We
        # diff the *processed* page against the in-memory copy of the
        # last successful capture (when available) so it's both fast and
        # representative of what the user would actually compare.
        motion_px = self._quick_motion(processed)

        # Probe the page-change detector's FSM WITHOUT consuming an
        # event - the real ``observe(...)`` still runs inside
        # ``capture_current_frame`` and may auto-bump the page counter.
        try:
            page_change_event = self.page_change_detector.peek_for_change(
                quad=quad, motion_px=motion_px
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("page-change peek failed: %s", exc)
            page_change_event = False

        # Hand everything to the new two-state controller.  Always go
        # through the `auto_capture` *property* so the controller is
        # lazily constructed on the first call.
        self.auto_capture.motion_trigger_px = self.page_change_motion_trigger_px
        result = self.auto_capture.observe(
            quad,
            motion_px=motion_px,
            page_change_event=page_change_event or None,
        )

        # Mirror FSM state onto the HUD-facing fields the rest of the
        # UI / webserver / tests already read.
        self._auto_capture_phase = result.phase
        self._auto_capture_progress = (
            int(result.progress[0]), int(result.progress[1])
        )

        if result.phase == "off":
            return

        # Document left the frame: re-arm the per-session "detect_start"
        # chime flag so the next visible quad triggers a fresh blip.
        # This was the legacy behaviour pre-FSM and is asserted by
        # smoke_sound.p7_fsm_sound_hooks ("doc-disappeared re-arms the
        # detect_start chime").
        if quad is None:
            self._sound_detect_start_played = False

        # *** Fire path *** -- controller says "go" while still in S1.
        # Execute the capture BEFORE we touch the phase/progress mirrors
        # below so the queued capture uses the live detection context.
        if result.should_fire:
            self._fire_auto_capture(frame, result)
            # _fire_auto_capture already updated _auto_capture_phase /
            # _auto_capture_progress on success or pushed us back to
            # S1 on rejection - nothing more to mirror here.
            return

        # Mirror FSM state onto the HUD-facing fields the rest of the
        # UI / webserver / tests already read.
        self._auto_capture_phase = result.phase
        self._auto_capture_progress = (
            int(result.progress[0]), int(result.progress[1])
        )

        # State 1 -- building streak.  Play "detected" once per session.
        if result.phase == "S1_seeking":
            if quad is not None and not self._sound_detect_start_played:
                self._sound_detect_start_played = True
                self.play_sound("detect_start")
                self.speak("detected")
            self.last_message = (
                f"AUTO | S1 seeking {result.progress[0]}/{result.progress[1]}"
            )
            return

        # State 2 -- post-fire.  The controller now returns two phase
        # labels from State 2:
        #   * "S2_match"   -- the live quad currently looks similar to
        #                     the contour that was saved at capture
        #                     time.  No-match timer stays at 0.
        #   * "S2_waiting" -- live quad differs from the saved contour
        #                     this tick; the 2 s no-match timer is
        #                     accumulating.  The HUD shows the countdown
        #                     so the user knows when State 1 returns.
        if result.phase == "S2_match":
            self.last_message = (
                f"AUTO | captured p{self.page_count()} | "
                f"same page (timer held)"
            )
            return

        if result.phase == "S2_waiting":
            remain = self.auto_capture.no_match_remaining
            tail = "change detected" if result.change_detected else "waiting for change"
            self.last_message = (
                f"AUTO | captured p{self.page_count()} | "
                f"{tail} {remain:.1f}s"
            )
            return

        # Unknown phase label - leave the mirrors as the controller
        # already set them and bail.
        return

    # ------------------------------------------------------------------ #
    def _fire_auto_capture(
        self, frame: np.ndarray, result: "ObserveResult"
    ) -> None:
        """Execute the capture path once ``AutoCaptureController.observe`` fires."""
        try:
            ok_cap, msg_cap, _proc, _det = self.capture_current_frame(frame)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("auto-capture firing failed: %s", exc)
            ok_cap, msg_cap = False, str(exc)

        if ok_cap:
            self._auto_capture_phase = self.auto_capture.phase_label()
            self._auto_capture_progress = (
                self.auto_capture.tracker.required_frames,
                self.auto_capture.tracker.required_frames,
            )
            self.last_message = (
                f"AUTO | captured page {self.page_count()} | "
                f"state 2"
            )
            logger.info(
                "auto-captured page %d (no-match timeout %.1fs)",
                self.page_count(), self.auto_capture.s2_no_match_timeout_s,
            )
            # Audio cues -- the user just heard state-1 say "ready";
            # now play the capture click + verbal confirmation.
            self.play_sound("detect_stable")
            self.play_sound("capture")
            self.speak("stable")
            self.speak("capture_auto")
        else:
            # Capture was rejected by the quality gate.  Push the FSM
            # back to State 1 so the next stable window gets another
            # chance without spamming messages.
            self.auto_capture.state = S1_SEEKING_STABLE
            self.auto_capture.tracker.reset()
            self._auto_capture_phase = "S1_seeking"
            self._auto_capture_progress = (
                self.auto_capture.tracker.stable_count,
                self.auto_capture.tracker.required_frames,
            )
            self.last_message = f"AUTO | {msg_cap}"
            self.speak("capture_rejected", reason=msg_cap)

    # ------------------------------------------------------------------ #
    def _quick_motion(self, processed: np.ndarray) -> float:
        """Cheap per-frame MAD against the last accepted page.

        Returns 0.0 on the first frame of a session.  Used by the
        auto-capture FSM to detect a "frame moved" signal in State 2
        without paying for a full ``QualityGate.evaluate(...)`` call.
        """
        prev = getattr(self, "_last_accepted_processed", None)
        if prev is None or prev.shape != processed.shape:
            self._last_accepted_processed = processed
            return 0.0
        try:
            import cv2 as _cv2  # local import keeps module import-time slim
            import numpy as _np
            diff = _cv2.absdiff(
                _cv2.cvtColor(prev, _cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev,
                _cv2.cvtColor(processed, _cv2.COLOR_BGR2GRAY) if processed.ndim == 3 else processed,
            )
            self._last_accepted_processed = processed
            return float(_np.mean(diff))
        except Exception:  # pragma: no cover - defensive
            return 0.0

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
        # Re-arm the "doc detected" cue so the next page gets its own blip.
        self._sound_detect_start_played = False
        try:
            self.page_change_detector.reset()
        except Exception:  # pragma: no cover - defensive
            pass
        # Drop the auto-capture lock so the next session starts fresh.
        if self._auto_capture is not None:
            self._auto_capture.tracker.reset()
            self._auto_capture.last_capture_timestamp = 0.0
            self._auto_capture.state = S1_SEEKING_STABLE
        self._auto_capture_phase = "S1_seeking"
        self._auto_capture_progress = (0, 0)
        return pdf_path

    # ------------------------------------------------------------------ #
    def start_new_document(self) -> None:
        """Close the PDF_VIEW dialog and start a fresh document."""
        self.doc_counter += 1
        self.pages = []
        self.last_pdf_path = None
        self.last_qr_path = None
        self.last_message = f"new document {self.doc_counter}"
        self.speak("document_new")
        # Re-arm the "doc detected" cue so the next page gets its own blip.
        self._sound_detect_start_played = False
        # Wipe the in-session captures folder so page_NNN.jpg restarts at 1.
        for f in self.scanned_dir.glob(f"{PAGE_PREFIX}*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            self.page_change_detector.reset()
        except Exception:  # pragma: no cover - defensive
            pass
        self.last_page_change = None
        # Reset auto-capture so the new document starts in S1.
        if self._auto_capture is not None:
            self._auto_capture.tracker.reset()
            self._auto_capture.last_capture_timestamp = 0.0
            self._auto_capture.state = S1_SEEKING_STABLE
        self._auto_capture_phase = "S1_seeking"
        self._auto_capture_progress = (0, 0)
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
            # Manual capture -- always succeeds as a capture, and the
            # FSM moves into State 2.  In State 2 the controller will
            # only flip back to State 1 after a continuous 2 s of "no
            # similar contour".  ``register_capture`` records the new
            # four-point contour (or clears the baseline if corners
            # are unknown) and resets the no-match timer.
            if ok:
                controller = self.auto_capture
                controller.register_capture(
                    _det.corners if _det is not None else None
                )
                self._auto_capture_phase = controller.phase_label()
                self._auto_capture_progress = (
                    controller.tracker.required_frames,
                    controller.tracker.required_frames,
                )
                # Audio cue - same "ka-chunk" as an auto-capture fire.
                self.play_sound("capture")
                self.speak("capture_manual")
            else:
                self.speak("capture_rejected", reason=msg)
            return
        if ch == "d":
            if self.page_count() == 0:
                self.last_message = "press C first - nothing to save"
                return
            saved = self.finish_pdf()
            if saved is not None:
                self.last_message = f"finished -> {saved.name}"
                # Audible save cue: a short rising chime paired with the
                # spoken "Document saved, N pages" so the user always
                # hears confirmation even if the TTS is interrupted by
                # the previous capture ka-chunk.
                self.play_sound("detect_stable")
                self.speak("document_saved", n=self.page_count())
            else:
                self.speak("capture_rejected", reason=msg)
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
            self.speak("document_new")
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
            _draw_text(
                canvas,
                self.last_message,
                (10, 90),
                color=(0, 0, 0) if self.auto_capture_enabled else (255, 255, 255),
                bg=(0, 0, 0) if not self.auto_capture_enabled else None,
            )
        # ------------------------------------------------------------------
        # AUTO pill - only when the feature is on.
        # ------------------------------------------------------------------
        if self.auto_capture_enabled:
            phase = self._auto_capture_phase
            if phase == "cooldown":
                pill_text = (
                    f"AUTO | captured p{self.page_count()} | "
                    f"cooldown {self.auto_capture_cooldown_s:.1f}s"
                )
                pill_color = (255, 255, 255)
                pill_bg = (0, 170, 90)
            elif phase == "identifying":
                c, r = self._auto_capture_progress
                pill_text = f"AUTO | identifying {c}/{r}"
                pill_color = (255, 255, 255)
                pill_bg = (0, 140, 255)
            elif phase == "idle":
                pill_text = "AUTO | waiting for document"
                pill_color = (255, 255, 255)
                pill_bg = (90, 90, 90)
            else:
                pill_text = ""
                pill_bg = None
            if pill_text:
                _draw_text(canvas, pill_text, (10, 118), color=pill_color, bg=pill_bg)
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
                # Tick the auto-capture state machine BEFORE rendering so
                # the HUD pill shows the result of this frame's check.
                if (
                    self.auto_capture_enabled
                    and self.state == ScannerState.LIVE_SCANNER_MODE
                    and not self.show_exit_modal
                ):
                    self._maybe_auto_capture()

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
        # Release any temp WAVs written for winsound playback.
        try:
            if self._sound is not None:
                self._sound.close()
        except Exception:  # pragma: no cover - defensive
            pass


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
    p.add_argument("--auto-capture", dest="auto_capture", action="store_true",
                   default=None,
                   help="Force auto-capture ON (default: on).")
    p.add_argument("--no-auto-capture", dest="auto_capture", action="store_false",
                   default=None,
                   help="Disable auto-capture (default: enabled).")
    p.add_argument("--auto-capture-cooldown", type=float, default=None,
                   help=("Seconds the FSM stays in 'cooldown' after each "
                         "auto- or manual-capture before re-arming "
                         "(default: 5.0)."))
    p.add_argument("--auto-capture-stable", type=int, default=None,
                   help=("Number of consecutive frames that must agree on "
                         "the same quad before an auto-capture fires "
                         "(default: 60, i.e. ~2.0 s at the 30 ms LIVE tick)."))
    p.add_argument("--auto-capture-tolerance", type=float, default=None,
                   help=("Maximum pixel drift between consecutive quads "
                         "before the stability counter resets (default: 18)."))
    p.add_argument("--sound", dest="sound", action="store_true",
                   default=None,
                   help="Enable audio cues (default: enabled).")
    p.add_argument("--no-sound", dest="sound", action="store_false",
                   default=None,
                   help="Disable all audio cues (--sound / --no-sound).")
    p.add_argument("--sound-volume", type=float, default=None,
                   help=("Linear gain 0.0 (silent) - 1.0 (full) for the audio "
                         "cues (default: 0.6)."))
    p.add_argument("--voice", dest="voice", action="store_true",
                   default=None,
                   help="Enable spoken-voice prompts (default: enabled).")
    p.add_argument("--no-voice", dest="voice", action="store_false",
                   default=None,
                   help="Disable spoken-voice prompts (--voice / --no-voice).")
    p.add_argument("--voice-language", type=str, default=None,
                   help=("espeak-ng voice id (default: en).  Examples: en, "
                         "en-us, de, fr.  Ignored on Windows by most SAPI5 voices."))
    p.add_argument("--voice-rate", type=int, default=None,
                   help=("Speaking rate in words per minute (default: 165)."))
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

    # Auto-capture CLI overrides -------------------------------------------------
    if getattr(args, "auto_capture", None) is True:
        session.auto_capture_enabled = True
    elif getattr(args, "auto_capture", None) is False:
        session.auto_capture_enabled = False
    if getattr(args, "auto_capture_cooldown", None) is not None:
        session.auto_capture_cooldown_s = float(args.auto_capture_cooldown)
    if getattr(args, "auto_capture_stable", None) is not None:
        session.auto_capture_stable_frames = int(args.auto_capture_stable)
    if getattr(args, "auto_capture_tolerance", None) is not None:
        session.auto_capture_tolerance_px = float(args.auto_capture_tolerance)
    logger.info(
        "AutoCapture: enabled=%s cooldown=%.2fs stable_frames=%d tolerance=%.1fpx",
        session.auto_capture_enabled,
        session.auto_capture_cooldown_s,
        session.auto_capture_stable_frames,
        session.auto_capture_tolerance_px,
    )

    # Sound CLI overrides ---------------------------------------------------------
    if getattr(args, "sound", None) is True:
        session.sound_enabled = True
    elif getattr(args, "sound", None) is False:
        session.sound_enabled = False
    if getattr(args, "sound_volume", None) is not None:
        session.sound_volume = float(args.sound_volume)
        # Force re-build of WAV cache at the new volume.
        session._sound = None
    logger.info(
        "Sound: enabled=%s volume=%.2f",
        session.sound_enabled, session.sound_volume,
    )

    # Voice CLI overrides --------------------------------------------------------
    if getattr(args, "voice", None) is True:
        session.voice_enabled = True
    elif getattr(args, "voice", None) is False:
        session.voice_enabled = False
    if getattr(args, "voice_language", None) is not None:
        session.voice_language = str(args.voice_language)
    if getattr(args, "voice_rate", None) is not None:
        session.voice_rate_wpm = int(args.voice_rate)
    logger.info(
        "Voice: enabled=%s language=%s rate=%d wpm",
        session.voice_enabled, session.voice_language, session.voice_rate_wpm,
    )

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
        session.speak("shutdown")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
