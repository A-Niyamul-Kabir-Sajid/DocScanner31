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
        X - delete the last captured page (in-memory + on-disk),
            renumber the remaining pages so the next PDF stays
            contiguous.  No-op when there are no captured pages.
        N - rejected (must D first)
        Q - if any pages captured: auto D, then modal Exit? Y/N
            else: modal Exit? Y/N

    PDF_VIEW_MODE
        N - reset pages, bump doc counter, return to LIVE_SCANNER_MODE
        Q - modal Exit? Y/N
        C/D - ignored

The LIVE canvas also renders a thumbnail of the most recently captured
page (bottom-left) so the user can see what would currently be removed
by pressing ``X``.

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
    CAMERA_RETRY_SECONDS,
    DEFAULT_AUTO_CAPTURE_COOLDOWN,
    DEFAULT_AUTO_CAPTURE_ENABLED,
    DEFAULT_AUTOFOCUS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_LENS_POSITION_DIOPTRES,
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

# ---------------------------------------------------------------------------
# Focus tuning (manual focus hotkeys + one-shot AF)
# ---------------------------------------------------------------------------
# libcamera's IMX519 driver reports a valid LensPosition in [0.0, ~10.0]
# dioptres (1/metres).  Values >= 10.0 either saturate or error out.  2.2
# dpt == ~45 cm, the default desk-to-doc distance.  We keep a hard ceiling
# at 10.0 so a runaway hotkey can't take the lens out of range.
FOCUS_LENS_MIN_DPT = 0.5     # ~2.0 m  (far wall)
FOCUS_LENS_MAX_DPT = 10.0    # ~0.1 m  (close-up)
FOCUS_LENS_STEP_DPT = 0.1    # +/- per [, ] keypress
FOCUS_FINE_STEP_DPT = 0.05   # shift-[ / shift-] (smaller nudge)
# One-shot AF timeout for the [F] hotkey -- matches Camera.trigger_autofocus.
FOCUS_LOCK_TIMEOUT_S = 2.0
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
from mp3_player import MP3Player
try:
    from config import (
        DEFAULT_MP3_CAPTURED_FILE,
        DEFAULT_MP3_DELETED_FILE,
        DEFAULT_MP3_DEVICE,
        DEFAULT_MP3_ENABLED,
        DEFAULT_MP3_VOLUME_DB,
    )
except ImportError:  # pragma: no cover - config is required at runtime
    DEFAULT_MP3_ENABLED = True
    DEFAULT_MP3_DEVICE = "plughw:2,0"
    DEFAULT_MP3_VOLUME_DB = 8.0
    DEFAULT_MP3_CAPTURED_FILE = "captured.mp3"
    DEFAULT_MP3_DELETED_FILE = "deleted.mp3"
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


def _probe_camera_url(source: object, *, timeout_s: float = 1.5) -> Optional[str]:
    """Best-effort reachability check for an OpenCV source.

    ``source`` is either an int (local webcam index) or a string URL such
    as ``"http://192.168.43.1:8080/video"``.  For URLs we open a TCP
    socket to ``host:port``; if the connect succeeds (or fails with
    something other than "host unreachable / timeout") we return ``None``
    so the caller's expensive ``cv2.VideoCapture(...)`` retry path is the
    one that decides.  When the host is plainly unreachable we return a
    short, human-friendly error string so it can be shown on the offline
    canvas *and* logged once at startup.

    Returns ``None`` for local indices (they have no host to probe) and
    for sources we don't recognise.
    """
    if not isinstance(source, str) or not source.lower().startswith(("http://", "https://", "rtsp://")):
        return None
    try:
        from urllib.parse import urlparse
        u = urlparse(source)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
    except Exception as exc:  # pragma: no cover - defensive
        return f"could not parse source URL ({exc})"
    if not host:
        return "source URL has no host"
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout_s)
            s.connect((host, int(port)))
        return None
    except socket.timeout:
        return (f"timed out talking to {host}:{port} -- is DroidCam on the "
                f"same Wi-Fi as this PC?")
    except OSError as exc:
        # ``ConnectionRefusedError`` -> port closed on that host
        # ``socket.gaierror`` -> DNS / wrong subnet
        # ``[Errno 10051]`` -> network unreachable on Windows
        return (f"{exc} -- check the URL and that both devices share the "
                f"same Wi-Fi (this PC sees {host} but nothing is "
                f"listening).")


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
# UI primitives
UI_BG = (20, 25, 45)
UI_PANEL = (30, 36, 64)
UI_PANEL_ALT = (40, 48, 90)
UI_BORDER = (58, 70, 110)
UI_TEXT = (230, 236, 255)
UI_MUTED = (138, 147, 184)
UI_ACCENT = (79, 140, 255)
UI_SUCCESS = (43, 212, 156)
UI_WARN = (255, 181, 71)
UI_ERR = (255, 93, 108)
UI_KEY_BG = (40, 48, 80)
UI_HOTKEY_BG = (28, 32, 56)
UI_STATUS_BAR_H = 44
UI_HOTKEY_BAR_H = 40


def _ui_panel(canvas, x, y, w, h, *, fill=UI_PANEL, border=UI_BORDER, border_thickness=1):
    ch, cw = canvas.shape[:2]
    x0 = max(0, int(x)); y0 = max(0, int(y))
    x1 = min(cw, int(x + w)); y1 = min(ch, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), fill, thickness=-1)
    if border is not None and border_thickness > 0:
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), border, thickness=border_thickness)


def _ui_chip(canvas, text, x, y, *, fg=UI_TEXT, bg=UI_KEY_BG, scale=0.5, thickness=1, pad=8):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    w = tw + pad * 2
    h = th + baseline + pad * 2
    ch, cw = canvas.shape[:2]
    x0 = max(0, int(x)); y0 = max(0, int(y))
    x1 = min(cw, int(x + w)); y1 = min(ch, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), bg, thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), UI_BORDER, thickness=1)
    ty = y0 + pad + th
    cv2.putText(canvas, text, (x0 + pad, ty), font, scale, fg, thickness, cv2.LINE_AA)


def _ui_key_chip(canvas, key, label, x, y, *, accent=UI_ACCENT, scale=0.45):
    key_text = f"[{key}]"
    _ui_chip(canvas, key_text, x, y, fg=UI_TEXT, bg=accent, scale=scale, pad=6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    key_w = cv2.getTextSize(key_text, font, scale, 1)[0][0] + 12
    text_x = x + key_w + 8
    _draw_text(canvas, label, (text_x, y + 14), color=UI_MUTED, bg=None, scale=scale)
    label_w = cv2.getTextSize(label, font, scale, 1)[0][0]
    return text_x + label_w + 18


def _ui_status_bar(canvas, *, title, subtitle="", pills=None):
    h, w = canvas.shape[:2]
    bar_h = UI_STATUS_BAR_H
    cv2.rectangle(canvas, (0, 0), (w, bar_h), UI_PANEL, thickness=-1)
    cv2.rectangle(canvas, (0, 0), (w, 3), UI_ACCENT, thickness=-1)
    cv2.line(canvas, (0, bar_h - 1), (w, bar_h - 1), UI_BORDER, thickness=1)
    _draw_text(canvas, title, (16, 28), color=UI_TEXT, bg=None, scale=0.7, thickness=2)
    if subtitle:
        _draw_text(canvas, subtitle, (16, bar_h - 6),
                   color=UI_MUTED, bg=None, scale=0.4, thickness=1)
    if pills:
        cx = w - 16
        for text, fg, bg in reversed(list(pills)):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            chip_w = tw + 16
            cx -= chip_w
            _ui_chip(canvas, text, cx, 10, fg=fg, bg=bg, scale=0.5, pad=8)
            cx -= 8


def _ui_hotkey_bar(canvas, keys, *, message=None, y=None):
    """Draw the bottom hotkey rail.

    Parameters
    ----------
    canvas : np.ndarray
        Target BGR canvas.
    keys : list[tuple[str, str, tuple]]
        ``(key, label, accent_bgr)`` chips.
    message : str, optional
        Soft message drawn just above the rail (e.g. "saved as scan_1.pdf").
    y : int, optional
        Top edge of the rail.  Defaults to the very bottom of the canvas.
    """
    h, w = canvas.shape[:2]
    bar_h = UI_HOTKEY_BAR_H
    top = h - bar_h if y is None else int(y)
    cv2.rectangle(canvas, (0, top), (w, top + bar_h), UI_HOTKEY_BG, thickness=-1)
    cv2.line(canvas, (0, top), (w, top), UI_BORDER, thickness=1)
    if message:
        _draw_text(canvas, message, (16, top - 8),
                   color=UI_WARN, bg=None, scale=0.45, thickness=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    widths = []
    total = 0
    for key, label, _accent in keys:
        key_text = f"[{key}]"
        kw = cv2.getTextSize(key_text, font, scale, 1)[0][0] + 12
        lw = cv2.getTextSize(label, font, scale, 1)[0][0]
        chip_total = kw + 8 + lw + 18
        widths.append(chip_total)
        total += chip_total
    x = max(16, (w - total) // 2)
    y = top + 10
    for (key, label, accent), w_each in zip(keys, widths):
        _ui_key_chip(canvas, key, label, x, y,
                     accent=accent or UI_ACCENT, scale=scale)
        x += w_each


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
    # ``"auto"`` (default) - the Camera wrapper picks picamera2 on a Pi and
    # opencv on a desktop / a network URL.  Use ``--backend opencv`` or
    # ``--backend picamera2`` on the CLI to override per session.
    camera_backend: str = "auto"
    camera_width: int = DEFAULT_CAMERA_WIDTH
    camera_height: int = DEFAULT_CAMERA_HEIGHT
    # Focus knobs.  ``None`` means "use the config default"; an explicit
    # ``True``/``False`` from the CLI overrides.  Stored on the dataclass so
    # tests can mutate it without monkey-patching module-level globals.
    camera_autofocus: Optional[bool] = None
    camera_lens_position: Optional[float] = None
    # Pi-camera-only: when True, ``capture_current_frame`` triggers a
    # single-shot autofocus on each capture (libcamera ``AfMode = Auto``,
    # poll ``AfStatus`` until locked) so desk-distance scans come out
    # sharp without running the lens motor continuously.  Off by default
    # because it adds ~1.5s of latency per shot on the IMX519 AF cycle
    # and most pages sit at the same distance -- enable with
    # ``--autofocus-on-capture`` when the doc-to-lens distance varies.
    autofocus_on_capture: bool = False
    # Rotate captured frames before they leave ``Camera.read()``.  Use this
    # when the camera module is physically mounted in portrait but the
    # sensor still delivers landscape frames, or when you simply want the
    # preview/output oriented as A4 portrait.  Allowed values:
    # ``0`` (no rotation), ``90``, ``180``, ``270``.  Applied symmetrically
    # on both OpenCV and picamera2 backends so the downstream detector /
    # scanner code never has to know which way the sensor was facing.
    camera_rotate: int = 0
    # Pi-camera-only: when True (default) the Camera wrapper disables
    # libcamera's auto-crop so the LIVE preview shows the same field of
    # view as ``rpicam-hello --width W --height H`` instead of being
    # cropped (digitally zoomed) to the requested main-stream aspect
    # ratio.  Ignored on the OpenCV backend.  Use --no-full-fov on the
    # CLI to recover the legacy picamera2 crop behaviour.
    camera_full_fov: bool = True
    web_host: str = DEFAULT_WEB_HOST
    web_port: int = DEFAULT_WEB_PORT
    scan_mode: str = SCAN_MODE
    # Display mode for the OpenCV window.  ``False`` (default) opens a
    # resizable window of (camera_width x camera_height); pass
    # ``--fullscreen`` to fill the whole monitor instead.
    window_fullscreen: bool = False

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
    # ``time.monotonic()`` deadline for the next camera-reopen attempt.
    # 0.0 means "no retry scheduled" (camera is online, or we haven't yet
    # observed the first failure). The LIVE loop ticks this forward by
    # ``CAMERA_RETRY_SECONDS`` while the camera is offline so the device
    # is probed at a steady cadence rather than every frame.
    _next_camera_retry_at: float = field(default=0.0, init=False)
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

    # Long-form MP3 cues (Raspberry Pi 5 + MAX98357A I2S amp).  Same
    # lazy / monkey-patchable contract as ``_sound`` / ``_voice`` above.
    mp3_enabled: bool = DEFAULT_MP3_ENABLED
    mp3_device: str = DEFAULT_MP3_DEVICE
    mp3_volume_db: float = DEFAULT_MP3_VOLUME_DB
    mp3_captured_file: str = DEFAULT_MP3_CAPTURED_FILE
    mp3_deleted_file: str = DEFAULT_MP3_DELETED_FILE
    _mp3: Optional[MP3Player] = field(default=None, init=False)
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
        """Lazy, crash-safe accessor for the underlying ``Camera``.

        The first call attempts to open the configured source.  If the
        device is missing we keep the partially-initialised handle (with
        ``is_open=False``) so the rest of the app can render a "camera not
        found" overlay and poll ``try_reopen()`` from the LIVE loop.
        """
        if self._camera is None:
            try:
                self._camera = Camera(
                    source=self.camera_source,
                    width=self.camera_width,
                    height=self.camera_height,
                    backend=self.camera_backend,
                    autofocus=self.camera_autofocus,
                    lens_position=self.camera_lens_position,
                    rotate=self.camera_rotate,
                    full_fov=self.camera_full_fov,
                    autofocus_on_capture=self.autofocus_on_capture,
                )
                # The Camera constructor resolves "auto" to a concrete
                # backend.  Reflect the resolved value back onto the session
                # so ``camera_status()`` always reports the truth.
                self.camera_backend = self._camera.backend
            except Exception as exc:  # pragma: no cover - defensive belt
                # Camera.__init__ no longer raises, but guard the legacy
                # picamera2 path that still does on missing libs.
                logger.warning("Camera init failed: %s", exc)
                self._camera = Camera.__new__(Camera)
                self._camera.source = self.camera_source
                self._camera.width = self.camera_width
                self._camera.height = self.camera_height
                self._camera.backend = self.camera_backend
                self._camera.rotate = int(self.camera_rotate or 0)
                self._camera._cap = None
                self._camera._pi_cam = None
                self._camera.is_open = False
                self._camera.last_open_error = str(exc)
                self._camera.autofocus = (
                    DEFAULT_AUTOFOCUS if self.camera_autofocus is None
                    else bool(self.camera_autofocus)
                )
                self._camera.lens_position = (
                    DEFAULT_LENS_POSITION_DIOPTRES if self.camera_lens_position is None
                    else float(self.camera_lens_position)
                )
        return self._camera

    # ------------------------------------------------------------------ #
    # Camera health / retry helpers
    # ------------------------------------------------------------------ #
    def _ensure_camera_alive(self) -> None:
        """Reconnect to the camera if it's currently offline.

        Called from the LIVE tick. We only fire a real reopen every
        ``CAMERA_RETRY_SECONDS`` so the network/source isn't hammered on
        every frame. The countdown is driven by ``time.monotonic`` so the
        interval stays correct across OpenCV's ``waitKey`` jitter.
        """
        cam = self._camera
        if cam is None:
            # Force the lazy property to materialise (covers picamera2).
            _ = self.camera
            cam = self._camera
        if cam is None or cam.is_open:
            self._next_camera_retry_at = 0.0
            return

        now = time.monotonic()
        if self._next_camera_retry_at == 0.0:
            self._next_camera_retry_at = now + CAMERA_RETRY_SECONDS
        if now < self._next_camera_retry_at:
            return

        logger.info("Retrying camera open (source=%r)…", self.camera_source)
        ok = cam.try_reopen()
        if ok:
            logger.info("Camera reconnected.")
            self._next_camera_retry_at = 0.0
            self.last_message = "Camera reconnected"
        else:
            self._next_camera_retry_at = now + CAMERA_RETRY_SECONDS
            logger.debug("Camera still offline: %s", cam.last_open_error)

    def camera_status(self) -> dict:
        """Snapshot used by the LIVE overlay to render the offline banner."""
        cam = self._camera
        online = bool(cam and cam.is_open)
        retry_in = 0.0
        if not online and self._next_camera_retry_at > 0.0:
            retry_in = max(0.0, self._next_camera_retry_at - time.monotonic())
        autofocus = bool(getattr(cam, "autofocus", DEFAULT_AUTOFOCUS)) if cam else DEFAULT_AUTOFOCUS
        lens_position = (
            float(getattr(cam, "lens_position", DEFAULT_LENS_POSITION_DIOPTRES))
            if cam else DEFAULT_LENS_POSITION_DIOPTRES
        )
        return {
            "online": online,
            "source": self.camera_source,
            "backend": self.camera_backend,
            "autofocus": autofocus,
            "lens_position": lens_position,
            "error": (cam.last_open_error if cam else "camera not initialised"),
            "retry_in": retry_in,
        }

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
            logger.debug("play_sound(%r): skipped (session.sound_enabled=False)", event)
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

    @property
    def mp3(self) -> MP3Player:
        """Lazy accessor for the long-form MP3 cue player.

        Constructed on first access so tests can replace ``_mp3`` without
        triggering ``__post_init__`` (mirrors :attr:`sound` / :attr:`voice`).
        On non-Linux hosts the underlying backend is a silent no-op, so this
        property is safe to access from any platform.
        """
        if self._mp3 is None:
            self._mp3 = MP3Player(
                enabled=self.mp3_enabled,
                captured_file=self.mp3_captured_file,
                deleted_file=self.mp3_deleted_file,
                device=self.mp3_device,
                volume_db=self.mp3_volume_db,
            )
        return self._mp3

    def play_mp3(self, event: str) -> None:
        """Safe wrapper around :meth:`MP3Player.play_event`.

        Recognised events match the project-root clip filenames:

        * ``"captured"``     - plays ``captured.mp3``.
        * ``"page_deleted"`` - plays ``deleted.mp3``.

        Never raises.  All exceptions (missing pydub, missing ffmpeg,
        refused ALSA device, file-not-found) are logged at DEBUG level
        so the LIVE loop is never stalled or crashed by audio.
        """
        if not self.mp3_enabled:
            logger.debug("play_mp3(%r): skipped (session.mp3_enabled=False)", event)
            return
        try:
            self.mp3.play_event(event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("play_mp3(%r) failed: %s", event, exc)

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
            # On picamera2 we get a single-shot AF lock right before the
            # snapshot when ``autofocus_on_capture`` is enabled.  This
            # block is cheap when the flag is off (returns False in <1ms
            # on the OpenCV backend and on cameras that already
            # continuous-AF).
            try:
                if self.autofocus_on_capture and getattr(
                    self.camera, "autofocus_on_capture", False
                ):
                    af_locked = self.camera.trigger_autofocus(timeout_s=2.0)
                    logger.debug(
                        "autofocus_on_capture: lock=%s", af_locked,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("trigger_autofocus raised (ignored): %s", exc)
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

        # Document left the frame: re-arm the per-session capture
        # chime flag so the next visible quad can trigger a fresh
        # sound when a page is actually captured.  Kept as a flag
        # (legacy hook for smoke_sound) but no longer plays audio.
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

        # State 1 -- building streak.  No audio cue here anymore;
        # the only sound is played when a page is actually captured
        # (see ``_fire_auto_capture`` and the manual-capture path).
        if result.phase == "S1_seeking":
            if quad is not None and not self._sound_detect_start_played:
                self._sound_detect_start_played = True
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
            # Audio cues -- play the single capture chime plus the
            # verbal confirmation.
            self.play_sound("captured")
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
    def delete_last_page(self) -> bool:
        """Drop the most recently captured page from the in-session buffer.

        Pops ``self.pages[-1]`` and deletes the matching ``page_NNN.jpg``
        (plus its raw sibling ``raw_NNN.jpg`` and any ``.reason.txt``
        breadcrumb) from disk.  Remaining pages are renumbered on disk so
        the next ``D`` press still produces a contiguous PDF.

        Returns ``True`` when a page was removed, ``False`` when there
        was nothing to delete.
        """
        if not self.pages:
            self.last_message = "no pages to delete"
            return False

        removed_index = len(self.pages)              # 1-indexed slot being dropped
        removed_image = self.pages.pop()
        del removed_image  # free the numpy buffer eagerly

        # Capture the rest as a snapshot (so we can renumber without
        # touching the in-memory order) and then rewrite the on-disk
        # files to match the new 1..N indexing.  Use the *contents* of
        # self.pages so memory is the source of truth.
        snapshot = list(self.pages)

        # 1) Delete the file that used to be at ``removed_index``.
        for prefix, directory in (
            (PAGE_PREFIX, self.scanned_dir),
            (RAW_PREFIX, self.raw_dir),
        ):
            stale = directory / f"{prefix}{removed_index}.jpg"
            if stale.exists():
                try:
                    stale.unlink()
                except OSError as exc:  # pragma: no cover - defensive
                    logger.debug("could not delete %s: %s", stale, exc)
            reason = stale.with_suffix(".reason.txt")
            if reason.exists():
                try:
                    reason.unlink()
                except OSError:
                    pass

        # 2) Shift everything after the removed slot down by one to
        #    keep the numbering contiguous.  We do this by reading each
        #    stale file and writing it to its new path; this also
        #    drops the in-memory numpy copies that were staler than
        #    the snapshots the user kept.
        for new_idx in range(removed_index, len(snapshot) + 1):
            old_path_scanned = self.scanned_dir / f"{PAGE_PREFIX}{new_idx + 1}.jpg"
            new_path_scanned = self.scanned_dir / f"{PAGE_PREFIX}{new_idx}.jpg"
            if old_path_scanned.exists():
                try:
                    old_path_scanned.rename(new_path_scanned)
                except OSError as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "rename %s -> %s failed: %s",
                        old_path_scanned, new_path_scanned, exc,
                    )

            old_path_raw = self.raw_dir / f"{RAW_PREFIX}{new_idx + 1}.jpg"
            new_path_raw = self.raw_dir / f"{RAW_PREFIX}{new_idx}.jpg"
            if old_path_raw.exists():
                try:
                    old_path_raw.rename(new_path_raw)
                except OSError as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "rename %s -> %s failed: %s",
                        old_path_raw, new_path_raw, exc,
                    )

            # Move the matching rejection-reason breadcrumb too.
            old_reason = (self.raw_dir / f"{RAW_PREFIX}{new_idx + 1}.jpg").with_suffix(
                ".reason.txt"
            )
            new_reason = (self.raw_dir / f"{RAW_PREFIX}{new_idx}.jpg").with_suffix(
                ".reason.txt"
            )
            if old_reason.exists():
                try:
                    old_reason.rename(new_reason)
                except OSError:
                    pass

        # 3) Re-seed the auto-capture baseline against the new "last
        #    page" so the page-change detector doesn't immediately
        #    declare a swap just because we deleted a frame.
        if self.pages:
            new_last = self.pages[-1]
            corners = None
            try:
                # Best-effort: ask the processor to reprocess the last
                # page so the page-change detector has a fresh quad.
                _processed, det = self.processor.process(new_last)
                corners = getattr(det, "corners", None)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("reprocess last page for baseline failed: %s", exc)
            try:
                self.page_change_detector.update_baseline_after_capture(
                    new_last, corners
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("baseline reset after delete failed: %s", exc)
        else:
            try:
                self.page_change_detector.reset()
            except Exception:  # pragma: no cover - defensive
                pass

        self.last_message = (
            f"deleted last page (now {self.page_count()} page(s))"
        )
        logger.info("deleted last page; %d remain", self.page_count())
        return True

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
                self.play_sound("captured")
                # Pi 5 deployment: also play the user-supplied
                # ``captured.mp3`` clip on the MAX98357A I2S amp.
                self.play_mp3("captured")
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
                # No extra sound here -- the user already heard
                # "captured" on every page; the spoken
                # "Document saved, N pages" is the only confirmation.
                self.speak("document_saved", n=self.page_count())
            else:
                self.speak("capture_rejected", reason=msg)
            return
        if ch == "n":
            # N is only valid in PDF_VIEW.  Stay in LIVE and tell the user.
            self.last_message = "press D first to finish this document"
            return
        if ch == "x":
            if self.delete_last_page():
                # Soft "undo" cue so the user hears confirmation even if
                # they're not looking at the HUD.
                self.play_sound("page_deleted")
                # Pi 5 deployment: also play the user-supplied
                # ``deleted.mp3`` clip on the MAX98357A I2S amp.
                self.play_mp3("page_deleted")
                self.speak("page_deleted")
            else:
                self.last_message = "no pages to delete"
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

        # -------------------------------------------------------------- #
        # Focus controls (manual focus / one-shot AF)
        # -------------------------------------------------------------- #
        # ``[`` / ``]`` nudge the manual lens position by ±FOCUS_LENS_STEP_DPT
        # (0.1 dpt).  Holding shift halves the step for fine adjustment.
        # We do not require AF to be off -- switching to manual mode on
        # nudge is the whole point of having a "lock focus here" key.
        if ch in ("[", "]", "{", "}"):
            shift = ch in ("{", "}")
            step = FOCUS_FINE_STEP_DPT if shift else FOCUS_LENS_STEP_DPT
            direction = 1.0 if ch in ("]", "}") else -1.0
            self.nudge_focus(direction * step)
            return

        # ``F`` triggers a single-shot autofocus cycle (libcamera AfMode=Auto)
        # and re-applies the user's preferred mode afterwards.  Works in
        # both manual and continuous-AF modes.
        if ch in ("f", "F"):
            self.trigger_focus_lock()
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
    # Focus helpers (used by the [, ] / F hotkeys)
    # ------------------------------------------------------------------ #
    def nudge_focus(self, delta_dpt: float) -> None:
        """Bump ``self.camera.lens_position`` by ``delta_dpt`` dioptres and
        immediately re-apply it as a libcamera control.

        Switching the camera to ``AfMode=Manual`` and setting ``LensPosition``
        is the only way to actually move the lens on the IMX519 stack from
        Python -- libcamera ignores writes to the property bag if the mode
        is ``Continuous``.  We always force the mode to ``Manual`` on nudge
        so the user gets instant feedback; the original ``self.autofocus``
        preference is preserved on the instance for the HUD / next session.

        The actual ``set_controls`` + verify-and-retry loop is delegated to
        ``Camera.set_manual_focus`` so runtime nudges get the same race
        protection (``_settle_manual_focus``) as the startup path -- the
        IMX519 + AK7375 actuator has a "first frame is a no-op" race that
        bites hotkey nudges just as hard as ``--lens-position``.
        """
        cam = getattr(self, "_camera", None)
        if cam is None or not cam.is_open:
            self.last_message = "camera not open yet"
            return
        if cam.backend != "picamera2":
            self.last_message = "manual focus only on --backend picamera2"
            return

        new_value = float(cam.lens_position) + float(delta_dpt)
        # Clamp into the IMX519's safe range so we never request infinity
        # or a value that the driver rounds back to zero.
        new_value = max(FOCUS_LENS_MIN_DPT, min(FOCUS_LENS_MAX_DPT, new_value))

        # Delegate the actual driver call + verify-and-retry to
        # ``Camera.set_manual_focus`` so hotkeys and ``--lens-position``
        # share the same race-resistant code path.
        applied = cam.set_manual_focus(new_value)
        cam.autofocus = False  # nudge means "lock here, manual"

        # LensPosition is in dioptres (1/m).  Show a friendly cm estimate.
        cm = 100.0 / max(0.1, new_value)
        if applied != applied:  # NaN check without importing math
            self.last_message = (
                f"focus -> {new_value:.2f} dpt  (≈{cm:0.0f} cm)  "
                f"step {delta_dpt:+.2f} -- driver didn't report back"
            )
        elif abs(applied - new_value) > 0.05:
            # Driver reported back a different value than we asked for.
            # On the IMX519 this means the actuator is stale or the
            # module is fixed-focus; either way the lens didn't move to
            # what we wanted.
            self.last_message = (
                f"focus -> {new_value:.2f} dpt (driver reports "
                f"{applied:.2f})  step {delta_dpt:+.2f}"
            )
        else:
            self.last_message = (
                f"focus -> {new_value:.2f} dpt  (≈{cm:0.0f} cm)  "
                f"step {delta_dpt:+.2f}"
            )

    def trigger_focus_lock(self) -> None:
        """Kick a one-shot AF cycle (``AfMode=Auto``) on the picamera2
        backend, then re-apply the user's preferred mode.

        This is the [F] hotkey -- gives you a single auto-focus pass for
        the current scene without leaving continuous-AF on permanently.
        """
        cam = getattr(self, "_camera", None)
        if cam is None or not cam.is_open:
            self.last_message = "camera not open yet"
            return
        if cam.backend != "picamera2":
            self.last_message = "autofocus only on --backend picamera2"
            return
        # ``Camera.trigger_autofocus`` polls AfStatus for up to
        # FOCUS_LOCK_TIMEOUT_S and then restores the user's AF mode.
        self.last_message = "focusing... (one-shot AF)"
        locked = bool(cam.trigger_autofocus(timeout_s=FOCUS_LOCK_TIMEOUT_S))
        # Update the last_message AFTER the call so the user sees a result.
        # We can't read AfStatus from the post-call AfMode, but the
        # trigger_autofocus() logger shows it; the user can press F again
        # if it didn't lock.
        if locked:
            self.last_message = "focus locked (one-shot AF complete)"
        else:
            self.last_message = (
                "focus didn't lock in time -- nudged default; try F again"
            )

    def _focus_status_label(self) -> str:
        """Human-readable focus mode + lens position for the HUD pill.

        We distinguish "what we asked for" (``cam.lens_position``) from
        "what the driver reports in metadata" (``cam.get_applied_lens_position``).
        On a fixed-focus module or a stale IMX519 actuator these two can
        disagree, and showing only the requested value masks the
        "lens didn't actually move" case the user is trying to debug.
        Sampling the metadata on every render is a bit pricey (~30 ms of
        capture_request wall time) but the HUD only redraws at ~30 fps
        anyway, so the cost is hidden.
        """
        cam = getattr(self, "_camera", None)
        if cam is None or not cam.is_open:
            return "focus: —"
        if cam.backend != "picamera2":
            # OpenCV / network streams -- focus is handled by the device.
            return "focus: auto (device)"
        # On picamera2 the truth is the live ``self.autofocus`` flag, which
        # ``nudge_focus`` flips to False so the HUD reflects what the
        # lens is actually doing.
        if cam.autofocus:
            return "focus: continuous AF"
        try:
            requested = float(cam.lens_position)
        except Exception:  # pragma: no cover - defensive
            return "focus: manual"
        applied = cam.get_applied_lens_position()
        cm_req = 100.0 / max(0.1, requested)
        if applied != applied:  # NaN: driver didn't report LensPosition
            return (
                f"focus: manual {requested:.2f} dpt (≈{cm_req:0.0f} cm) "
                f"[no driver feedback]"
            )
        if abs(applied - requested) > 0.05:
            # Driver disagrees with what we asked for -- classic symptom
            # of a fixed-focus module or a wedged actuator.
            cm_app = 100.0 / max(0.1, applied)
            return (
                f"focus: manual {requested:.2f} dpt "
                f"(driver: {applied:.2f}, ≈{cm_app:0.0f} cm)"
            )
        cm = 100.0 / max(0.1, applied)
        return f"focus: manual {applied:.2f} dpt (≈{cm:0.0f} cm)"

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
        # Heartbeat the camera watchdog BEFORE we try to read so the
        # "retry in X.Xs" countdown is always based on the freshest state.
        self._ensure_camera_alive()

        if frame is None:
            ok, frame = self.camera.read()
        else:
            # Caller already supplied a frame (smoke tests, off-screen
            # preview, replay).  Optimistically trust it - the
            # camera_status() branch below will still divert to the
            # offline renderer if the camera is actually broken.
            ok = True

        # If the camera is offline (no handle / read failed / no quad
        # possible on an all-black frame), short-circuit to a dedicated
        # "camera not found" canvas so the app never crashes mid-session.
        status = self.camera_status()
        if (not status["online"]) or (frame is None) or (not ok):
            return self._render_camera_offline(status, frame)

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

        h, w = canvas.shape[:2]

        # ------------------------------------------------------------------
        # Top status bar
        # ------------------------------------------------------------------
        page_count = self.page_count()
        page_label = f"{page_count} page{'s' if page_count != 1 else ''} captured"
        conf_label = f"page detected — {int(round(detection.confidence * 100))}% sure"
        # Friendly subtitle that explains what the "scan_mode" value means.
        mode_phrase = {
            "color": "color pages",
            "grayscale": "black & white pages",
            "bw": "high-contrast pages",
        }.get(self.scan_mode, self.scan_mode)
        _ui_status_bar(
            canvas,
            title="Smart Document Scanner",
            subtitle=f"Saving as {mode_phrase}",
            pills=[
                ("LIVE", UI_TEXT, UI_SUCCESS),
                (page_label, UI_TEXT, UI_ACCENT),
                (f"mode: {self.scan_mode}", UI_TEXT, UI_KEY_BG),
                (conf_label, UI_TEXT, UI_KEY_BG),
                (self._focus_status_label(), UI_TEXT, UI_KEY_BG),
            ],
        )

        # ------------------------------------------------------------------
        # Bottom hotkey rail - leaves room above for thumbs.  Labels are
        # plain English so a first-time user can guess what each key does.
        # ------------------------------------------------------------------
        _ui_hotkey_bar(
            canvas,
            [
                ("C", "capture this page", UI_ACCENT),
                ("D", "finish & save PDF", UI_SUCCESS),
                ("X", "delete last page", UI_WARN),
                ("M", "switch color / B&W", UI_KEY_BG),
                ("[ ]", "focus nearer / farther", UI_KEY_BG),
                ("F", "one-shot autofocus", UI_KEY_BG),
                ("N", "start a new document", UI_KEY_BG),
                ("Q", "quit app", UI_ERR),
            ],
            message=self.last_message or None,
            y=h - UI_HOTKEY_BAR_H,
        )

        # ------------------------------------------------------------------
        # AUTO pill - only when the feature is on (just below status bar).
        # ------------------------------------------------------------------
        if self.auto_capture_enabled:
            phase = self._auto_capture_phase
            if phase == "cooldown":
                pill_text = (
                    f"Auto-capture: saved page {self.page_count()}, "
                    f"next one in {self.auto_capture_cooldown_s:.1f}s"
                )
                pill_color, pill_bg = UI_TEXT, UI_SUCCESS
            elif phase == "identifying":
                c, r = self._auto_capture_progress
                pill_text = (
                    f"Auto-capture: confirming this page "
                    f"({c} of {r} frames stable)"
                )
                pill_color, pill_bg = UI_TEXT, UI_ACCENT
            elif phase == "idle":
                pill_text = "Auto-capture: on, waiting for a new page"
                pill_color, pill_bg = UI_TEXT, UI_MUTED
            else:
                pill_text, pill_bg = "", None
            if pill_text:
                _ui_chip(canvas, pill_text, 10, UI_STATUS_BAR_H + 8,
                         fg=pill_color, bg=pill_bg, scale=0.5, thickness=2, pad=10)

        # ------------------------------------------------------------------
        # Quality readout just above the hotkey rail (so it's still visible).
        # Yellow when the gate is currently rejecting, green when C will
        # succeed, plain grey before we've ever sampled a frame.
        # ------------------------------------------------------------------
        rail_top = h - UI_HOTKEY_BAR_H
        readout_y = min(rail_top - 30, grid_label_bottom(panels, DEBUG_GRID_SCALE) + 20)
        if self.last_quality is not None:
            q = self.last_quality
            qcolor = UI_SUCCESS if q.ok else UI_WARN
            if q.ok:
                qtext = (
                    f"Quality looks good — press C to save this page "
                    f"(focus {q.blur:.0f}, brightness {q.brightness:.0f}, "
                    f"movement {q.motion:.1f}px)"
                )
            else:
                # Strip the "rejected: " / "accepted: " prefix the gate
                # prepends so the user sees a plain reason instead of a
                # double verb.
                reason = (q.reason or "this frame is too blurry to save")
                prefix = reason.split(":", 1)[-1].strip() if ":" in reason else reason
                qtext = (
                    f"Quality: not ready to save — {prefix} "
                    f"(focus {q.blur:.0f}, brightness {q.brightness:.0f}, "
                    f"movement {q.motion:.1f}px)"
                )
        else:
            qcolor = UI_MUTED
            qtext = "Quality: press C to check this frame"
        _ui_chip(canvas, qtext, 10, readout_y, fg=UI_TEXT, bg=qcolor,
                 scale=0.5, thickness=1, pad=8)

        # ------------------------------------------------------------------
        # Thumb of the final processed page in the bottom-right of the grid,
        # so the user can see what would be saved on C.  Framed with the
        # success accent.
        # ------------------------------------------------------------------
        thumb = _make_thumb(processed, size=160)
        if thumb is not None and w > thumb.shape[1] + 20:
            tx = w - thumb.shape[1] - 20
            ty = rail_top - thumb.shape[0] - 30
            if ty > UI_STATUS_BAR_H + 10:
                canvas[ty : ty + thumb.shape[0], tx : tx + thumb.shape[1]] = thumb
                _ui_panel(canvas, tx - 4, ty - 4,
                          thumb.shape[1] + 8, thumb.shape[0] + 8,
                          fill=UI_PANEL, border=UI_SUCCESS, border_thickness=2)
                _ui_chip(canvas, "Preview — this is what C will save",
                         tx, ty - 12,
                         fg=UI_TEXT, bg=UI_SUCCESS, scale=0.45, thickness=1, pad=6)

        # ------------------------------------------------------------------
        # Thumb of the last successfully captured page in the bottom-left
        # so the user has a visible confirmation of which page ``X`` will
        # delete.  Falls back to a small "no pages yet" plaque when the
        # session is empty.
        # ------------------------------------------------------------------
        last_thumb = _make_thumb(self.pages[-1], size=160) if self.pages else None
        if last_thumb is not None and h > last_thumb.shape[0] + 20:
            lx = 20
            ly = rail_top - last_thumb.shape[0] - 30
            if ly > UI_STATUS_BAR_H + 10:
                canvas[ly : ly + last_thumb.shape[0], lx : lx + last_thumb.shape[1]] = last_thumb
                _ui_panel(canvas, lx - 4, ly - 4,
                          last_thumb.shape[1] + 8, last_thumb.shape[0] + 8,
                          fill=UI_PANEL, border=UI_WARN, border_thickness=2)
                label = f"Page {self.page_count()} saved — press X to undo"
                _ui_chip(canvas, label, lx, ly - 12,
                         fg=UI_TEXT, bg=UI_WARN, scale=0.45, thickness=1, pad=6)

        if self.show_exit_modal:
            self._draw_exit_modal(canvas)

        return canvas

    # ------------------------------------------------------------------ #
    def _render_camera_offline(
        self, status: dict, last_frame: Optional[np.ndarray]
    ) -> np.ndarray:
        """Modern 'camera not found' card with a centred status grid and a
        soft red wash so the offline state is impossible to miss."""
        h, w = self.camera_height, self.camera_width
        canvas = (
            last_frame.copy()
            if (last_frame is not None and last_frame.shape[:2] == (h, w))
            else np.zeros((h, w, 3), dtype=np.uint8)
        )
        # Base wash: deep navy + soft red overlay so the state is impossible
        # to miss while staying on-palette with the rest of the UI.
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), UI_BG, thickness=-1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        wash = canvas.copy()
        cv2.rectangle(wash, (0, 0), (w, h), (60, 26, 38), thickness=-1)
        cv2.addWeighted(wash, 0.45, canvas, 0.55, 0, canvas)

        src = status.get("source", "?")
        backend = status.get("backend", "?")
        err = status.get("error") or "no signal from the camera"
        retry_in = float(status.get("retry_in") or 0.0)

        # Friendly label for the backend so non-engineers know what it means.
        backend_phrase = {
            "opencv": "OpenCV (network / USB camera)",
            "picamera2": "Raspberry Pi camera module",
            "auto": "Auto-detect (Pi camera or USB)",
        }.get(backend, backend)

        # Top status bar with offline pill.
        _ui_status_bar(
            canvas,
            title="Camera offline",
            subtitle="Looking for the camera — the app will keep trying.",
            pills=[
                ("OFFLINE", UI_TEXT, UI_ERR),
            ],
        )

        # Centred "error card" with reason, source/backend, retry countdown.
        card_w = min(w - 80, 720)
        card_h = 280
        card_x = (w - card_w) // 2
        card_y = UI_STATUS_BAR_H + 40
        _ui_panel(canvas, card_x, card_y, card_w, card_h,
                  fill=UI_PANEL_ALT, border=UI_ERR, border_thickness=2)

        cx = card_x + 32
        cy = card_y + 24

        # Big banner inside the card.
        _draw_text(canvas, "No camera detected", (cx, cy + 30),
                   color=UI_ERR, bg=None, scale=1.0, thickness=2)
        cy += 60

        def _row(label, value, *, value_color=UI_TEXT):
            nonlocal cy
            _draw_text(canvas, label, (cx, cy + 22),
                       color=UI_MUTED, bg=None, scale=0.55, thickness=1)
            _draw_text(canvas, str(value), (cx + 200, cy + 22),
                       color=value_color, bg=None, scale=0.55, thickness=1)
            cy += 36

        _row("Camera URL", src)
        _row("Stream type", backend_phrase)
        _row("Why it failed", err, value_color=UI_WARN)
        _row("Trying again in", f"{retry_in:0.1f} seconds", value_color=UI_ACCENT)

        # Tip pinned to the bottom of the card.
        tip_y = card_y + card_h - 50
        cv2.line(canvas, (cx, tip_y - 18), (card_x + card_w - 32, tip_y - 18),
                 UI_BORDER, thickness=1)
        _draw_text(
            canvas,
            "Tip: open the DroidCam (or IP Webcam) app on your phone, then "
            "double-check the IP:port number it shows.",
            (cx, tip_y + 12),
            color=UI_MUTED, bg=None, scale=0.5, thickness=1,
        )

        # Bottom hotkey rail - only Q to quit makes sense in this state.
        retry_msg = (
            f"The app will keep trying every {CAMERA_RETRY_SECONDS:.0f} "
            f"seconds until the camera responds."
        )
        _ui_hotkey_bar(
            canvas,
            [("Q", "quit app", UI_ERR)],
            message=retry_msg,
        )

        if self.show_exit_modal:
            self._draw_exit_modal(canvas)
        return canvas

    # ------------------------------------------------------------------ #
    def _render_live_fallback(self, frame: np.ndarray) -> np.ndarray:
        """Single-tile fallback when ``process_with_debug`` or stacking fails.

        Modern chrome (top status bar + bottom hotkey rail) wraps the raw
        frame so the user still sees the state and can capture / finish.
        """
        try:
            processed, detection = self.processor.process(frame)
            canvas = DocumentProcessor.draw_overlay(
                frame, detection, processed_preview=processed
            )
        except Exception:
            canvas = frame.copy()
            _draw_text(canvas, "Detection error — showing the raw camera view.",
                       (10, 30), color=UI_ERR, bg=UI_PANEL)

        # Bottom strip so the camera frame doesn't bleed under the chrome.
        _ui_panel(canvas, 0, canvas.shape[0] - UI_HOTKEY_BAR_H,
                  canvas.shape[1], UI_HOTKEY_BAR_H,
                  fill=UI_HOTKEY_BG, border=UI_BORDER, border_thickness=1)

        _ui_status_bar(
            canvas,
            title="Smart Document Scanner",
            subtitle="Live preview (simplified view)",
            pills=[
                ("LIVE", UI_TEXT, UI_SUCCESS),
                (f"{self.page_count()} page{'s' if self.page_count() != 1 else ''} captured",
                 UI_TEXT, UI_ACCENT),
                (f"mode: {self.scan_mode}", UI_TEXT, UI_KEY_BG),
            ],
        )

        _ui_hotkey_bar(
            canvas,
            [
                ("C", "capture this page", UI_ACCENT),
                ("D", "finish & save PDF", UI_SUCCESS),
                ("X", "delete last page", UI_WARN),
                ("M", "switch color / B&W", UI_KEY_BG),
                ("N", "start a new document", UI_KEY_BG),
                ("Q", "quit app", UI_ERR),
            ],
            message=self.last_message or None,
        )

        if self.show_exit_modal:
            self._draw_exit_modal(canvas)
        return canvas

    # ------------------------------------------------------------------ #
    def _render_pdf_view(self) -> np.ndarray:
        # NOTE: the camera frame is intentionally NOT drawn here - per spec.
        h, w = self.camera_height, self.camera_width
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = UI_BG

        # Top status bar
        page_label = "No pages yet"
        if self.last_pdf_path is not None:
            n_pages = max(self.page_count_for_pdf(self.last_pdf_path), 0)
            if n_pages == 0:
                page_label = "0 pages captured"
            elif n_pages == 1:
                page_label = "1 page captured"
            else:
                page_label = f"{n_pages} pages captured"
        _ui_status_bar(
            canvas,
            title="Document saved",
            subtitle="Scan the QR code or open the link on your phone.",
            pills=[
                ("PDF VIEW", UI_TEXT, UI_SUCCESS),
                (page_label, UI_TEXT, UI_ACCENT),
            ],
        )

        # Bottom hotkey rail
        _ui_hotkey_bar(
            canvas,
            [
                ("N", "start a new document", UI_ACCENT),
                ("Q", "quit app", UI_ERR),
            ],
            message=self.last_message or None,
            y=h - UI_HOTKEY_BAR_H,
        )

        # Centred info card holding the file / url / QR.
        card_w, card_h = 720, 420
        card_x = (w - card_w) // 2
        card_y = UI_STATUS_BAR_H + (h - UI_STATUS_BAR_H - UI_HOTKEY_BAR_H - card_h) // 2
        _ui_panel(canvas, card_x, card_y, card_w, card_h,
                  fill=UI_PANEL_ALT, border=UI_SUCCESS, border_thickness=2)

        # Header
        _draw_text(canvas, "Document saved!",
                   (card_x + 24, card_y + 40),
                   color=UI_SUCCESS, scale=0.9, thickness=2)
        _draw_text(canvas, "Scan the QR code or open the link on your phone.",
                   (card_x + 24, card_y + 70),
                   color=UI_MUTED, scale=0.55, thickness=1)

        # PDF name row
        row_y = card_y + 110
        if self.last_pdf_path is not None:
            n_pages = max(self.page_count_for_pdf(self.last_pdf_path), 0)
            _draw_text(canvas, f"File: {self.last_pdf_path.name}",
                       (card_x + 24, row_y),
                       color=UI_TEXT, scale=0.55, thickness=1)
            row_y += 28
            page_phrase = (
                "no pages inside"
                if n_pages == 0
                else f"{n_pages} page{'s' if n_pages != 1 else ''} inside"
            )
            _draw_text(canvas, f"Pages: {page_phrase}",
                       (card_x + 24, row_y),
                       color=UI_TEXT, scale=0.55, thickness=1)
            row_y += 28

            host = self.flask_server.host or self.web_host
            port = self.flask_server.port or self.web_port
            url = f"http://{host}:{port}/{self.last_pdf_path.name}"
            _draw_text(canvas, f"Download link: {url}",
                       (card_x + 24, row_y),
                       color=UI_ACCENT, scale=0.55, thickness=1)

        # QR on the right side of the card
        qr_size = 220
        if self.last_qr_path is not None and self.last_qr_path.exists():
            qr_img = cv2.imread(str(self.last_qr_path), cv2.IMREAD_COLOR)
            if qr_img is not None:
                qr_img = cv2.resize(qr_img, (qr_size, qr_size),
                                    interpolation=cv2.INTER_AREA)
                qx = card_x + card_w - qr_size - 24
                qy = card_y + (card_h - qr_size) // 2
                canvas[qy:qy + qr_size, qx:qx + qr_size] = qr_img
                _ui_panel(canvas, qx - 6, qy - 6,
                          qr_size + 12, qr_size + 12,
                          fill=UI_PANEL, border=UI_ACCENT, border_thickness=2)
                _ui_chip(canvas, "Point your phone here", qx, qy + qr_size + 12,
                         fg=UI_TEXT, bg=UI_ACCENT, scale=0.45, thickness=1, pad=6)

        # Divider near the bottom of the card
        divider_y = card_y + card_h - 18
        cv2.line(canvas, (card_x + 24, divider_y),
                 (card_x + card_w - 24, divider_y), UI_BORDER, 1)

        if self.show_exit_modal:
            self._draw_exit_modal(canvas)

        return canvas

    # ------------------------------------------------------------------ #
    def _draw_exit_modal(self, canvas: np.ndarray) -> None:
        """Modern centred exit-confirm dialog with Y/N key chips.

        The whole-screen overlay softens whatever is behind so the modal
        reads as a focused decision surface.  Colors come from the new
        palette so it stays consistent with the chrome in the other states.
        """
        h, w = canvas.shape[:2]
        # Soften the entire frame first so the modal really stands out.
        dim = canvas.copy()
        cv2.rectangle(dim, (0, 0), (w, h), UI_BG, thickness=-1)
        cv2.addWeighted(dim, 0.55, canvas, 0.45, 0, canvas)

        box_w, box_h = 520, 230
        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2

        _ui_panel(canvas, x0, y0, box_w, box_h,
                  fill=UI_PANEL_ALT, border=UI_WARN, border_thickness=2)

        # Accent strip across the top of the modal so it pops even when
        # overlaid on the LIVE grid (which is busy).
        _ui_panel(canvas, x0, y0, box_w, 6,
                  fill=UI_WARN, border=UI_WARN, border_thickness=0)

        if self.page_count() > 0 and self.last_pdf_path is None:
            title = "Save this document and quit?"
        else:
            title = "Quit the app?"

        _draw_text(canvas, title, (x0 + 28, y0 + 56),
                   color=UI_TEXT, scale=0.9, thickness=2)

        if self.page_count() > 0 and self.last_pdf_path is None:
            n = self.page_count()
            pages_word = "page" if n == 1 else "pages"
            subtitle = (
                f"You have {n} {pages_word} that haven't been saved "
                "as a PDF yet."
            )
        else:
            subtitle = "Anything you haven't saved will be lost."
        _draw_text(canvas, subtitle, (x0 + 28, y0 + 92),
                   color=UI_MUTED, scale=0.55, thickness=1)

        # Y / N key chips on the bottom row of the modal.
        chip_y = y0 + box_h - 64
        y_chip_x = x0 + 28
        n_chip_x = _ui_key_chip(canvas, "Y", "yes, quit", y_chip_x, chip_y, accent=UI_SUCCESS)
        # gap then N chip
        n_chip_x = max(n_chip_x + 28, y_chip_x + 220)
        _ui_key_chip(canvas, "N", "no, stay", n_chip_x, chip_y, accent=UI_ERR)

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

        # Decide window mode.  The desktop users overwhelmingly prefer a
        # resizable window (so they can see the taskbar / drag the window
        # between monitors / use Win+arrow snap), so the default flipped
        # to "windowed".  ``--fullscreen`` keeps the old behaviour.
        if self.window_fullscreen:
            # ``WINDOW_FULLSCREEN`` keeps the OS title bar hidden and is
            # supported on Windows / macOS / Linux.  If the platform
            # rejects the property (some headless / WSL setups) the
            # window simply stays in its normal (resizable) state.
            try:
                cv2.setWindowProperty(
                    WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("could not enable fullscreen window: %s", exc)
        else:
            # Windowed mode: size the OpenCV canvas to whatever the user
            # gave us (default 1280x720).  We give the OS a sensible
            # aspect ratio and let the user resize afterwards.
            try:
                cv2.resizeWindow(
                    WINDOW_TITLE,
                    int(self.camera_width),
                    int(self.camera_height),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("could not resize window: %s", exc)
        try:
            while not self.quit_requested:
                # Camera heartbeat FIRST: keeps the "retry in X.Xs"
                # countdown honest and prevents auto-capture / render
                # from calling ``camera.read()`` before we've had a
                # chance to recover the handle.
                self._ensure_camera_alive()

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
    p.add_argument(
        "--backend",
        default="auto",
        choices=["opencv", "picamera2", "auto"],
        help=("Camera backend. 'auto' (default) picks picamera2 on a "
              "Raspberry Pi and opencv elsewhere. URL sources always "
              "collapse to opencv regardless of this flag."),
    )
    p.add_argument(
        "--autofocus",
        dest="autofocus",
        action="store_true",
        default=None,
        help=("Enable continuous autofocus on the Pi camera (default: "
              "fixed focus, see --lens-position).  Ignored on OpenCV / "
              "phone-stream sources."),
    )
    p.add_argument(
        "--no-autofocus",
        dest="autofocus",
        action="store_false",
        help="Disable continuous autofocus (default).",
    )
    p.add_argument(
        "--lens-position",
        type=float,
        default=None,
        help=("Manual focus distance in dioptres (1/metres) used when "
              "autofocus is off. 2.2 ≈ 45 cm (typical desk mount); lower "
              "for a higher stand, higher for a desk-flat scanner. "
              "Ignored on OpenCV / phone-stream sources."),
    )
    p.add_argument("--scan-mode", default=SCAN_MODE,
                   choices=["color", "grayscale", "bw"],
                   help="Output style for captured pages (default: %(default)s)")
    p.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help=("Rotate captured frames by N degrees clockwise "
                         "before they leave the Camera wrapper.  Use 90 or "
                         "270 to switch a landscape sensor into portrait "
                         "orientation; 0 (default) keeps the sensor as-is. "
                         "Applied on both OpenCV and picamera2 backends."))
    p.add_argument("--full-fov", dest="full_fov", action="store_true",
                   default=True,
                   help=("Pi-camera only.  Use the full sensor area "
                         "(ScalerCrop=(0,0,1,1)) so the LIVE preview matches "
                         "``rpicam-hello --width W --height H`` instead of "
                         "being cropped (digitally zoomed) to the main-stream "
                         "aspect ratio.  This is the default; pass "
                         "--no-full-fov to recover the legacy picamera2 "
                         "auto-crop.  Ignored on OpenCV."))
    p.add_argument("--no-full-fov", dest="full_fov", action="store_false",
                   help=("Pi-camera only.  Allow libcamera to crop the sensor "
                         "to the requested aspect ratio (legacy behaviour, "
                         "looks digitally zoomed on a 4:3 sensor)."))
    p.add_argument("--autofocus-on-capture", dest="autofocus_on_capture",
                   action="store_true", default=False,
                   help=("Pi-camera only.  Before each captured frame, kick a "
                         "single-shot autofocus cycle (libcamera AfMode=Auto) "
                         "and wait up to 2s for AfStatus=Focused.  Use this "
                         "when the document-to-lens distance varies between "
                         "pages; leave it off (default) for fixed-distance "
                         "desk scans where the extra latency is wasted. "
                         "Honoured only when --backend picamera2 is active."))
    p.add_argument("--fullscreen", dest="fullscreen", action="store_true",
                   default=False,
                   help=("Open the OpenCV window in fullscreen mode (default: "
                         "windowed, i.e. resizable)."))
    p.add_argument("--windowed", dest="fullscreen", action="store_false",
                   help=("Force windowed (resizable) mode.  This is the "
                         "default; the flag exists for symmetry with "
                         "--fullscreen."))
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
    p.add_argument("--mp3", dest="mp3", action="store_true",
                   default=None,
                   help=("Enable long-form MP3 cues (default: enabled). "
                         "Plays captured.mp3 / deleted.mp3 on the Pi 5 "
                         "MAX98357A I2S amp."))
    p.add_argument("--no-mp3", dest="mp3", action="store_false",
                   default=None,
                   help="Disable long-form MP3 cues (--mp3 / --no-mp3).")
    p.add_argument("--mp3-device", type=str, default=None,
                   help=("ALSA device for MP3 cues "
                         "(default: plughw:2,0 = MAX98357A I2S amp)."))
    p.add_argument("--mp3-volume", type=float, default=None,
                   help=("Software gain in dB applied to MP3 cues before "
                         "write (default: +8.0).  Increase for louder "
                         "playback; the MAX98357A has no hw mixer."))
    p.add_argument("--mp3-captured", type=str, default=None,
                   help=("Path to the MP3 played on a successful page "
                         "capture (default: <project_root>/captured.mp3)."))
    p.add_argument("--mp3-deleted", type=str, default=None,
                   help=("Path to the MP3 played on page deletion "
                         "(default: <project_root>/deleted.mp3)."))
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
        camera_autofocus=getattr(args, "autofocus", None),
        camera_lens_position=getattr(args, "lens_position", None),
        camera_rotate=int(getattr(args, "rotate", 0) or 0),
        camera_full_fov=bool(getattr(args, "full_fov", True)),
        autofocus_on_capture=bool(getattr(args, "autofocus_on_capture", False)),
        web_host=args.host,
        web_port=args.port,
        scan_mode=args.scan_mode,
        window_fullscreen=bool(args.fullscreen),
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

    # MP3 cue CLI overrides -------------------------------------------------
    if getattr(args, "mp3", None) is True:
        session.mp3_enabled = True
    elif getattr(args, "mp3", None) is False:
        session.mp3_enabled = False
    if getattr(args, "mp3_device", None) is not None:
        session.mp3_device = str(args.mp3_device)
        # Force re-build so the new device string is honoured.
        session._mp3 = None
    if getattr(args, "mp3_volume", None) is not None:
        session.mp3_volume_db = float(args.mp3_volume)
        session._mp3 = None
    if getattr(args, "mp3_captured", None) is not None:
        session.mp3_captured_file = str(args.mp3_captured)
        session._mp3 = None
    if getattr(args, "mp3_deleted", None) is not None:
        session.mp3_deleted_file = str(args.mp3_deleted)
        session._mp3 = None
    logger.info(
        "MP3: enabled=%s device=%s volume=%+0.1f dB captured=%s deleted=%s",
        session.mp3_enabled, session.mp3_device, session.mp3_volume_db,
        session.mp3_captured_file, session.mp3_deleted_file,
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
