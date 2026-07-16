"""Configuration constants for the Smart Document Scanner.

All defaults live here so every module imports a single source of truth.
The runtime ``AppConfig`` dataclass can be constructed with overrides for
testing, but production code should just import the module-level constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Window / app
# --------------------------------------------------------------------------- #
WINDOW_TITLE = "Smart Document Scanner  -  C capture  D finish PDF  N new session  Q quit"

# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CAPTURES_DIR: Path = PROJECT_ROOT / "captures"
SCANNED_DIR: Path = CAPTURES_DIR / "scanned"
RAW_DIR: Path = CAPTURES_DIR / "raw"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
PDF_DIR: Path = OUTPUT_DIR / "pdf"
QR_DIR: Path = OUTPUT_DIR / "qr"
TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"

# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
DEFAULT_CAMERA_WIDTH: int = 1280
DEFAULT_CAMERA_HEIGHT: int = 720
# Accepted ``backend`` values:
#   "opencv"    - ``cv2.VideoCapture`` (USB cams / DroidCam URLs on any OS)
#   "picamera2" - Raspberry Pi Camera Module via libcamera + picamera2
#   "auto"      - pick "picamera2" on a Pi, "opencv" everywhere else
DEFAULT_BACKEND: str = "auto"
# When the configured camera source is unreachable, the LIVE loop shows a
# "camera not found" overlay and re-attempts ``Camera.try_reopen()`` every
# CAMERA_RETRY_SECONDS seconds (the heartbeat tick comes from the same 30 ms
# render loop). The app stays alive indefinitely until the device comes back.
CAMERA_RETRY_SECONDS: float = 3.0

# A document scanner benefits more from fixed focus than continuous autofocus:
# the document sits at a known distance, AF hunting during a multi-page scan
# produces focus-driven blur frames, and the IMX519 + AK7375 stack on the Pi
# accepts manual ``LensPosition`` in *dioptres* (1/metres) via libcamera.
# ``DEFAULT_LENS_POSITION_DIOPTRES = 2.2`` corresponds to ~0.45 m (45 cm),
# a comfortable desk-to-document distance. Tune per setup via --lens-position.
DEFAULT_AUTOFOCUS: bool = False
DEFAULT_LENS_POSITION_DIOPTRES: float = 2.2

# --------------------------------------------------------------------------- #
# Document scanning pipeline
# --------------------------------------------------------------------------- #
SCAN_MODE: str = "color"                  # "color" | "grayscale" | "bw"
A4_WIDTH_PX: int = 1654                   # 8.27 in * 200 dpi  (long edge ~220 dpi A4)
A4_HEIGHT_PX: int = 2339                  # 11.69 in * 200 dpi
DOC_BORDER_CROP_PX: int = 20              # final crop after perspective warp
# Maximum upscale ratio between the warped crop and the final A4-sized
# output.  Phone cameras crop out a ~1000 px wide page; upscaling 2.4x to
# 300 dpi A4 produced a soft, waxy result (Laplacian variance drops from
# 300 -> 6).  Cap the upscale so the warped region is rendered at its
# natural resolution whenever possible.
MAX_UPSCALE_RATIO: float = 1.0            # 1.0 = never upscale; cap to source
JPEG_QUALITY: int = 90
SHADOW_REMOVAL: bool = False
SHARPEN: bool = True
# Hard sharpness floor that ALWAYS fires before a page lands in the PDF,
# even when QUALITY_GATE_ENABLED is False.  50 catches the "6.0 variance
# waxy upscale" failure mode while still accepting clean MJPEG streams.
ABSOLUTE_BLUR_MIN_VARIANCE: float = 50.0

# --------------------------------------------------------------------------- #
# Flask / web UI
# --------------------------------------------------------------------------- #
DEFAULT_WEB_HOST: str = "0.0.0.0"
DEFAULT_WEB_PORT: int = 5000

# --------------------------------------------------------------------------- #
# Document detection (OpenCV contour pass)
# --------------------------------------------------------------------------- #
# When True, ``DocumentProcessor`` first asks ``DocumentDetector`` for a coarse
# document bbox and refines the corners inside that ROI; if that yields nothing
# it falls back to refining against the full-frame edge map.  Set False to skip
# the ROI stage and always use the full-frame edge path.
USE_ROI_DETECTOR: bool = True
DOC_MIN_AREA_RATIO: float = 0.08          # doc must cover >= 8% of frame

# --------------------------------------------------------------------------- #
# Quality gate (per-capture)
# --------------------------------------------------------------------------- #
QUALITY_GATE_ENABLED: bool = False  # user asked to turn all rejection conditions off
# Laplacian variance threshold for the "blurry" check.  80 is fine for a
# scanner over a desk; phone-camera MJPEG streams over Wi-Fi typically
# land around 15-30 because of compression + autofocus hunting.  Use
# ``--blur-min`` on the CLI to override per session.
BLUR_MIN_VARIANCE: float = 15.0
BRIGHTNESS_MIN: float = 25.0              # reject near-black frames
BRIGHTNESS_MAX: float = 240.0             # reject near-white frames
# Mean absolute pixel difference between the previous and current processed
# frame.  12 px is fine for a stationary scanner, but MJPEG over Wi-Fi
# routinely shows 15-30 px just from macroblock shimmer, and *real* camera
# shake can push it to 60+.  Raise this on the CLI via ``--motion-max`` if
# "rejected: motion" keeps firing.
MOTION_MAX_PX: float = 60.0
CORNER_CONFIDENCE_MIN: float = 0.30

# --------------------------------------------------------------------------- #
# Auto capture
# --------------------------------------------------------------------------- #
# When enabled, the LIVE loop auto-captures the current frame whenever a
# stable document contour has been observed for ``DEFAULT_STABLE_FRAMES``
# consecutive frames (~2 s at the default 30 ms LIVE tick, i.e. 60 frames).
# After each capture the FSM parks in State 2 and only flips back to
# State 1 after ``DEFAULT_AUTO_CAPTURE_COOLDOWN`` seconds of continuous
# no-match, giving the user time to swap pages.
DEFAULT_AUTO_CAPTURE_ENABLED: bool = True
DEFAULT_AUTO_CAPTURE_COOLDOWN: float = 3.0  # 3 s no-match window in State 2 before flipping back to State 1

# --------------------------------------------------------------------------- #
# Audio cues - short WAV beeps for "doc detected / stable / captured".
# The ``sound`` module is pure-stdlib and self-synthesises the tones, so
# no extra pip dependency is required.  On Windows ``winsound`` is used
# directly; on macOS / Linux the platform's built-in audio player
# (``afplay`` / ``aplay`` / ``paplay``) is invoked through ``subprocess``.
# If no backend is available the calls silently no-op so the LIVE loop
# is never blocked or crashed by a missing sound card.
# --------------------------------------------------------------------------- #
DEFAULT_SOUND_ENABLED: bool = True
DEFAULT_SOUND_VOLUME: float = 0.6  # 0.0 (silent) - 1.0 (full)
DEFAULT_SOUND_SAMPLE_RATE: int = 22050
# ALSA I2S backend - same MAX98357A card as the long-form MP3 pipeline.
# ``plughw:2,0`` is the overlay card 2, device 0 on the Pi 5.
DEFAULT_SOUND_ALSA_DEVICE: str = "plughw:2,0"
DEFAULT_SOUND_ALSA_CHUNK_BYTES: int = 4096
# Stable corners required before auto-capture fires.  At the 30 ms LIVE
# tick (~33 fps) ``required_frames`` is roughly the wait time in seconds
# times 33, so the default 60 is ~1.8 s on the LIVE loop.  Bump this for a
# longer "hold still" window, drop it for snappier auto-capture.
DEFAULT_STABLE_FRAMES: int = 10  # ~1.8 s at the 30 ms LIVE tick
# Maximum corner drift (pixels) tolerated between consecutive frames.
# The contour + approxPolyDP pass routinely jitters 8-15 px even when the document
# is held still, so a tight 6 px threshold never lets the streak build.
# 18 px is comfortably above that noise floor but still below the drift
# you get from a slow hand swap (~40+ px).
DEFAULT_STABILITY_TOLERANCE: float = 18.0

# --------------------------------------------------------------------------- #
# Voice prompts (spoken cues layered on top of the tone ``sound`` module).
# The ``voice`` module renders WAV blobs via ``pyttsx3`` on Windows or
# ``espeak-ng`` (subprocess) on Linux / Raspberry Pi, then forwards them
# to ``sound.SoundPlayer._play_wav`` so tones and voice share the same
# audio backend.  Both backends are fully offline - no internet needed.
# --------------------------------------------------------------------------- #
DEFAULT_VOICE_ENABLED: bool = True
DEFAULT_VOICE_LANGUAGE: str = "en"      # espeak-ng voice id; "en", "en-us", "de", ...
DEFAULT_VOICE_RATE_WPM: int = 165       # speaking rate (pyttsx3 honours it; espeak gets ~170 wpm equivalent)
DEFAULT_VOICE_BACKEND: str = "auto"     # "pyttsx3" | "espeak" | "auto"

# --------------------------------------------------------------------------- #
# Long-form MP3 cues (Raspberry Pi 5 + MAX98357A I2S amp deployment).
#
# The ``mp3_player`` module plays user-supplied ``captured.mp3`` /
# ``deleted.mp3`` clips directly on the Pi 5's hardware I2S amp via
# ``pydub`` + ``pyalsaaudio`` (no PulseAudio / PipeWire in the path).
# On Windows / macOS / any host without pydub+alsaaudio+ffmpeg the
# player is a silent no-op so the dev box keeps working.
# --------------------------------------------------------------------------- #
DEFAULT_MP3_ENABLED: bool = True
DEFAULT_MP3_DEVICE: str = "plughw:2,0"   # MAX98357A card 2, device 0
DEFAULT_MP3_VOLUME_DB: float = 8.0       # +8 dB software gain (no hw mixer)
DEFAULT_MP3_CAPTURED_FILE: str = "captured.mp3"
DEFAULT_MP3_DELETED_FILE: str = "deleted.mp3"

# --------------------------------------------------------------------------- #
# Auto page-change detection
# --------------------------------------------------------------------------- #
# When enabled, ``PageChangeDetector`` watches the live pipeline for a
# "stable -> moved -> stable-again" gesture AND a phash delta on the warped
# A4 page.  On confirm it can auto-bump the in-session page counter and
# surface a "new page detected" message on the LIVE overlay.
PAGE_CHANGE_ENABLED: bool = True
# Hamming distance (over 64 bits) between the previous captured page's
# perceptual hash and the current one that qualifies as a *different* page.
PAGE_CHANGE_HASH_DISTANCE: int = 10
# Mean absolute pixel-difference that must be sustained before we declare
# the page "moved".  Mirrors QualityGate.MOTION_MAX_PX but in the other
# direction.
PAGE_CHANGE_MOTION_TRIGGER_PX: float = 25.0
# After a swap we wait until motion settles below this threshold for
# PAGE_CHANGE_REST_FRAMES consecutive frames before re-arming.
PAGE_CHANGE_MOTION_REST_PX: float = 6.0
PAGE_CHANGE_REST_FRAMES: int = 6
# Minimum corner-distance jump (px) required between the last stable quad
# and the new stable quad for the displacement arm to fire.
PAGE_CHANGE_QUAD_JUMP_PX: float = 35.0
# Whether the detector is allowed to bump the page counter automatically.
# Tests / demos can set this to False to observe events without side effects.
AUTO_PAGE_CHANGE_BUMP: bool = True

# --------------------------------------------------------------------------- #
# PDF / QR file naming
# --------------------------------------------------------------------------- #
DOCUMENT_PREFIX: str = "document_"
PAGE_PREFIX: str = "page_"
RAW_PREFIX: str = "raw_"
DOCUMENT_COUNTER_START: int = 1
PDF_DPI: float = 300.0
QR_FILENAME_SUFFIX: str = ".png"

# --------------------------------------------------------------------------- #
# Legacy aliases (kept so older imports keep working)
# --------------------------------------------------------------------------- #
DEFAULT_CAMERA_WIDTH = DEFAULT_CAMERA_WIDTH  # noqa: PLW0127
DEFAULT_CAMERA_HEIGHT = DEFAULT_CAMERA_HEIGHT  # noqa: PLW0127
DEFAULT_MIN_AREA_RATIO = DOC_MIN_AREA_RATIO
DEFAULT_OUTPUT_WIDTH = A4_WIDTH_PX
DEFAULT_OUTPUT_HEIGHT = A4_HEIGHT_PX
DEFAULT_PDF_DPI = PDF_DPI
DEFAULT_STABLE_FRAMES = DEFAULT_STABLE_FRAMES
DEFAULT_STABILITY_TOLERANCE = DEFAULT_STABILITY_TOLERANCE
DEFAULT_AUTO_CAPTURE_ENABLED = DEFAULT_AUTO_CAPTURE_ENABLED
DEFAULT_AUTO_CAPTURE_COOLDOWN = DEFAULT_AUTO_CAPTURE_COOLDOWN
DEFAULT_SOUND_ENABLED = DEFAULT_SOUND_ENABLED
DEFAULT_SOUND_VOLUME = DEFAULT_SOUND_VOLUME
DEFAULT_SOUND_SAMPLE_RATE = DEFAULT_SOUND_SAMPLE_RATE
DEFAULT_SOUND_ALSA_DEVICE = DEFAULT_SOUND_ALSA_DEVICE
DEFAULT_SOUND_ALSA_CHUNK_BYTES = DEFAULT_SOUND_ALSA_CHUNK_BYTES
DEFAULT_VOICE_ENABLED = DEFAULT_VOICE_ENABLED
DEFAULT_VOICE_LANGUAGE = DEFAULT_VOICE_LANGUAGE
DEFAULT_VOICE_RATE_WPM = DEFAULT_VOICE_RATE_WPM
DEFAULT_VOICE_BACKEND = DEFAULT_VOICE_BACKEND
PDF_DEFAULT_NAME = "scan.pdf"
PDF_PREFIX = "scan_"
DEFAULT_MP3_ENABLED = DEFAULT_MP3_ENABLED  # noqa: PLW0127
DEFAULT_MP3_DEVICE = DEFAULT_MP3_DEVICE  # noqa: PLW0127
DEFAULT_MP3_VOLUME_DB = DEFAULT_MP3_VOLUME_DB  # noqa: PLW0127
DEFAULT_MP3_CAPTURED_FILE = DEFAULT_MP3_CAPTURED_FILE  # noqa: PLW0127
DEFAULT_MP3_DELETED_FILE = DEFAULT_MP3_DELETED_FILE  # noqa: PLW0127

# --------------------------------------------------------------------------- #
# Runtime dataclass (overridable for tests)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    """Application configuration values."""

    camera_width: int = DEFAULT_CAMERA_WIDTH
    camera_height: int = DEFAULT_CAMERA_HEIGHT
    backend: str = DEFAULT_BACKEND
    web_host: str = DEFAULT_WEB_HOST
    web_port: int = DEFAULT_WEB_PORT
    output_width: int = DEFAULT_OUTPUT_WIDTH
    output_height: int = DEFAULT_OUTPUT_HEIGHT
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
    pdf_dpi: float = DEFAULT_PDF_DPI
    stable_frames: int = DEFAULT_STABLE_FRAMES
    stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE
    auto_capture_enabled: bool = DEFAULT_AUTO_CAPTURE_ENABLED
    auto_capture_cooldown: float = DEFAULT_AUTO_CAPTURE_COOLDOWN
    scan_mode: str = SCAN_MODE
    use_roi_detector: bool = USE_ROI_DETECTOR
    quality_gate_enabled: bool = QUALITY_GATE_ENABLED
    shadow_removal: bool = SHADOW_REMOVAL
    sharpen: bool = SHARPEN
    page_change_enabled: bool = PAGE_CHANGE_ENABLED
    page_change_hash_distance: int = PAGE_CHANGE_HASH_DISTANCE
    page_change_motion_trigger_px: float = PAGE_CHANGE_MOTION_TRIGGER_PX
    page_change_motion_rest_px: float = PAGE_CHANGE_MOTION_REST_PX
    page_change_rest_frames: int = PAGE_CHANGE_REST_FRAMES
    page_change_quad_jump_px: float = PAGE_CHANGE_QUAD_JUMP_PX
    voice_enabled: bool = DEFAULT_VOICE_ENABLED
    voice_language: str = DEFAULT_VOICE_LANGUAGE
    voice_rate_wpm: int = DEFAULT_VOICE_RATE_WPM
    voice_backend: str = DEFAULT_VOICE_BACKEND
    sound_alsa_device: str = DEFAULT_SOUND_ALSA_DEVICE
    sound_alsa_chunk_bytes: int = DEFAULT_SOUND_ALSA_CHUNK_BYTES
    mp3_enabled: bool = DEFAULT_MP3_ENABLED
    mp3_device: str = DEFAULT_MP3_DEVICE
    mp3_volume_db: float = DEFAULT_MP3_VOLUME_DB
    mp3_captured_file: str = DEFAULT_MP3_CAPTURED_FILE
    mp3_deleted_file: str = DEFAULT_MP3_DELETED_FILE
    extra: dict = field(default_factory=dict)


# Ensure runtime directories exist (idempotent).
for _d in (CAPTURES_DIR, SCANNED_DIR, RAW_DIR, OUTPUT_DIR, PDF_DIR, QR_DIR):
    _d.mkdir(parents=True, exist_ok=True)
