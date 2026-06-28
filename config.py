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
DEFAULT_BACKEND: str = "opencv"           # "opencv" | "picamera2"

# --------------------------------------------------------------------------- #
# Document scanning pipeline
# --------------------------------------------------------------------------- #
SCAN_MODE: str = "color"                  # "color" | "grayscale" | "bw"
A4_WIDTH_PX: int = 2480                   # 8.27 in * 300 dpi
A4_HEIGHT_PX: int = 3508                  # 11.69 in * 300 dpi
DOC_BORDER_CROP_PX: int = 20              # final crop after perspective warp
JPEG_QUALITY: int = 90
SHADOW_REMOVAL: bool = False
SHARPEN: bool = True

# --------------------------------------------------------------------------- #
# Flask / web UI
# --------------------------------------------------------------------------- #
DEFAULT_WEB_HOST: str = "0.0.0.0"
DEFAULT_WEB_PORT: int = 5000

# --------------------------------------------------------------------------- #
# Document detection (pure OpenCV, no YOLO)
# --------------------------------------------------------------------------- #
ENABLE_YOLO: bool = False                 # legacy flag - kept for back-compat
YOLO_MODEL_PATH: Path = PROJECT_ROOT / "yolov8n.pt"
YOLO_CONFIDENCE: float = 0.35
DOC_MIN_AREA_RATIO: float = 0.08          # doc must cover >= 8% of frame

# Contour scoring weights (sum = 1.0).  Tuned for typical desk-mounted webcams.
SCORE_W_AREA: float = 0.40
SCORE_W_RECTANGULARITY: float = 0.30
SCORE_W_CONVEXITY: float = 0.15
SCORE_W_ASPECT: float = 0.10
SCORE_W_CENTER: float = 0.05
# Acceptable aspect ratio range (w/h).  Receipts ~0.5, A4 ~0.71, books ~0.8.
ASPECT_MIN: float = 0.45
ASPECT_MAX: float = 1.55
# Canny / morphology tuning.
CANNY_KERNEL: int = 5                     # 5x5 Gaussian + closing kernel
HOUGH_THRESHOLD: int = 80                 # accumulator threshold for HoughLinesP
HOUGH_MIN_LINE: int = 60                  # min line length (px)
HOUGH_MAX_GAP: int = 10                   # max gap between line segments
MIN_QUAD_CONFIDENCE: float = 0.4          # reject detections below this
# Corner sub-pixel search window.
CORNERSUBPIX_WIN: int = 7
CORNERSUBPIX_ZEROZONE: int = -1
CORNERSUBPIX_CRITERIA_EPS: float = 0.01
CORNERSUBPIX_MAX_ITER: int = 40

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
DEFAULT_AUTO_CAPTURE_ENABLED: bool = False
DEFAULT_AUTO_CAPTURE_COOLDOWN: float = 3.0
DEFAULT_STABLE_FRAMES: int = 12
DEFAULT_STABILITY_TOLERANCE: float = 4.0   # pixels of corner drift tolerated

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
PDF_DEFAULT_NAME = "scan.pdf"
PDF_PREFIX = "scan_"

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
    enable_yolo: bool = ENABLE_YOLO
    quality_gate_enabled: bool = QUALITY_GATE_ENABLED
    shadow_removal: bool = SHADOW_REMOVAL
    sharpen: bool = SHARPEN
    extra: dict = field(default_factory=dict)


# Ensure runtime directories exist (idempotent).
for _d in (CAPTURES_DIR, SCANNED_DIR, RAW_DIR, OUTPUT_DIR, PDF_DIR, QR_DIR):
    _d.mkdir(parents=True, exist_ok=True)
