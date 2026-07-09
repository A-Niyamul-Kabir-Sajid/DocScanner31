"""
smoke_modern_desktop_ui.py
==========================

Renders every desktop state through the modernized helpers in ``app.py``
without touching a real camera.  Each result is saved as a PNG so we can
visually confirm the chrome (status bar + hotkey rail) lays out the way
the rewritten ``_render_*`` methods intend.

Run from the project root:

    .venv\\Scripts\\python.exe runs\\smoke_modern_desktop_ui.py

It deliberately uses lightweight stubs for the camera / processor so a CI
runner without any hardware still produces deterministic artifacts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app import (  # noqa: E402
    DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH,
    ScanSession, ScannerState, QualityReport, UI_BG,
)
from camera import Camera  # noqa: E402


ARTIFACTS = HERE / "_artifacts" / "modern_ui"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Stub collaborators
# --------------------------------------------------------------------------- #
class StubCamera:
    """Camera stub. ``is_open=False`` triggers the offline renderer."""

    def __init__(self, *, online: bool):
        self._online = online
        self._shape = (DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3)
        # Mirror the attribute the real ``Camera`` exposes - the offline
        # renderer reads it for the reason row.
        self.last_open_error = "" if online else "TCP probe failed (timed out after 1.5s)"

    def is_open(self) -> bool:
        return self._online

    def read(self):
        if not self._online:
            return False, None
        return True, self._frame()

    def _frame(self) -> np.ndarray:
        # Gradient-ish pattern so processing has something to chew on.
        h, w, _ = self._shape
        gx = np.linspace(0, 255, w, dtype=np.uint8)
        gy = np.linspace(0, 255, h, dtype=np.uint8)
        xs = np.tile(gx, (h, 1))
        ys = np.tile(gy[:, None], (1, w))
        img = np.stack([xs, ys, ((xs + ys) // 2)], axis=-1).copy()
        cv2.rectangle(img, (40, 40), (w - 40, h - 40), (200, 200, 200), 4)
        return img

    def try_reopen(self) -> bool:
        return self._online

    def close(self) -> None:
        pass


class StubProcessor:
    """Processor stub that always succeeds and yields 'document found'."""

    def process(self, frame):
        from dataclasses import dataclass
        @dataclass
        class Det:
            confidence: float = 0.93
            found: bool = True
        return frame.copy(), Det()

    def process_with_debug(self, frame):
        h, w = frame.shape[:2]
        blank = np.zeros_like(frame)
        # brand the panels with descriptive tints so the saved PNGs are easy
        # to tell apart at a glance.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        contour = frame.copy(); cv2.drawContours(
            contour, [np.array([[60, 60], [w - 60, 60],
                                [w - 60, h - 60], [60, h - 60]],
                               dtype=np.int32)], -1, (0, 255, 0), 4)
        biggest = frame.copy(); cv2.polylines(
            biggest, [np.array([[60, 60], [w - 60, 60],
                                [w - 60, h - 60], [60, h - 60]],
                               dtype=np.int32)], True, (0, 255, 255), 4)
        warped = frame.copy()
        warped_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
        adaptive = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        return (
            frame.copy(),           # processed (saved page)
            self.process(frame)[1],
            gray, edges, contour, biggest,
            warped, warped_gray, adaptive,
        )


def _make_session(*, online: bool, with_pages: bool) -> ScanSession:
    """Build a ScanSession with stubs for the heavy collaborators."""
    session = ScanSession()
    session._camera = StubCamera(online=online)   # type: ignore[assignment]
    session._processor = StubProcessor()          # type: ignore[assignment]
    if with_pages:
        for i in range(3):
            canvas = np.full((DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3),
                             40 + i * 20, dtype=np.uint8)
            cv2.putText(canvas, f"page {i + 1}", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            session.pages.append(canvas)
    return session


def _save(canvas: np.ndarray, name: str) -> Path:
    path = ARTIFACTS / f"{name}.png"
    cv2.imwrite(str(path), canvas)
    return path


def _try_attr(obj, *names):
    """Read the first available attribute or return None."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


# --------------------------------------------------------------------------- #
# Renderers exercised
# --------------------------------------------------------------------------- #
def render_live(online: bool) -> np.ndarray:
    session = _make_session(online=online, with_pages=True)
    session.last_message = "auto-capture active in 2.1s"
    session.last_quality = QualityReport(
        ok=True, reason="stable", blur=320.0, brightness=140.0, motion=1.2)
    # Force the LIVE state.
    session.state = ScannerState.LIVE_SCANNER_MODE
    # Pull a synthetic frame (offline will short-circuit).
    frame = session._camera.read()[1] if online else None
    return session._render_live(frame)


def render_live_with_auto(online: bool) -> np.ndarray:
    session = _make_session(online=online, with_pages=True)
    session.last_message = "captured page 3"
    session.last_quality = QualityReport(
        ok=False, reason="blur too high", blur=80.0, brightness=120.0,
        motion=4.5)
    session.auto_capture_enabled = True
    session._auto_capture_phase = "identifying"
    session._auto_capture_progress = (4, 12)
    session.state = ScannerState.LIVE_SCANNER_MODE
    frame = session._camera.read()[1] if online else None
    return session._render_live(frame)


def render_live_fallback(online: bool) -> np.ndarray:
    session = _make_session(online=online, with_pages=True)
    session.last_message = "pipeline error - showing raw frame"
    session.state = ScannerState.LIVE_SCANNER_MODE
    frame = session._camera.read()[1] if online else None
    return session._render_live_fallback(frame)


def render_camera_offline() -> np.ndarray:
    session = _make_session(online=False, with_pages=False)
    status = {
        "online": False,
        "source": "http://192.168.1.42:4747/video",
        "backend": "opencv",
        "reason": "TCP probe failed (timed out after 1.5s)",
        "retry_in": 2.0,
    }
    return session._render_camera_offline(status, None)


def render_pdf_view() -> np.ndarray:
    session = _make_session(online=True, with_pages=True)
    session.last_message = "saved as scan_3.pdf"
    # Inject a fake PDF + QR artifact pointing at an actual file in tree so
    # the QR load doesn't crash.  Use the QR generator helper if available.
    qr_dir = session.qr_dir
    qr_dir.mkdir(parents=True, exist_ok=True)
    qr_img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.putText(qr_img, "QR STUB", (40, 180), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), 4)
    qr_path = qr_dir / "demo_qr.png"
    cv2.imwrite(str(qr_path), qr_img)
    session.last_qr_path = qr_path
    fake_pdf = qr_dir / "demo.pdf"
    # minimal PDF that page_count_for_pdf can count (handles 0 pages ok).
    fake_pdf.write_bytes(b"%PDF-1.4\n% fake\n")
    session.last_pdf_path = fake_pdf
    # Stub a flask server so the URL row has values.
    class _Srv:
        host = "127.0.0.1"
        port = 5050
    session._flask_server = _Srv()  # type: ignore[assignment]
    session.state = ScannerState.PDF_VIEW_MODE
    return session._render_pdf_view()


def render_exit_modal(state: str = "live") -> np.ndarray:
    if state == "live":
        session = _make_session(online=True, with_pages=True)
        canvas = render_live(online=True)
    else:
        session = _make_session(online=False, with_pages=False)
        canvas = render_camera_offline()
    session.show_exit_modal = True
    session._draw_exit_modal(canvas)
    return canvas


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    cases = [
        ("live_online",          lambda: render_live(online=True)),
        ("live_offline",         lambda: render_live(online=False)),
        ("live_auto_capture",    lambda: render_live_with_auto(online=True)),
        ("live_fallback",        lambda: render_live_fallback(online=True)),
        ("camera_offline",       lambda: render_camera_offline()),
        ("pdf_view",             lambda: render_pdf_view()),
        ("exit_modal_live",      lambda: render_exit_modal("live")),
        ("exit_modal_offline",   lambda: render_exit_modal("offline")),
    ]
    failures = 0
    t0 = time.time()
    for name, fn in cases:
        try:
            canvas = fn()
            assert isinstance(canvas, np.ndarray), f"{name}: not ndarray"
            assert canvas.shape[2] == 3, f"{name}: not 3-channel"
            assert canvas.dtype == np.uint8, f"{name}: not uint8"
            path = _save(canvas, name)
            print(f"  [ok] {name:22s} -> {path.name}  shape={canvas.shape}")
        except Exception as exc:  # pragma: no cover - diagnostic
            failures += 1
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    dt = time.time() - t0
    print(f"\nrendered {len(cases) - failures}/{len(cases)} states in {dt:.2f}s")
    print(f"artifacts: {ARTIFACTS}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())