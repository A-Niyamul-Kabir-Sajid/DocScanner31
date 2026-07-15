"""Flask web UI for the document scanner.

The server is started **once** on the first ``D`` press; subsequent ``D``
presses re-use the same thread (so DroidCam / picamera2 threads don't leak).

Routes
------
``GET  /``                - gallery (captures + saved PDFs + QR preview)
``GET  /latest``          - redirect to the most recent document_*.pdf
``GET  /pdf/<doc_id>``    - download ``document_NNN.pdf``
``GET  /qr/<doc_id>``     - serve ``document_NNN.png`` from ``output/qr``
``POST /quit``            - request the camera loop to exit
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from flask import (
    Flask,
    Response,
    abort,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from frame_bus import STAGE_KEYS

from config import (
    DEFAULT_WEB_HOST,
    DEFAULT_WEB_PORT,
    DOCUMENT_PREFIX,
    PDF_DIR,
    PROJECT_ROOT,
    QR_DIR,
    TEMPLATES_DIR,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app import ScanSession

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
def _list_document_pdfs(pdf_dir: Path):
    """Return (doc_id, path) pairs sorted descending by doc_id."""
    pdfs = []
    for p in pdf_dir.glob(f"{DOCUMENT_PREFIX}*.pdf"):
        try:
            num = int(p.stem[len(DOCUMENT_PREFIX):])
        except ValueError:
            continue
        pdfs.append((num, p))
    pdfs.sort(key=lambda t: t[0])
    return pdfs


def _list_document_qrs(qr_dir: Path):
    pngs = []
    for p in qr_dir.glob(f"{DOCUMENT_PREFIX}*.png"):
        try:
            num = int(p.stem[len(DOCUMENT_PREFIX):])
        except ValueError:
            continue
        pngs.append((num, p))
    pngs.sort(key=lambda t: t[0])
    return pngs


def _list_captures(scanned_dir: Path):
    if not scanned_dir.exists():
        return []
    pages = [
        p for p in scanned_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    pages.sort()
    return pages


# --------------------------------------------------------------------------- #
class FlaskServer:
    """Wraps a Flask app bound to a :class:`ScanSession`.

    The server is started once via :meth:`ensure_running`.  Repeated calls
    are idempotent.
    """

    def __init__(
        self,
        session: "ScanSession",
        *,
        host: str = DEFAULT_WEB_HOST,
        port: int = DEFAULT_WEB_PORT,
    ) -> None:
        self.session = session
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._app: Optional[Flask] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    def ensure_running(self) -> None:
        """Start the Flask server in a daemon thread (idempotent)."""
        if self.is_running:
            return
        with self._lock:
            if self.is_running:
                return
            app = self.create_app()
            self._app = app
            self._thread = threading.Thread(
                target=app.run,
                kwargs={
                    "host": self.host,
                    "port": self.port,
                    "debug": False,
                    "use_reloader": False,
                    # ``threaded=True`` is REQUIRED: the live MJPEG streams hold
                    # their connections open indefinitely, so a single-threaded
                    # dev server would block every other request behind them.
                    "threaded": True,
                },
                daemon=True,
                name="doc-scanner-flask",
            )
            self._thread.start()
            logger.info("Flask server started on http://%s:%d", self.host, self.port)

    # ------------------------------------------------------------------ #
    def create_app(self) -> Flask:
        scanned_dir = self.session.scanned_dir
        pdf_dir = self.session.output_dir / "pdf" if (self.session.output_dir / "pdf").exists() else PDF_DIR
        qr_dir = self.session.output_dir / "qr" if (self.session.output_dir / "qr").exists() else QR_DIR
        output_dir = self.session.output_dir

        app = Flask(
            __name__,
            template_folder=str(TEMPLATES_DIR),
            static_folder=str(TEMPLATES_DIR),
        )

        @app.route("/")
        def index():
            captures = _list_captures(scanned_dir)
            pdfs = _list_document_pdfs(pdf_dir)
            qrs = _list_document_qrs(qr_dir)
            latest = pdfs[-1] if pdfs else None
            return render_template(
                "index.html",
                captures=[p.name for p in captures],
                pdfs=[(num, p.name) for num, p in pdfs],
                qrs=[(num, p.name) for num, p in qrs],
                latest=latest[1] if latest else None,
                latest_num=latest[0] if latest else 0,
                latest_url=url_for("download_pdf", doc_id=latest[0]) if latest else None,
                latest_qr=url_for("download_qr", doc_id=latest[0]) if latest else None,
                session_pages=self.session.page_count(),
            )

        @app.route("/latest")
        def latest():
            pdfs = _list_document_pdfs(pdf_dir)
            if not pdfs:
                abort(404, "No documents yet — press D to finish the current session.")
            doc_id, _path = pdfs[-1]
            return redirect(url_for("download_pdf", doc_id=doc_id))

        @app.route("/pdf/<int:doc_id>")
        def download_pdf(doc_id: int):
            target = pdf_dir / f"{DOCUMENT_PREFIX}{doc_id:03d}.pdf"
            if not target.exists():
                abort(404, f"document_{doc_id:03d}.pdf not found")
            return send_from_directory(pdf_dir, target.name, as_attachment=True)

        @app.route("/qr/<int:doc_id>")
        def download_qr(doc_id: int):
            target = qr_dir / f"{DOCUMENT_PREFIX}{doc_id:03d}.png"
            if not target.exists():
                abort(404, f"document_{doc_id:03d}.png not found")
            return send_from_directory(qr_dir, target.name)

        @app.route("/download/<path:filename>")
        def download_output(filename: str):
            # Legacy route used by the templates from earlier versions.
            return send_from_directory(output_dir, filename, as_attachment=True)

        @app.route("/captures/<path:filename>")
        def serve_capture(filename: str):
            return send_from_directory(scanned_dir, filename)

        @app.route("/quit", methods=["POST"])
        def quit_view():
            self.session.request_quit()
            return {"ok": True}

        @app.route("/delete-last-page", methods=["POST"])
        def delete_last_page():
            removed = self.session.delete_last_page()
            return {
                "ok": removed,
                "session_pages": self.session.page_count(),
                "last_message": self.session.last_message,
            }

        # ------------------------------------------------------------------ #
        # New endpoints for the modernized control panel.
        # All four are POST-friendly: the buttons in the UI POST to them and
        # the JSON body is ignored (kept light for a Raspberry Pi 5).
        # ------------------------------------------------------------------ #
        @app.route("/finish", methods=["POST"])
        def finish_view():
            pdf_path = self.session.finish_pdf()
            return {
                "ok": pdf_path is not None,
                "pdf_name": pdf_path.name if pdf_path else None,
                "session_pages": self.session.page_count(),
                "last_message": self.session.last_message,
            }

        @app.route("/new-session", methods=["POST"])
        def new_session_view():
            self.session.start_new_document()
            return {
                "ok": True,
                "session_pages": self.session.page_count(),
                "last_message": self.session.last_message,
            }

        @app.route("/quit", methods=["POST"], endpoint="ui_quit")
        def quit_view():
            # Optional ?save=1 / ?discard=1 hints from the UI; the desktop
            # window handles the actual save/discard decision before calling
            # request_quit(), but we accept the hints for symmetry.
            args = request.args
            if args.get("save") == "1":
                self.session.finish_pdf()
            # "discard=1" is implicit: we just quit without saving.
            self.session.request_quit()
            return {"ok": True, "last_message": self.session.last_message}

        @app.route("/capture", methods=["POST"])
        def capture_view():
            """Manually capture the current frame (the browser's ``C`` key)."""
            ok, msg = self.session.capture_page()
            return {
                "ok": bool(ok),
                "message": msg,
                "session_pages": self.session.page_count(),
                "last_message": self.session.last_message,
            }

        # ------------------------------------------------------------------ #
        # Live pipeline streaming.  The main camera loop publishes every stage
        # to ``session.frame_bus``; these endpoints just encode the latest one.
        # ------------------------------------------------------------------ #
        def _valid_stage(stage: str) -> bool:
            return stage in STAGE_KEYS

        def _clamp_width(raw) -> Optional[int]:
            try:
                w = int(raw)
            except (TypeError, ValueError):
                return None
            return max(64, min(1920, w)) if w > 0 else None

        @app.route("/frame/<stage>.jpg")
        def frame_snapshot(stage: str):
            """Single JPEG snapshot of one pipeline stage (thumbnails / fallback)."""
            if not _valid_stage(stage):
                abort(404, f"unknown stage {stage!r}")
            max_w = _clamp_width(request.args.get("w"))
            data = self.session.frame_bus.encode_jpeg(
                stage, quality=80, max_w=max_w, min_interval=0.0
            )
            return Response(data, mimetype="image/jpeg")

        @app.route("/stream/<stage>")
        def stream_stage(stage: str):
            """MJPEG (multipart/x-mixed-replace) live stream of one stage."""
            if not _valid_stage(stage):
                abort(404, f"unknown stage {stage!r}")
            max_w = _clamp_width(request.args.get("w"))
            bus = self.session.frame_bus
            # The static "last captured" panel changes only on capture, so it
            # streams slowly to save CPU; the live stages run ~12 fps.
            interval = 0.5 if stage == "last_captured" else 0.08

            def generate():
                boundary = b"--frame\r\n"
                while True:
                    try:
                        data = bus.encode_jpeg(
                            stage, quality=70, max_w=max_w,
                            min_interval=interval,
                        )
                        yield (
                            boundary
                            + b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(data)}\r\n\r\n".encode()
                            + data
                            + b"\r\n"
                        )
                        time.sleep(interval)
                    except GeneratorExit:  # client disconnected
                        break
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("stream %s ended: %s", stage, exc)
                        break

            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @app.route("/api/status")
        def api_status():
            """Lightweight JSON used by the polling loop in the UI."""
            captures = _list_captures(scanned_dir)
            pdfs = _list_document_pdfs(pdf_dir)
            qrs = _list_document_qrs(qr_dir)
            latest = pdfs[-1] if pdfs else None
            state = getattr(self.session, "state", None)
            return {
                "session_pages": self.session.page_count(),
                "captures": [p.name for p in captures],
                "pdfs": [num for num, _ in pdfs],
                "qrs": [num for num, _ in qrs],
                "latest": {
                    "num": latest[0] if latest else 0,
                    "pdf": latest[1].name if latest else None,
                },
                "last_message": getattr(self.session, "last_message", None),
                "scan_mode": getattr(self.session, "scan_mode", None),
                "state": state.value if hasattr(state, "value") else str(state),
                "auto_capture_phase": getattr(
                    self.session, "_auto_capture_phase", None
                ),
                "stages": list(STAGE_KEYS),
            }

        return app
