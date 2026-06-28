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
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    send_from_directory,
    url_for,
)

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
                kwargs={"host": self.host, "port": self.port, "debug": False, "use_reloader": False},
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

        return app
