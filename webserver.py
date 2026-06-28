"""
webserver.py
============

Flask app that exposes:

* ``GET  /``             — gallery of captured pages + Finish / QR buttons.
* ``GET  /captures/<f>`` — serve a page image.
* ``POST /build``        — render ``output/scan.pdf`` and return its filename.
* ``POST /finish``       — save the in-progress session as ``scan_N.pdf``.
* ``POST /quit``         — signal the camera loop to quit (with save).
* ``GET  /download/<f>`` — download a file from ``output/`` (PDF or QR PNG).
* ``POST /qr``           — build a QR PNG pointing at a given URL.

The Flask app is intentionally decoupled from the camera loop.  ``app.py``
starts it in a daemon thread so the OpenCV window and the website share the
same ``captures/`` and ``output/`` folders via a shared :class:`ScanSession`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from flask import (
    Flask,
    abort,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from flask_server import FlaskServer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app import ScanSession

logger = logging.getLogger(__name__)


def _list_session_pdfs(output_dir: Path) -> List[Path]:
    """Return ``scan_*.pdf`` files in ``output_dir`` sorted by their number."""
    pdfs = []
    for p in output_dir.glob("scan_*.pdf"):
        try:
            num = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        pdfs.append((num, p))
    pdfs.sort(key=lambda t: t[0])
    return [p for _, p in pdfs]


def create_app(session: "ScanSession") -> Flask:
    """Application factory.  Takes the shared :class:`ScanSession`."""
    return FlaskServer(session).create_app()
