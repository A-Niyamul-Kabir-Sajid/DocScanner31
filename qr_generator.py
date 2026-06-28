"""
Generate QR code PNGs.

The QR always points at the device's LAN IP + Flask port so the printed
sheet can be scanned from a phone.  PNGs are written into ``output/qr``.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Optional

import qrcode

from config import (
    DEFAULT_WEB_HOST,
    DEFAULT_WEB_PORT,
    DOCUMENT_PREFIX,
    QR_DIR,
    QR_FILENAME_SUFFIX,
)

logger = logging.getLogger(__name__)


class QRGenerator:
    """High-error-correction QR code PNG generator."""

    def __init__(
        self,
        output_dir: Path = QR_DIR,
        *,
        host: str = DEFAULT_WEB_HOST,
        port: int = DEFAULT_WEB_PORT,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port

    # ------------------------------------------------------------------ #
    def make(
        self,
        data: str,
        filename: str = "qrcode.png",
        *,
        box_size: int = 10,
        border: int = 4,
    ) -> Path:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=box_size,
            border=border,
        )
        qr.add_data(data)
        qr.make(fit=True)
        image = qr.make_image(fill_color="black", back_color="white")
        target = self.output_dir / filename
        image.save(target)
        return target

    # ------------------------------------------------------------------ #
    def make_for_document(
        self,
        doc_id: int,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> Path:
        """Generate a QR PNG that points at the device-IP URL of document_NNN."""
        effective_host = host or self._discover_host()
        effective_port = port or self.port
        pdf_name = f"{DOCUMENT_PREFIX}{doc_id:03d}.pdf"
        url = f"http://{effective_host}:{effective_port}/{pdf_name}"
        filename = f"{DOCUMENT_PREFIX}{doc_id:03d}{QR_FILENAME_SUFFIX}"
        return self.make(url, filename=filename)

    # ------------------------------------------------------------------ #
    def _discover_host(self) -> str:
        """Return the device's LAN IP (best effort)."""
        if self.host and self.host not in ("0.0.0.0", "127.0.0.1", "localhost"):
            return self.host
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                ip = sock.getsockname()[0]
                if ip:
                    return ip
        except OSError:
            pass
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


# --------------------------------------------------------------------------- #
def document_qr_filename(doc_id: int, *, prefix: str = DOCUMENT_PREFIX) -> str:
    return f"{prefix}{doc_id:03d}{QR_FILENAME_SUFFIX}"