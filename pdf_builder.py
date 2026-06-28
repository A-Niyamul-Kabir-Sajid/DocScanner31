"""Build multi-page PDFs from processed page images.

Per the spec:
    - One PDF per scanning session
    - A4 portrait, 300 DPI
    - JPEG quality 90
    - Page order preserved
    - Images written to ``output/pdf/document_NNN.pdf``
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image

from config import (
    A4_HEIGHT_PX,
    A4_WIDTH_PX,
    DOCUMENT_PREFIX,
    JPEG_QUALITY,
    PDF_DIR,
    PDF_DPI,
)

logger = logging.getLogger(__name__)


class PDFBuilder:
    """Compose ordered processed pages into an A4 portrait 300 DPI PDF."""

    def __init__(
        self,
        pages_dir: Path,
        output_dir: Path = PDF_DIR,
        *,
        dpi: float = PDF_DPI,
        jpeg_quality: int = JPEG_QUALITY,
        page_width_px: int = A4_WIDTH_PX,
        page_height_px: int = A4_HEIGHT_PX,
    ) -> None:
        self.pages_dir = Path(pages_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.jpeg_quality = jpeg_quality
        self.page_width_px = page_width_px
        self.page_height_px = page_height_px

    # ------------------------------------------------------------------ #
    def build_from_paths(
        self,
        page_paths: Sequence[Path],
        filename: str,
    ) -> Optional[Path]:
        """Compose ``page_paths`` into a single PDF named ``filename``."""
        if not page_paths:
            return None

        pil_pages: List[Image.Image] = []
        for p in page_paths:
            img = Image.open(p)
            if img.mode != "RGB":
                img = img.convert("RGB")
            pil_pages.append(self._resize_to_a4(img))

        target = self.output_dir / filename
        first, *rest = pil_pages
        first.save(
            target,
            save_all=True,
            append_images=rest,
            format="PDF",
            resolution=float(self.dpi),
            quality=self.jpeg_quality,
            subsampling=2,
        )
        logger.info("Wrote %d-page PDF -> %s", len(pil_pages), target)
        return target

    def build(self, filename: str = "scan.pdf") -> Optional[Path]:
        """Compose all supported images in ``pages_dir`` into a PDF."""
        return self.build_from_paths(self.list_pages(), filename)

    # ------------------------------------------------------------------ #
    def list_pages(self) -> List[Path]:
        """Return sorted page files from ``pages_dir``."""
        if not self.pages_dir.exists():
            return []
        pages = [
            p for p in self.pages_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        pages.sort(key=_page_sort_key)
        return pages

    # ------------------------------------------------------------------ #
    def _resize_to_a4(self, image: Image.Image) -> Image.Image:
        if image.size == (self.page_width_px, self.page_height_px):
            return image
        return image.resize(
            (self.page_width_px, self.page_height_px),
            Image.LANCZOS,
        )


# --------------------------------------------------------------------------- #
def document_filename(doc_id: int, *, prefix: str = DOCUMENT_PREFIX) -> str:
    """Return the canonical ``document_NNN.pdf`` name."""
    return f"{prefix}{doc_id:03d}.pdf"


def _page_sort_key(path: Path) -> tuple:
    stem = path.stem
    try:
        num = int(stem.split("_")[-1])
        return (0, num)
    except ValueError:
        return (1, stem)
