"""
pdf_generator.py
================

Legacy compatibility wrapper around the modern PDFBuilder class.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pdf_builder import PDFBuilder


class PDFGenerator(PDFBuilder):
    """Compatibility wrapper around the modern PDFBuilder API."""

    def build_pdf(self, filename: str = "scan.pdf") -> Optional[Path]:
        return self.build(filename)

    def list_pages(self) -> List[Path]:
        return super().list_pages()

    @staticmethod
    def _sort_key(path: Path) -> tuple:
        return PDFBuilder._sort_key(path)
