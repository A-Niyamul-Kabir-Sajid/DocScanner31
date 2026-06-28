"""
scanner.py
==========

Legacy compatibility wrapper for the new document processor implementation.
"""

from __future__ import annotations

from document_processor import DocumentProcessor


class DocumentScanner(DocumentProcessor):
    """Compatibility wrapper around the modern DocumentProcessor."""

    def scan(self, frame, quad):
        return self.process(frame, quad)
