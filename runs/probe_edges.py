"""Inspect what DocumentProcessor.process() actually does."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import RAW_DIR  # noqa: E402
from document_processor import DocumentProcessor  # noqa: E402


def main() -> int:
    files = sorted(RAW_DIR.glob("raw_*.jpg"), key=lambda p: p.stat().st_mtime)
    raw = files[-1]
    print(f"[probe] reading {raw}")
    frame = cv2.imread(str(raw))
    h, w = frame.shape[:2]
    print(f"[probe] frame shape = {w}x{h}")

    # Same construction as ScanSession uses.
    proc = DocumentProcessor(use_roi_detector=False)

    gray = proc._to_grayscale(frame)
    blurred = proc._gaussian_blur(gray)
    contrast = proc._clahe(blurred)
    edges = proc._auto_canny(contrast)
    closed = proc._dilate_then_erode(edges)

    cv2.imwrite(str(ROOT / "runs" / "probe_out" / "ed_gray.png"), gray)
    cv2.imwrite(str(ROOT / "runs" / "probe_out" / "ed_contrast.png"), contrast)
    cv2.imwrite(str(ROOT / "runs" / "probe_out" / "ed_canny.png"), edges)
    cv2.imwrite(str(ROOT / "runs" / "probe_out" / "ed_closed.png"), closed)

    n = int(np.count_nonzero(closed))
    print(f"[probe] closed-edge non-zero: {n} ({100*n/(w*h):.1f}% of frame)")
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[probe] contours found: {len(contours)}")
    for i, c in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:5]):
        a = cv2.contourArea(c)
        print(f"   #{i} area={a:.0f}  ({100*a/(w*h):.1f}% of frame)  approx(2%)={len(cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True))} pts")

    processed, detection = proc.process(frame)
    print(f"[probe] processed shape = {processed.shape}")
    print(f"[probe] detection.corners  = {detection.corners}")
    print(f"[probe] detection.confidence = {detection.confidence:.3f}")
    print(f"[probe] detection.bbox    = {detection.bbox}")

    cv2.imwrite(str(ROOT / "runs" / "probe_out" / "ed_processed.png"), processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())