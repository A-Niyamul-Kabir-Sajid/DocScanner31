"""End-to-end probe: feed raw_4.jpg through the REAL DocumentProcessor."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from document_processor import DocumentProcessor

RAW = ROOT / "captures" / "raw"
raws = sorted(RAW.glob("raw_*.jpg"))
src = raws[-1]
print(f"[input] {src}")

img = cv2.imread(str(src))
H, W = img.shape[:2]
print(f"[input] shape={img.shape}")

processor = DocumentProcessor(enable_yolo=False)
processed, detection = processor.process(img)
print(f"[detect]  corners={'YES' if detection.corners is not None else 'NO'}")
print(f"[detect]  confidence={detection.confidence:.3f}")
print(f"[detect]  bbox={detection.bbox}")
print(f"[process] shape={processed.shape}  (raw was {H}x{W})")

cropped = processed.shape[:2] != (H, W)
print(f"[process] cropped={cropped}")

out = ROOT / "captures" / "scanned" / f"processor_{src.stem}.jpg"
cv2.imwrite(str(out), processed)
print(f"[save]    {out}  ({out.stat().st_size} bytes)")
