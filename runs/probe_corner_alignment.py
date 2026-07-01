"""More realistic probe: page corners are *inside* the YOLO bbox with margin.

The previous probe gave the bbox exactly at the paper edges so the bug was
masked.  Here we shrink the paper by 60 px on each side and let YOLO return
a bbox that hugs the paper with 20 px of padding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from document_processor import DocumentProcessor  # noqa: E402

H, W = 720, 1280
canvas = np.full((H, W, 3), (40, 30, 25), dtype=np.uint8)

# Inner paper corners.
gt = np.array(
    [
        [480, 270],   # tl
        [1130, 290],  # tr
        [1170, 610],  # br
        [440, 590],   # bl
    ],
    dtype=np.int32,
)

src = np.full((800, 600, 3), 245, dtype=np.uint8)
src[:, :, 0] = 240
M = cv2.getPerspectiveTransform(
    np.array([[0, 0], [599, 0], [599, 799], [0, 799]], dtype=np.float32),
    gt.astype(np.float32),
)
warped = cv2.warpPerspective(src, M, (W, H))
mask = cv2.warpPerspective(
    np.ones((800, 600), dtype=np.uint8),
    M,
    (W, H),
)
mask3 = cv2.merge([mask, mask, mask])
canvas = np.where(mask3 == 1, warped, canvas)

# YOLO bbox with 20 px padding around the page.
x, y = int(gt[:, 0].min()) - 20, int(gt[:, 1].min()) - 20
right, bottom = int(gt[:, 0].max()) + 20, int(gt[:, 1].max()) + 20
w, h = right - x, bottom - y


class FakeYOLO:
    def __init__(self, bbox):
        self._bbox = bbox
        self.scan_mode = "color"

    def detect(self, _frame):
        return self._bbox


proc = DocumentProcessor(
    scan_mode="color",
    enable_yolo=True,
    detector=FakeYOLO((x, y, w, h)),
    corner_refiner=None,
)

processed, detection = proc.process(canvas)
print(f"YOLO bbox returned:       {detection.bbox}")
print(f"Corners from pipeline:    {detection.corners.tolist() if detection.corners is not None else None}")
print(f"Ground truth (frame cs):  {gt.tolist()}")
if detection.corners is not None:
    err = np.linalg.norm(detection.corners.astype(float) - gt.astype(float), axis=1)
    print(f"Per-corner error (px):    {[f'{e:.1f}' for e in err]}")
    print(f"Mean error (px):          {err.mean():.1f}")

out = PROJECT / "runs" / "probe_out" / "corner_alignment2.png"
out.parent.mkdir(parents=True, exist_ok=True)
overlay = DocumentProcessor.draw_overlay(canvas, detection, processed_preview=None)
cv2.polylines(
    overlay,
    [gt.astype(int).reshape(-1, 1, 2)],
    isClosed=True,
    color=(0, 255, 255),
    thickness=3,
)
cv2.imwrite(str(out), overlay)
print(f"\nWrote overlay -> {out}")
print("  blue polygon = pipeline corners")
print("  cyan polygon = ground truth")