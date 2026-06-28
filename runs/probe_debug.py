"""Debug the corner-refine internals on the latest raw frame."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import RAW_DIR  # noqa: E402
from corner_refiner import CornerRefiner  # noqa: E402


def latest_raw() -> Path:
    files = sorted(RAW_DIR.glob("raw_*.jpg"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no raw_*.jpg in {RAW_DIR}")
    return files[-1]


def main() -> int:
    raw = latest_raw()
    print(f"[debug] reading {raw}")
    frame = cv2.imread(str(raw))
    h, w = frame.shape[:2]
    print(f"[debug] frame shape = {w}x{h}")

    cr = CornerRefiner(min_area_ratio=0.05)

    # Path 1: direct edges on the full frame
    edges = cr._edges(frame)
    n_white = int(np.count_nonzero(edges))
    print(f"[debug] full-frame edges: non-zero={n_white} ({100*n_white/(w*h):.1f}% of frame)")
    cv2.imwrite(str(ROOT / "runs" / "probe_out" / "dbg_edges_full.png"), edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[debug] full-frame contours found: {len(contours)}")
    for i, c in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:5]):
        a = cv2.contourArea(c)
        print(f"   #{i} area={a:.0f}  ({100*a/(w*h):.1f}% of frame)  approx(2%)={len(cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True))} pts")

    corners, conf = cr.from_edges(edges, w, h)
    print(f"[debug] from_edges result: corners={None if corners is None else 'found'} confidence={conf:.3f}")

    # Path 2: simulate the YOLO ROI path - assume bbox covers most of the doc
    bbox = (194, 71, 1003, 600)
    x, y, bw, bh = bbox
    roi = frame[y:y+bh, x:x+bw]
    print(f"[debug] ROI shape = {roi.shape}")
    edges_roi = cr._edges(roi)
    n_white_roi = int(np.count_nonzero(edges_roi))
    print(f"[debug] ROI edges: non-zero={n_white_roi} ({100*n_white_roi/(bw*bh):.1f}% of ROI)")
    cv2.imwrite(str(ROOT / "runs" / "probe_out" / "dbg_edges_roi.png"), edges_roi)

    corners_roi, conf_roi = cr.refine(roi)
    print(f"[debug] refine result: corners={None if corners_roi is None else 'found'} confidence={conf_roi:.3f}")
    if corners_roi is not None:
        print(f"[debug] corners_roi = {corners_roi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())