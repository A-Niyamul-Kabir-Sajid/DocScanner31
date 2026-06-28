"""Run the LIVE pipeline once on the latest raw frame and dump each step.

Usage:
    .venv\\Scripts\\python.exe runs/probe_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import ScanSession  # noqa: E402
from camera import Camera  # noqa: E402
from config import RAW_DIR  # noqa: E402


def latest_raw() -> Path:
    files = sorted(RAW_DIR.glob("raw_*.jpg"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no raw_*.jpg found in {RAW_DIR}")
    return files[-1]


def main() -> int:
    raw_path = latest_raw()
    print(f"[probe] reading {raw_path} ({raw_path.stat().st_size} bytes)")
    frame = cv2.imread(str(raw_path))
    if frame is None:
        raise SystemExit("cv2.imread returned None")
    print(f"[probe] frame shape={frame.shape} dtype={frame.dtype}")

    sess = ScanSession(
        camera_source=0,
        camera_width=frame.shape[1],
        camera_height=frame.shape[0],
    )
    # Use the injected frame so we don't need the camera.
    sess._camera = Camera(0, width=frame.shape[1], height=frame.shape[0])  # not used

    ok, msg, processed, detection = sess.capture_current_frame(frame=frame)
    print(f"[probe] capture_current_frame -> ok={ok} msg={msg!r}")
    print(f"[probe] detection.bbox       = {detection.bbox}")
    print(f"[probe] detection.confidence = {detection.confidence:.3f}")
    print(f"[probe] detection.corners    = {detection.corners}")
    print(f"[probe] processed shape      = {None if processed is None else processed.shape}")

    # Dump each pipeline artefact for visual inspection.
    probe_dir = ROOT / "runs" / "probe_out"
    probe_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    cv2.imwrite(str(probe_dir / "01_gray.png"), gray)
    cv2.imwrite(str(probe_dir / "02_blurred.png"), blurred)
    cv2.imwrite(str(probe_dir / "03_edges.png"), edges)

    if detection.corners is not None:
        quad_vis = frame.copy()
        cv2.polylines(quad_vis, [detection.corners.astype(int)], True, (0, 255, 0), 3)
        cv2.imwrite(str(probe_dir / "04_quad.png"), quad_vis)

    if processed is not None:
        cv2.imwrite(str(probe_dir / "05_processed.png"), processed)

    print(f"[probe] wrote artefacts to {probe_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())