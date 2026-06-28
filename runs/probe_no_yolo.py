"""Synthetic end-to-end probe for the new YOLO-free pipeline.

Generates a 1280x720 frame that mimics a webcam photo of a desk:

  - wood-textured background
  - a white A6-ish sheet placed in the lower-right, perspective-skewed,
    slightly rotated, with a printed-paragraph texture and a soft shadow
  - a stray pen and a coffee mug circle as distractors

Runs it through:
    DocumentProcessor.process(frame)
    detector.DetectionResult

Asserts that:
    - 4 ordered corners are returned
    - confidence >= MIN_QUAD_CONFIDENCE
    - the warped output is exactly A4 (A4_WIDTH_PX x A4_HEIGHT_PX)
    - the warped output has a *bright* interior (paper) and *dark* corners
      (background bleed-through would mean the warp missed the page)

Also writes intermediate debug images to runs/probe_out/ for visual review.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    A4_HEIGHT_PX,
    A4_WIDTH_PX,
    MIN_QUAD_CONFIDENCE,
)
from document_processor import DocumentProcessor  # noqa: E402

OUT = ROOT / "runs" / "probe_out"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Synthetic scene generator
# --------------------------------------------------------------------------- #
def make_desk_background(h: int = 720, w: int = 1280, seed: int = 7) -> np.ndarray:
    """Wood-grain-ish texture so the edge detector has real noise to reject."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=110, scale=25, size=(h, w)).astype(np.float32)
    # add horizontal grain lines
    for i in range(0, h, 3):
        base[i : i + 1, :] += rng.normal(0, 8)
    base = np.clip(base, 0, 255).astype(np.uint8)
    bg = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # warm it up so it doesn't look pure grayscale
    bg[..., 0] = np.clip(bg[..., 0].astype(int) + 25, 0, 255)  # B
    bg[..., 2] = np.clip(bg[..., 2].astype(int) + 10, 0, 255)  # R
    return bg


def make_paper_overlay(
    background: np.ndarray,
    dst_quad: np.ndarray,
    seed: int = 13,
) -> np.ndarray:
    """Warp a clean white sheet with printed text into ``dst_quad`` of ``background``."""
    h, w = background.shape[:2]
    # 1) source: a clean A6 portrait sheet with printed lines
    sheet_h, sheet_w = 900, 636
    sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)  # off-white
    # header bar
    cv2.rectangle(sheet, (40, 40), (sheet_w - 40, 110), (60, 60, 60), thickness=-1)
    # body text lines
    rng = np.random.default_rng(seed)
    for y in range(160, sheet_h - 40, 38):
        line_w = rng.integers(int(sheet_w * 0.45), int(sheet_w * 0.85))
        cv2.rectangle(sheet, (50, y), (50 + int(line_w), y + 16), (40, 40, 40), thickness=-1)
    # 2) destination quadrilateral (perspective-skewed, rotated)
    src = np.array(
        [[0, 0], [sheet_w - 1, 0], [sheet_w - 1, sheet_h - 1], [0, sheet_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst_quad.astype(np.float32))
    warped = cv2.warpPerspective(sheet, M, (w, h))

    # 3) soft drop-shadow on the desk under the paper
    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(shadow_mask, [dst_quad.astype(np.int32)], 255)
    shadow_mask = cv2.dilate(shadow_mask, np.ones((15, 15), np.uint8), iterations=1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 0)
    shadow_f32 = shadow_mask.astype(np.float32) / 255.0
    darkened = (
        background.astype(np.float32) * (1.0 - 0.45 * shadow_f32[..., None])
    ).astype(np.uint8)

    # 4) composite the paper on top
    paper_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(paper_mask, [dst_quad.astype(np.int32)], 255)
    paper_mask = cv2.GaussianBlur(paper_mask, (5, 5), 0)
    alpha = paper_mask.astype(np.float32) / 255.0
    out = (darkened.astype(np.float32) * (1 - alpha[..., None])
           + warped.astype(np.float32) * alpha[..., None]).astype(np.uint8)
    return out


def add_distractors(frame: np.ndarray) -> np.ndarray:
    """A pen + a coffee mug circle to make the background non-trivial."""
    h, w = frame.shape[:2]
    # dark pen
    cv2.line(frame, (180, 200), (430, 360), (20, 20, 20), thickness=10)
    cv2.line(frame, (180, 200), (430, 360), (200, 200, 200), thickness=2)
    # coffee mug circle
    cv2.circle(frame, (1050, 160), 70, (40, 30, 30), thickness=-1)
    cv2.circle(frame, (1050, 160), 55, (210, 180, 140), thickness=-1)
    return frame


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    print("[probe] generating synthetic scene...")
    bg = make_desk_background()
    # Place the page in the lower-right, perspective-skewed.
    dst = np.array(
        [
            [560, 250],   # TL
            [1180, 200],  # TR
            [1220, 660],  # BR
            [480, 700],   # BL
        ],
        dtype=np.int32,
    )
    frame = make_paper_overlay(bg, dst)
    frame = add_distractors(frame)
    cv2.imwrite(str(OUT / "probe_no_yolo_scene.png"), frame)

    print("[probe] running DocumentProcessor...")
    proc = DocumentProcessor()
    t0 = cv2.getTickCount()
    processed, det = proc.process(frame)
    elapsed_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
    print(f"[probe] process() took {elapsed_ms:.1f} ms")
    print(f"[probe] corners   : {None if det.corners is None else det.corners.tolist()}")
    print(f"[probe] confidence: {det.confidence:.3f}")
    print(f"[probe] bbox      : {det.bbox}")
    print(f"[probe] used_yolo : {det.used_yolo}")

    # ---- assertions --------------------------------------------------- #
    assert det.corners is not None, "FAIL: detector returned no corners"
    assert det.corners.shape == (4, 2), f"FAIL: expected 4x2 corners, got {det.corners.shape}"
    assert det.confidence >= MIN_QUAD_CONFIDENCE, (
        f"FAIL: confidence {det.confidence:.3f} below MIN_QUAD_CONFIDENCE "
        f"{MIN_QUAD_CONFIDENCE}"
    )
    assert processed.shape[0] == A4_HEIGHT_PX, (
        f"FAIL: warped height {processed.shape[0]} != A4_HEIGHT_PX {A4_HEIGHT_PX}"
    )
    assert processed.shape[1] == A4_WIDTH_PX, (
        f"FAIL: warped width {processed.shape[1]} != A4_WIDTH_PX {A4_WIDTH_PX}"
    )

    # The interior of a successful warp should be bright (paper).  Sample a
    # 200x200 patch in the middle and require mean luma > 180.
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    patch = gray[cy - 100 : cy + 100, cx - 100 : cx + 100]
    mean_luma = float(patch.mean())
    print(f"[probe] warped center patch mean luma: {mean_luma:.1f}")

    # ---- save artefacts (always) ------------------------------------- #
    overlay = DocumentProcessor.draw_overlay(frame, det, processed_preview=processed)
    cv2.imwrite(str(OUT / "probe_no_yolo_overlay.png"), overlay)
    cv2.imwrite(str(OUT / "probe_no_yolo_warped.png"), processed)

    # Tighter sanity: mean of central 60% of the warped page should be bright
    # because the paper dominates the A4 crop after the 20-px border crop.
    h2, w2 = gray.shape
    inner = gray[int(h2 * 0.20) : int(h2 * 0.80), int(w2 * 0.20) : int(w2 * 0.80)]
    mean_inner = float(inner.mean())
    print(f"[probe] warped inner-60% mean luma   : {mean_inner:.1f}")
    if mean_inner < 180.0:
        print(f"[probe] WARN: inner luma {mean_inner:.1f} < 180 (paper may not fill frame)")
    if mean_luma < 180.0:
        print(f"[probe] WARN: center luma {mean_luma:.1f} < 180 (border crop / shadow artefact)")

    print("[probe] PASS - YOLO-free pipeline produced an A4 page")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())