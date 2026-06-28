"""Trace the user's 3-step spec explicitly:

  1. Original Image -> Grey Scale -> apply Edge detector
  2. Contours -> take the biggest contours -> warp perspective (to get desired doc)
  3. Colored -> scanned saving
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from document_processor import DocumentProcessor

# Pick the most recent raw frame
RAW_DIR = ROOT / "captures" / "raw"
raws = sorted(RAW_DIR.glob("raw_*.jpg"))
if not raws:
    raise SystemExit("no raw frames found")
src = raws[-1]
print(f"[input]  {src}  ({src.stat().st_size} bytes)")

img_bgr = cv2.imread(str(src))
H, W = img_bgr.shape[:2]
print(f"[input]  shape={img_bgr.shape}  dtype={img_bgr.dtype}")

# ---------- STEP 1: Original -> Grey -> Edge detector ----------
print("\n=== STEP 1: grayscale -> Canny ===")
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f"[gray]   shape={gray.shape}  mean={gray.mean():.1f}  std={gray.std():.1f}  "
      f"min={gray.min()}  max={gray.max()}")

blur = cv2.GaussianBlur(gray, (5, 5), 0)
print(f"[blur]   shape={blur.shape}  mean={blur.mean():.1f}")

# Reproduce processor's Canny exactly
v = float(np.median(blur))
lo = max(30, 0.5 * v)
hi = min(255, max(lo + 40, 1.5 * v))
print(f"[canny]  median={v:.1f}  lo={lo:.1f}  hi={hi:.1f}")
edges = cv2.Canny(blur, int(lo), int(hi))
edge_ratio = float(edges.sum() / 255) / edges.size
print(f"[canny]  edge_pixel_ratio={edge_ratio:.4f}")

# dilate so small gaps close into a contour
closed = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
closed_ratio = float(closed.sum() / 255) / closed.size
print(f"[closed] pixel_ratio={closed_ratio:.4f}")

# ---------- STEP 2: Contours -> biggest -> warp ----------
print("\n=== STEP 2: contours -> quad -> warp ===")
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(f"[contour]  total={len(contours)}")

frame_area = H * W
for i, c in enumerate(contours[:5]):
    area = cv2.contourArea(c)
    pct = area / frame_area * 100
    x, y, w, h = cv2.boundingRect(c)
    print(f"  #{i}: area={area:.0f}  ({pct:.2f}% of frame)  bbox=({x},{y},{w}x{h})  "
          f"pts={len(c)}")

if not contours:
    print("[contour]  NO CONTOURS -> cannot warp")
    sys.exit(0)

biggest = contours[0]
print(f"\n[biggest] area={cv2.contourArea(biggest):.0f}  pts={len(biggest)}")

# ConvexHull first - collapses noisy wiggles into the doc's true outer boundary
try:
    hull = cv2.convexHull(biggest)
    print(f"[hull]    pts={len(hull)}  area={cv2.contourArea(hull):.0f}")
except cv2.error:
    hull = biggest
    print(f"[hull]    FAILED, falling back to raw contour")

# Try to find a 4-corner quad with progressive epsilon on the HULL
quad = None
hull_peri = cv2.arcLength(hull, True)
for eps_factor in (0.02, 0.04, 0.06, 0.08, 0.12, 0.18):
    eps = eps_factor * hull_peri
    approx = cv2.approxPolyDP(hull, eps, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = approx.reshape(4, 2)
        print(f"[quad]    FOUND at eps_factor={eps_factor}  points=4  convex=True")
        break
    print(f"[quad]    eps={eps_factor}: got {len(approx)} pts, convex={cv2.isContourConvex(approx)}")

if quad is None:
    print("[quad]    NO QUAD from polyDP -> trying minAreaRect")
    rect = cv2.minAreaRect(biggest)
    box = cv2.boxPoints(rect)
    quad = box.astype(np.float32)
    print(f"[quad]    minAreaRect fallback: angle={rect[2]:.1f}  size=({rect[1][0]:.0f}x{rect[1][1]:.0f})")

# Order TL, TR, BR, BL
s = quad.sum(axis=1)
diff = np.diff(quad, axis=1).ravel()
ordered = np.zeros((4, 2), dtype=np.float32)
ordered[0] = quad[np.argmin(s)]
ordered[2] = quad[np.argmax(s)]
ordered[1] = quad[np.argmin(diff)]
ordered[3] = quad[np.argmax(diff)]
print(f"[quad]    ordered corners:")
for label, p in zip("TL TR BR BL".split(), ordered):
    print(f"  {label}: ({p[0]:.0f}, {p[1]:.0f})")

# Compute warp destination based on the quad's edge lengths
tl, tr, br, bl = ordered
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
W_out = int(max(widthA, widthB))
H_out = int(max(heightA, heightB))
print(f"[warp]    output size={W_out}x{H_out}")

dst = np.array([[0, 0], [W_out - 1, 0], [W_out - 1, H_out - 1], [0, H_out - 1]],
               dtype=np.float32)
M = cv2.getPerspectiveTransform(ordered, dst)
warped = cv2.warpPerspective(img_bgr, M, (W_out, H_out))
print(f"[warp]    warped.shape={warped.shape}")

# ---------- STEP 3: Colored -> scanned saving ----------
print("\n=== STEP 3: coloured save ===")
out = ROOT / "captures" / "scanned" / f"probe_{src.stem}.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
ok = cv2.imwrite(str(out), warped)
print(f"[save]    {out}  ok={ok}  bytes={out.stat().st_size if ok else 0}")

# Also write the quad-overlay onto the raw frame so we can SEE what was detected
overlay = img_bgr.copy()
cv2.polylines(overlay, [ordered.astype(int)], True, (0, 255, 0), 4)
for label, p in zip("TL TR BR BL".split(), ordered):
    cv2.circle(overlay, tuple(p.astype(int)), 12, (0, 0, 255), -1)
    cv2.putText(overlay, label, tuple((p + np.array([10, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
overlay_path = ROOT / "runs" / f"probe_{src.stem}_overlay.jpg"
cv2.imwrite(str(overlay_path), overlay)
print(f"[save]    {overlay_path}")

# Final summary
cropped = warped.shape[:2] != (H, W)
print("\n=== SUMMARY ===")
print(f"  raw          : {W}x{H}")
print(f"  edge coverage: {edge_ratio*100:.1f}%")
print(f"  biggest ctr  : {cv2.contourArea(biggest)/frame_area*100:.1f}% of frame, {len(biggest)} pts")
print(f"  quad found   : {quad is not None}")
print(f"  warp cropped : {cropped}")
print(f"  output file  : {out}")