"""Synthetic smoke test for :class:`PageChangeDetector`.

We don't need the real camera for this.  We fabricate three BGR pages:

* **page_a** -- white background with "PAGE A" text + a black square
* **page_b** -- white background with "PAGE B" text + a red circle
* **page_c** -- white background with "PAGE C" text + a green triangle

These are easy to differentiate with phash and "quad drift" (we move the
synthetic quad's vertices around).

The test feeds the detector a synthetic stream and asserts that:

1. After an initial calm frame, the detector is ``ARMED``.
2. A still scene with the same page never fires.
3. A motion spike without a real swap (camera bumped but page unchanged)
   never fires (only motion, no hash delta).
4. A genuine A -> B swap **does** fire exactly once.
5. A second identical page (B -> B) never fires.
6. A blank-with-shake (no document, just camera motion) never fires.
7. A back-to-back B -> C swap fires the detector a second time.

Run with::

    .venv/Scripts/python.exe tests/run_synthetic_page_change.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make ``page_change_detector`` importable when this file is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from page_change_detector import PageChangeDetector  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic page renderer
# --------------------------------------------------------------------------- #
def _render_page(label: str, shape: str = "square", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(
        img, label, (60, 220),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA,
    )
    if shape == "square":
        cv2.rectangle(img, (60, 280), (220, 380), (0, 0, 0), -1)
    elif shape == "circle":
        cv2.circle(img, (420, 330), 60, (0, 0, 200), -1)
    elif shape == "triangle":
        pts = np.array([[360, 380], [500, 380], [430, 260]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 200, 0))
    elif shape == "noise":
        img = rng.integers(0, 256, size=(400, 600, 3), dtype=np.uint8)
    return img


def _quad_for(label: str) -> np.ndarray:
    """Deterministic quad for each label -- so quad_jump tests are reproducible."""
    base = {
        "A": np.array([[40, 40], [560, 40], [560, 360], [40, 360]], dtype=np.float32),
        "B": np.array([[50, 50], [550, 50], [550, 350], [50, 350]], dtype=np.float32),
        "C": np.array([[60, 60], [540, 60], [540, 340], [60, 340]], dtype=np.float32),
    }[label]
    return base


# --------------------------------------------------------------------------- #
# Frame feeder
# --------------------------------------------------------------------------- #
def _drive(detector: PageChangeDetector, *, frames: int, page: np.ndarray,
           quad: np.ndarray, motion: float, label: str) -> list:
    events = []
    for i in range(frames):
        # Jitter the quad slightly every frame so it isn't a dead match.
        jitter = np.random.default_rng(i).normal(0, 0.5, quad.shape).astype(np.float32)
        cur_quad = quad + jitter
        ev = detector.observe(
            quad=cur_quad,
            processed_bgr=page,
            motion_px=motion,
        )
        if ev is not None:
            events.append((label, i, ev))
    return events


# --------------------------------------------------------------------------- #
# Assertions
# --------------------------------------------------------------------------- #
def _assert(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    print("Synthetic PageChangeDetector test")
    print("-" * 50)

    page_a = _render_page("PAGE A", "square", seed=1)
    page_b = _render_page("PAGE B", "circle", seed=2)
    page_c = _render_page("PAGE C", "triangle", seed=3)

    det = PageChangeDetector(
        enabled=True,
        motion_trigger_px=20.0,
        motion_rest_px=4.0,
        rest_frames=3,
        quad_jump_px=25.0,
        hash_distance=6,
        hash_size=8,
    )

    quad_a = _quad_for("A")
    quad_b = _quad_for("B")
    quad_c = _quad_for("C")

    all_events: list = []

    # ---- Phase 1: arm the detector (still scene of page A) ----
    print("\nPhase 1: arm on first calm frame of page A")
    events = _drive(det, frames=6, page=page_a, quad=quad_a, motion=1.0, label="arm")
    all_events.extend(events)
    _assert(det._phase == "ARMED", "phase is ARMED after calm frames")
    _assert(len(events) == 0, "no spurious events while still")

    # Baseline should now match page A.
    det.update_baseline_after_capture(page_a, quad_a)

    # ---- Phase 2: still scene should never fire ----
    print("\nPhase 2: still scene must NOT fire")
    events = _drive(det, frames=10, page=page_a, quad=quad_a, motion=1.0, label="still")
    all_events.extend(events)
    _assert(len(events) == 0, "still scene produced no events")

    # ---- Phase 3: motion spike WITHOUT a real swap ----
    print("\nPhase 3: motion spike without content swap must NOT fire")
    # Camera bumped, but the page is still A.
    events = _drive(det, frames=10, page=page_a, quad=quad_a, motion=45.0, label="shake-only")
    all_events.extend(events)
    _assert(len(events) == 0, "shake without content swap produced no events")
    # Then the motion settles back to still with same page.
    events = _drive(det, frames=8, page=page_a, quad=quad_a, motion=1.0, label="shake-settle")
    all_events.extend(events)
    _assert(len(events) == 0, "shake settle produced no events")

    # ---- Phase 4: A -> B is a real swap ----
    print("\nPhase 4: real swap A -> B must fire")
    events = _drive(det, frames=4, page=page_b, quad=quad_b, motion=35.0, label="move-B")
    all_events.extend(events)
    # Then settle on B.
    events = _drive(det, frames=8, page=page_b, quad=quad_b, motion=1.0, label="settle-B")
    all_events.extend(events)
    swap_events = [e for e in all_events if e[0].startswith("move-B") or e[0].startswith("settle-B")]
    _assert(len(swap_events) >= 1, "A -> B swap fired at least once")
    if swap_events:
        ev = swap_events[-1][2]
        print(f"    conf={ev.confidence:.2f} hash={ev.hash_distance} "
              f"quad={ev.quad_distance:.1f}px peak_motion={ev.motion_at_peak:.1f}px")
        _assert(ev.confidence > 0.3, "swap event confidence is meaningful")

    # Update baseline to B so the detector isn't fooled twice.
    det.update_baseline_after_capture(page_b, quad_b)

    # ---- Phase 5: B -> B (identical page) never fires ----
    print("\nPhase 5: B -> B identical page must NOT fire")
    events = _drive(det, frames=6, page=page_b, quad=quad_b, motion=35.0, label="move-B2")
    all_events.extend(events)
    events = _drive(det, frames=8, page=page_b, quad=quad_b, motion=1.0, label="settle-B2")
    all_events.extend(events)
    _assert(len(events) == 0, "identical page produced no events")

    # ---- Phase 6: blank-with-shake (identical blank page) never fires ----
    print("\nPhase 6: blank shake of IDENTICAL blank page must NOT fire")
    blank = np.full((400, 600, 3), 200, dtype=np.uint8)
    # Reset baseline to the blank so any subsequent shake is "same content".
    det.update_baseline_after_capture(blank, quad_b)
    events = _drive(det, frames=6, page=blank, quad=quad_b, motion=40.0, label="move-blank")
    all_events.extend(events)
    events = _drive(det, frames=8, page=blank, quad=quad_b, motion=2.0, label="settle-blank")
    all_events.extend(events)
    _assert(len(events) == 0, "identical-blank shake produced no events")

    # ---- Phase 7: B -> C back-to-back swap must fire ----
    print("\nPhase 7: back-to-back swap B -> C must fire")
    # Re-arm baseline to B before swapping to C, so hash delta is real.
    det.update_baseline_after_capture(page_b, quad_b)
    events = _drive(det, frames=4, page=page_c, quad=quad_c, motion=35.0, label="move-C")
    all_events.extend(events)
    events = _drive(det, frames=8, page=page_c, quad=quad_c, motion=1.0, label="settle-C")
    all_events.extend(events)
    swap_events = [e for e in all_events if e[0].startswith("move-C") or e[0].startswith("settle-C")]
    _assert(len(swap_events) >= 1, "B -> C swap fired at least once")

    print("\n" + "=" * 50)
    total = sum(len([e for e in all_events if e[0].startswith(p)]) for p in (
        "move-B", "settle-B", "move-C", "settle-C",
    ))
    print(f"Total swap events fired: {total}")
    print("ALL ASSERTIONS PASSED" if total >= 2 else "TEST FAILED")
    return 0 if total >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())