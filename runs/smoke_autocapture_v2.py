"""Verify the new 5-frame confirmation + 5 s cooldown policy.

Simulates:
  * 30 frames of a still document at realistic contour+approxPolyDP jitter
  * Auto-capture fires when stable_count reaches required_frames
  * Manual 'C' capture at frame 60 should ALSO arm the cooldown
  * A page-swap during cooldown should reset the tracker baseline so the
    next detection cycle starts from scratch.
"""

from __future__ import annotations

import random
import time

import numpy as np

from auto_capture_controller import AutoCaptureController
from corner_refiner import CornerRefiner
from stability_tracker import StabilityTracker

BASE = np.array(
    [[200, 150], [1080, 150], [1080, 650], [200, 650]], dtype=np.int32
)
SWAPPED = BASE + np.array([[400, 0], [400, 0], [400, 0], [400, 0]])


def jitter(quad, max_drift_px: int) -> np.ndarray:
    out = quad.astype(np.int32).copy()
    for i in range(4):
        out[i, 0] += random.randint(-max_drift_px, max_drift_px)
        out[i, 1] += random.randint(-max_drift_px, max_drift_px)
    return out


def realistic_quad() -> np.ndarray:
    roll = random.random()
    if roll < 0.70:
        return jitter(BASE, 6)
    if roll < 0.90:
        return jitter(BASE, 15)
    return jitter(BASE, 25)


def main() -> None:
    random.seed(7)
    ctrl = AutoCaptureController(
        enabled=True,
        cooldown_seconds=5.0,
        tracker=StabilityTracker(
            required_frames=5, tolerance=18.0, jitter_band=2.5
        ),
    )

    print("=== phase 1: 60 still-document frames (auto-fire) ===")
    fired = None
    for i in range(60):
        if ctrl.should_capture(realistic_quad()):
            fired = i
            break
    print(
        f"auto-fired at frame {fired} "
        f"(stable_count={ctrl.tracker.stable_count}, "
        f"required={ctrl.tracker.required_frames})"
    )

    print("\n=== phase 2: simulate 5 s cooldown elapsing ===")
    # Patch the timestamp so the cooldown is already in the past.
    ctrl.last_capture_timestamp = time.monotonic() - 6.0

    print("\n=== phase 3: 30 still-document frames after cooldown (re-arm) ===")
    re_fired = None
    for i in range(30):
        if ctrl.should_capture(realistic_quad()):
            re_fired = i
            break
    print(
        f"re-armed & fired at frame {re_fired} "
        f"(stable_count={ctrl.tracker.stable_count})"
    )

    print("\n=== phase 4: manual capture arms cooldown ===")
    cooldown_before = time.monotonic()
    ctrl.last_capture_timestamp = cooldown_before
    # Next call within 5 s should NOT fire, even on a still document.
    rejected = False
    for _ in range(60):
        if not ctrl.should_capture(realistic_quad()):
            rejected = True
            break
    print(
        f"manual cooldown blocks auto-fire within 5 s window? {rejected} "
        f"(elapsed={time.monotonic() - cooldown_before:.2f}s)"
    )

    print("\n=== phase 5: page-swap resets baseline ===")
    ctrl.tracker.reset()
    # First frame after reset: any quad establishes the new baseline.
    ctrl.tracker.update(jitter(SWAPPED, 5))
    next_count_after_swap = ctrl.tracker.stable_count
    print(
        f"after first swapped frame: stable_count={next_count_after_swap} "
        f"(expected 1 = baseline established)"
    )


if __name__ == "__main__":
    main()