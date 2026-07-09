"""Smoke-test: StabilityTracker + jitter-band under realistic YOLO jitter."""
import random

import numpy as np

from stability_tracker import StabilityTracker
from corner_refiner import CornerRefiner

# Base quad for a 1280x720 frame, simulating a document at roughly
# (200,150) -> (1080,150) -> (1080,650) -> (200,650).
BASE = np.array(
    [[200, 150], [1080, 150], [1080, 650], [200, 650]], dtype=np.int32
)


def jitter(quad, max_drift_px):
    """Return a quad perturbed by up to ``max_drift_px`` per corner, integers."""
    out = quad.astype(np.int32).copy()
    for i in range(4):
        dx = random.randint(-max_drift_px, max_drift_px)
        dy = random.randint(-max_drift_px, max_drift_px)
        out[i, 0] += dx
        out[i, 1] += dy
    return out


random.seed(42)
tracker = StabilityTracker(required_frames=45, tolerance=18.0, jitter_band=2.5)

is_stable = False
peak_count = 0
fired = False
for frame_index in range(200):
    # 70% of frames jitter <= tolerance, 20% jitter between tol and band,
    # 10% jump > 2.5x tolerance (simulating a brief detection glitch).
    roll = random.random()
    if roll < 0.70:
        q = jitter(BASE, 6)        # well within tolerance
    elif roll < 0.90:
        q = jitter(BASE, 15)       # above tolerance but inside band (jitter)
    else:
        q = jitter(BASE, 25)       # worst-case noise (~ band limit)
        # Occasionally push real motion:
        if random.random() < 0.05:
            q = q + np.array([[120, 0], [120, 0], [120, 0], [120, 0]])

    stable_now = tracker.update(q)
    peak_count = max(peak_count, tracker.stable_count)
    if stable_now and not is_stable:
        fired = True
        print(
            f"FIRED at frame {frame_index}: stable_count={tracker.stable_count}, "
            f"required={tracker.required_frames}"
        )
        break

print(f"peak_count={peak_count}, fired={fired}")
