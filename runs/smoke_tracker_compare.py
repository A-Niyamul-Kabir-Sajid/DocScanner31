"""Compare old vs new StabilityTracker under the same noise distribution."""
import random

import numpy as np

from stability_tracker import StabilityTracker
from corner_refiner import CornerRefiner

BASE = np.array(
    [[200, 150], [1080, 150], [1080, 650], [200, 650]], dtype=np.int32
)


def jitter(quad, max_drift_px):
    out = quad.astype(np.int32).copy()
    for i in range(4):
        dx = random.randint(-max_drift_px, max_drift_px)
        dy = random.randint(-max_drift_px, max_drift_px)
        out[i, 0] += dx
        out[i, 1] += dy
    return out


def run(label, *, required, tolerance, band=None, frames=400):
    random.seed(42)
    kwargs = {"required_frames": required, "tolerance": tolerance}
    if band is not None:
        kwargs["jitter_band"] = band
    tracker = StabilityTracker(**kwargs)
    peak = 0
    fired_frame = None
    for frame_index in range(frames):
        roll = random.random()
        if roll < 0.70:
            q = jitter(BASE, 6)
        elif roll < 0.90:
            q = jitter(BASE, 15)
        else:
            q = jitter(BASE, 25)
            if random.random() < 0.05:
                q = q + np.array([[120, 0], [120, 0], [120, 0], [120, 0]])
        if tracker.update(q):
            fired_frame = frame_index
            break
        peak = max(peak, tracker.stable_count)
    print(
        f"{label:40s} required={required:3d} tol={tolerance:5.1f} "
        f"band={band}  fired_at={fired_frame}  peak_count={peak}"
    )


run("OLD (tolerance=6, no band)", required=60, tolerance=6.0)
run("NEW (tolerance=18, no band)", required=45, tolerance=18.0)
run("NEW (tolerance=18, band=2.5)", required=45, tolerance=18.0, band=2.5)