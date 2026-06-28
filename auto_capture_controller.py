"""Optional auto-capture logic driven by document stability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from stability_tracker import StabilityTracker
from corner_refiner import Quad


@dataclass
class AutoCaptureController:
    """Decide when a stable document should be captured automatically."""

    enabled: bool = False
    cooldown_seconds: float = 3.0
    tracker: StabilityTracker = field(default_factory=StabilityTracker)
    last_capture_timestamp: float = 0.0

    def should_capture(self, quad: Quad) -> bool:
        """Return True when the quad has been stable long enough and cooldown passed."""
        if not self.enabled:
            self.tracker.update(quad)
            return False

        stable = self.tracker.update(quad)
        if not stable:
            return False

        now = time.monotonic()
        if now - self.last_capture_timestamp < self.cooldown_seconds:
            return False

        self.last_capture_timestamp = now
        return True
