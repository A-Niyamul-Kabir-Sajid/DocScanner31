"""Optional auto-capture logic driven by document stability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TypeAlias

from stability_tracker import StabilityTracker
from corner_refiner import Point

Quad: TypeAlias = tuple[Point, Point, Point, Point]


@dataclass
class AutoCaptureController:
    """Decide when a stable document should be captured automatically."""

    enabled: bool = False
    stable_seconds: float = 2.0
    cooldown_seconds: float = 3.0
    tracker: StabilityTracker = field(default_factory=StabilityTracker)
    last_capture_timestamp: float = 0.0

    def should_capture(self, quad: Quad, *, now: float | None = None) -> bool:
        """Return True when the quad has been stable long enough and cooldown passed."""
        import time

        now = time.monotonic() if now is None else now
        if not self.enabled:
            self.tracker.update(quad, now=now)
            return False

        stable = self.tracker.update(quad, now=now)
        if not stable:
            return False

        if not self.tracker.is_stable_for(self.stable_seconds, now=now):
            return False

        if now - self.last_capture_timestamp < self.cooldown_seconds:
            return False

        self.last_capture_timestamp = now
        return True
