"""Track document corner stability across successive frames."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, TypeAlias

from corner_refiner import CornerRefiner, Point

Quad: TypeAlias = tuple[Point, Point, Point, Point]


@dataclass
class StabilityTracker:
    """Track whether a detected document quad is stable across frames."""

    required_frames: int = 8
    tolerance: float = 20.0
    stable_count: int = 0
    last_quad: Optional[Quad] = None
    stable_since: Optional[float] = None
    corner_refiner: CornerRefiner = field(default_factory=CornerRefiner)

    def update(self, quad: Optional[Quad], now: Optional[float] = None) -> bool:
        """Update the tracker with the newest quad and return whether it is stable."""
        now = time.monotonic() if now is None else now
        if quad is None:
            self.reset()
            return False

        if self.last_quad is None:
            self.last_quad = quad
            self.stable_count = 1
            self.stable_since = now
            return False

        distance = self.corner_refiner.distance(self.last_quad, quad)
        if distance <= self.tolerance:
            self.stable_count += 1
            if self.stable_since is None:
                self.stable_since = now
        else:
            self.last_quad = quad
            self.stable_count = 1
            self.stable_since = now

        return self.is_stable

    @property
    def is_stable(self) -> bool:
        """Return True when the same quad has been observed for enough frames."""
        return self.stable_count >= self.required_frames

    def is_stable_for(self, seconds: float, now: Optional[float] = None) -> bool:
        """Return True when the same quad has stayed stable for ``seconds``."""
        if not self.is_stable or self.stable_since is None:
            return False
        now = time.monotonic() if now is None else now
        return (now - self.stable_since) >= seconds

    def reset(self) -> None:
        """Reset the stability tracker state."""
        self.stable_count = 0
        self.last_quad = None
        self.stable_since = None
