"""Track document corner stability across successive frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from corner_refiner import CornerRefiner, Point, Quad


@dataclass
class StabilityTracker:
    """Track whether a detected document quad is stable across frames."""

    required_frames: int = 8
    tolerance: float = 20.0
    stable_count: int = 0
    last_quad: Optional[Quad] = None
    corner_refiner: CornerRefiner = field(default_factory=CornerRefiner)

    def update(self, quad: Optional[Quad]) -> bool:
        """Update the tracker with the newest quad and return whether it is stable."""
        if quad is None:
            self.reset()
            return False

        if self.last_quad is None:
            self.last_quad = quad
            self.stable_count = 1
            return False

        distance = self.corner_refiner.distance(self.last_quad, quad)
        if distance <= self.tolerance:
            self.stable_count += 1
        else:
            self.last_quad = quad
            self.stable_count = 1

        return self.is_stable

    @property
    def is_stable(self) -> bool:
        """Return True when the same quad has been observed for enough frames."""
        return self.stable_count >= self.required_frames

    def reset(self) -> None:
        """Reset the stability tracker state."""
        self.stable_count = 0
        self.last_quad = None
