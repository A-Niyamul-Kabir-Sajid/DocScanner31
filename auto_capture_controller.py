"""Two-state auto-capture FSM.

The controller drives ``ScanSession._maybe_auto_capture`` with a tiny
finite state machine that mirrors the user's mental model of the loop:

* **State 1 -- ``SEEKING_STABLE``** (default)

    Look for a stable quad and fire one capture as soon as the doc has
    been still for ``stable_frames`` consecutive frames.  After firing
    the controller flips itself to **State 2** so the same page can't
    double-fire while the user is still moving it out of the frame.
    A manual ``c`` key-press from State 1 also captures and arms
    State 2.

* **State 2 -- ``WAITING_FOR_CHANGE``**

    After every type of capture (auto or manual) the controller sits
    here watching the live quad against ``last_captured_quad`` (the
    four-point contour that was saved the moment of the capture).
    A "no-match" timer accumulates continuous time during which the
    live quad is *not* similar to ``last_captured_quad``.  The moment
    the live quad *is* similar, the timer resets to zero.  When the
    no-match timer reaches ``s2_no_match_timeout_s`` (3 s by default)
    the controller flips itself back to State 1, ready for the next
    auto-capture.

    A manual ``c`` key-press from State 2 captures again, overwrites
    ``last_captured_quad`` with the freshly captured corners, and
    resets the no-match timer back to zero -- the FSM stays in State 2.

There is **no fixed cooldown**.  The 2 s no-match timer is the only
gate between State 2 and the next auto-capture.

The class also exposes the legacy ``should_capture(quad)`` entry point
for unit tests so the synthetic smoke test keeps working without
touching the new observe/consume_change API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from corner_refiner import Quad
from stability_tracker import StabilityTracker


# --------------------------------------------------------------------------- #
# State constants -- exposed as public enum-like strings so the HUD, the
# tests and the controller itself can speak the same vocabulary.
# --------------------------------------------------------------------------- #
S1_SEEKING_STABLE = "S1_SEEKING_STABLE"
S2_WAITING_FOR_CHANGE = "S2_WAITING_FOR_CHANGE"


@dataclass
class AutoCaptureController:
    """Decide when a stable document should be captured automatically."""

    enabled: bool = False
    # Maximum corner drift (pixels) tolerated when comparing the live
    # quad against ``last_captured_quad`` in State 2.  Mirrors the
    # ``StabilityTracker`` tolerance so a "similar" contour really
    # means the user is still showing the same page.
    tolerance: float = 18.0
    # Continuous seconds of "no match" required to flip State 2 -> 1.
    s2_no_match_timeout_s: float = 2.0
    motion_trigger_px: float = 25.0
    tracker: StabilityTracker = field(default_factory=StabilityTracker)

    # FSM state -- default to State 1 ("look for a stable page").
    state: str = field(default=S1_SEEKING_STABLE, init=False)
    last_capture_timestamp: float = 0.0
    # Quad observed at the moment of the last successful capture.  The
    # "frame changed" detector in State 2 measures drift away from this
    # baseline so a small swipe is treated as motion, not a new page.
    last_captured_quad: Optional[Quad] = field(default=None, init=False)
    # Accumulated seconds the live quad has been *not similar* to
    # ``last_captured_quad``.  Reset to 0 the moment a similar quad
    # is seen; reaches ``s2_no_match_timeout_s`` -> flip to State 1.
    s2_no_match_seconds: float = field(default=0.0, init=False)
    # Last-frame cache of "did the live quad match the baseline?".
    # Used by the HUD phase label (S2_match vs S2_waiting).
    _last_similar_seen: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Public API -- new two-state FSM
    # ------------------------------------------------------------------ #
    def observe(
        self,
        quad: Optional[Quad],
        *,
        motion_px: float = 0.0,
        page_change_event=None,
    ) -> "ObserveResult":
        """Feed one frame's signals in and return a result describing what to do.

        ``quad`` is the detected document quad (or ``None`` if nothing is
        in frame).  ``motion_px`` is the per-frame MAD used by the page
        change detector.  ``page_change_event`` is whatever the
        :class:`PageChangeDetector` produced for this tick (or ``None``).

        The returned :class:`ObserveResult` carries:

        * ``should_fire`` -- True when a capture must happen *now*.
        * ``fire_reason`` -- "stable" when State 1 reached the streak.
        * ``phase``       -- the HUD-friendly phase string:
                              "off" / "S1_seeking" / "S2_waiting"
                              / "S2_cooling"
        * ``progress``    -- (stable_count, required_frames) for the HUD.
        * ``change_detected`` -- True when a State 2 -> State 1 transition
                                  was triggered by this frame.
        """
        result = ObserveResult(
            should_fire=False,
            fire_reason="",
            phase=self.phase_label(),
            progress=(self.tracker.stable_count, self.tracker.required_frames),
            change_detected=False,
        )

        if not self.enabled:
            result.phase = "off"
            return result

        # State 2: do NOT fire; only watch for "change of frame".
        if self.state == S2_WAITING_FOR_CHANGE:
            return self._observe_state2(quad, motion_px, page_change_event, result)

        # State 1: seek a stable quad.
        return self._observe_state1(quad, motion_px, result)

    # ------------------------------------------------------------------ #
    # Public API -- legacy hook for the existing synthetic test
    # ------------------------------------------------------------------ #
    def should_capture(self, quad: Quad) -> bool:
        """Return True when the quad has been stable long enough.

        Kept as a backwards-compatible wrapper so unit tests don't have
        to migrate to ``observe()``.  The live ``_maybe_auto_capture``
        uses :meth:`observe` instead so it can react to State 2 events.
        No cooldown gate anymore.
        """
        if not self.enabled:
            self.tracker.update(quad)
            return False

        # No real change signals in this legacy path, so treat every
        # frame as State 1.
        self.state = S1_SEEKING_STABLE
        self.tracker.update(quad)
        stable = self.tracker.is_stable
        if not stable:
            return False

        self.last_capture_timestamp = time.monotonic()
        return True

    # ------------------------------------------------------------------ #
    # Convenience: explicitly consume a "change" (e.g. operator keystroke
    # while in State 2).  Tests + non-FSM callers can flip State 2 -> 1.
    # ------------------------------------------------------------------ #
    def consume_change(self) -> bool:
        """Manually transition State 2 -> State 1.  Returns True if state changed."""
        return self._flip_to_state1()

    # ------------------------------------------------------------------ #
    # Notify the FSM that an OUT-OF-BAND capture just happened (the user
    # pressed ``c`` from either state).  The four-point contour that was
    # saved is recorded as the new baseline; the no-match timer is
    # reset; the FSM is placed in State 2.
    # ------------------------------------------------------------------ #
    def register_capture(self, quad: Optional[Quad] = None) -> None:
        """Record a manual capture and force the FSM into State 2.

        Parameters
        ----------
        quad:
            The four-point contour of the page that was just saved.
            ``None`` is allowed (the controller simply clears its
            baseline; the next live quad that arrives becomes the
            effective baseline for the no-match comparison).
        """
        self.last_capture_timestamp = time.monotonic()
        self.last_captured_quad = quad
        self.state = S2_WAITING_FOR_CHANGE
        self.s2_no_match_seconds = 0.0
        self._last_similar_seen = True
        self._last_state2_tick_t = time.monotonic()

    # ------------------------------------------------------------------ #
    # Properties used by the HUD
    # ------------------------------------------------------------------ #
    @property
    def no_match_remaining(self) -> float:
        """Seconds still needed in State 2 before the no-match timer fires."""
        if self.state != S2_WAITING_FOR_CHANGE:
            return 0.0
        return max(0.0, self.s2_no_match_timeout_s - self.s2_no_match_seconds)

    @property
    def current_quad_matches(self) -> bool:
        """True when a live quad was just observed similar to ``last_captured_quad``.

        Used by the HUD to distinguish "user is showing the same page"
        from "live quad differs from the last capture".
        """
        return self._last_similar_seen

    def phase_label(self) -> str:
        """Human-readable phase string for the HUD / last_message."""
        if not self.enabled:
            return "off"
        if self.state == S2_WAITING_FOR_CHANGE:
            if self._last_similar_seen:
                return "S2_match"
            return "S2_waiting"
        return "S1_seeking"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _observe_state1(
        self,
        quad: Optional[Quad],
        motion_px: float,
        result: "ObserveResult",
    ) -> "ObserveResult":
        """State 1 logic -- build the streak and fire when ready."""
        if quad is None:
            # Doc left the frame during State 1: nothing to do.  Reset
            # the streak so we don't claim stability from a vanished doc.
            self.tracker.reset()
            result.progress = (0, self.tracker.required_frames)
            result.phase = "S1_seeking"
            return result

        self.tracker.update(quad)
        result.progress = (self.tracker.stable_count, self.tracker.required_frames)

        if not self.tracker.is_stable:
            result.phase = "S1_seeking"
            return result

        # Stable -- FIRE.  No cooldown gate; the next page goes through
        # State 2 -> 1 via the 2 s no-match timer.
        now = time.monotonic()
        self.last_capture_timestamp = now
        self.last_captured_quad = quad
        self.state = S2_WAITING_FOR_CHANGE
        self.s2_no_match_seconds = 0.0
        self._last_similar_seen = True

        result.should_fire = True
        result.fire_reason = "stable"
        result.phase = self.phase_label()
        return result

    def _observe_state2(
        self,
        quad: Optional[Quad],
        motion_px: float,
        page_change_event,
        result: "ObserveResult",
        now: Optional[float] = None,
    ) -> "ObserveResult":
        """State 2 logic -- accumulate "no-match" time against the captured contour.

        The tick is *similar* when the live quad is non-None and within
        ``tolerance`` of ``last_captured_quad``.  In that case the
        no-match timer is reset to 0.  Anything else (None, motion
        spike, drifted quad, page-change event) increments the timer
        by however much wall-clock time has elapsed since the last
        tick.  When the timer reaches ``s2_no_match_timeout_s`` the FSM
        flips to State 1.
        """
        if now is None:
            now = time.monotonic()

        # 1) Dedicated page-change detector fired -> strongest signal;
        #    cancel the timer so the flip is instant.
        if page_change_event is not None:
            if self._flip_to_state1():
                result.change_detected = True
            return result

        # 2) Decide whether THIS tick is a "match" against the captured
        #    four-point contour.  Three ways to be a non-match:
        #      * no live quad at all,
        #      * motion spike,
        #      * quad drift beyond tolerance.
        similar = False
        if quad is not None and self.last_captured_quad is not None:
            try:
                drift = self.tracker.corner_refiner.distance(
                    self.last_captured_quad, quad
                )
            except Exception:  # pragma: no cover - defensive
                drift = 0.0
            if drift <= self.tolerance and motion_px <= self.motion_trigger_px:
                similar = True

        if similar:
            # The user is still showing the same page; the no-match
            # streak must not grow.  Reset to zero so future ticks
            # start counting from now.
            self.s2_no_match_seconds = 0.0
            self._last_similar_seen = True
            result.phase = self.phase_label()
            return result

        # Non-match: add the elapsed wall-clock since the previous
        # tick.  Use the controller's own per-tick delta when
        # available; otherwise assume one LIVE-tick (33 ms).
        prev_t = getattr(self, "_last_state2_tick_t", now)
        delta = max(0.0, now - prev_t)
        # Cap the delta so a paused process doesn't dump minutes into
        # the timer on the very next tick.
        if delta > 0.5:
            delta = 0.5
        self.s2_no_match_seconds += delta
        self._last_similar_seen = False
        self._last_state2_tick_t = now

        if self.s2_no_match_seconds >= self.s2_no_match_timeout_s:
            if self._flip_to_state1():
                result.change_detected = True

        result.phase = self.phase_label()
        return result

    def _flip_to_state1(self) -> bool:
        """Move State 2 -> State 1 unconditionally.

        There is no longer a cooldown gate.  Called either by the
        2 s no-match timer, a page-change event, or a manual
        ``consume_change()``.
        """
        if self.state != S2_WAITING_FOR_CHANGE:
            return False
        self.state = S1_SEEKING_STABLE
        self.tracker.reset()
        self.last_captured_quad = None
        self.s2_no_match_seconds = 0.0
        self._last_similar_seen = False
        return True


@dataclass
class ObserveResult:
    """Outcome of one ``AutoCaptureController.observe(...)`` call."""

    should_fire: bool = False
    fire_reason: str = ""
    phase: str = "off"
    progress: tuple = (0, 0)
    change_detected: bool = False
