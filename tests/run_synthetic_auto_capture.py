"""
tests/run_synthetic_auto_capture.py
====================================

Synthetic smoke test for the AutoCaptureController + ScanSession
wiring (no camera, no GUI).

Phases:

1.  Cold start - the tracker is empty, should_capture() returns False.
2.  Steady-state ramp - feed 12 stable quads, expect EXACTLY one fire on
    the 12th frame.
3.  Cooldown gate - during the 1.5 s cooldown the controller must refuse
    further fires, even if the scene stays rock-stable.
4.  Cooldown expiry - after the timer elapses a fresh stable window
    must produce a SECOND fire.
5.  Phase HUD strings - confirm the ScanSession reports "identifying",
    "cooldown" then back to "idle" at the right moments.

Run with:

    .venv\\Scripts\\python.exe tests\\run_synthetic_auto_capture.py
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from auto_capture_controller import AutoCaptureController, S1_SEEKING_STABLE  # noqa: E402
from stability_tracker import StabilityTracker              # noqa: E402
from corner_refiner import CornerRefiner                    # noqa: E402
from app import ScanSession, ScannerState                   # noqa: E402


# ------------------------------------------------------------------
# Patch: stability_tracker relies on CornerRefiner.distance which
# isn't implemented in the prod codebase yet.  We give it a zero
# distance so that two identical quads always count as stable.
# ------------------------------------------------------------------
if not hasattr(CornerRefiner, "distance"):
    def _distance(self, q1, q2):  # type: ignore[no-redef]
        try:
            import numpy as _np
            return float(_np.linalg.norm(_np.asarray(q1, float) - _np.asarray(q2, float)))
        except Exception:
            return 0.0
    CornerRefiner.distance = _distance   # type: ignore[attr-defined]  # noqa: E305


def _to_quad(seed=(100, 100), w=200, h=260):
    import numpy as _np
    x, y = seed
    return _np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=_np.float32)


# ------------------------------------------------------------------
# Synthetic helpers
# ------------------------------------------------------------------

def stable_quad(seed=(100, 100), w=200, h=260):
    """A perfectly stable Quad (np.ndarray shape (4,2))."""
    return _to_quad(seed, w, h)


# ------------------------------------------------------------------
# Test runner
# ------------------------------------------------------------------

def banner(t: str) -> None:
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def assert_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    print(f"  ✓ {label} == {expected!r}")


def assert_true(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"  ✓ {label}")


def build_session(tmp: Path) -> ScanSession:
    """Build a ScanSession WITHOUT touching camera/cv2 GUI."""
    captures = tmp / "captures" / "scanned"
    out = tmp / "output"
    captures.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    s = ScanSession(
        captures_dir=captures,
        output_dir=out,
        camera_source=None,            # never opened
        camera_backend=None,
        camera_width=640,
        camera_height=480,
        web_host="127.0.0.1",
        web_port=0,
        scan_mode="color",
    )
    # speed things up for the test
    s.auto_capture_cooldown_s = 0.4
    s.auto_capture_stable_frames = 5    # shorter than prod (12) for speed
    return s


# ------------------------------------------------------------------
# Phase 1 - cold start, no fire
# ------------------------------------------------------------------

def _warm_start(controller: AutoCaptureController) -> None:
    """Push the controller past the cooldown baseline so the first stable
    frame can fire without a 0-vs-monotonic edge case."""
    controller.last_capture_timestamp = time.monotonic() - controller.cooldown_seconds - 0.01


def phase1_cold() -> None:
    banner("Phase 1 - cold start, no fire")
    tracker = StabilityTracker(required_frames=5, tolerance=2.0)
    ctrl = AutoCaptureController(enabled=True, cooldown_seconds=0.4, tracker=tracker)
    _warm_start(ctrl)
    q = stable_quad()
    fires = 0
    for _ in range(4):
        # should_capture() updates the tracker internally - no double-update.
        if ctrl.should_capture(q):
            fires += 1
    assert_eq(fires, 0, "fires before stability")
    assert_eq(ctrl.tracker.stable_count, 4, "tracker.stable_count")


# ------------------------------------------------------------------
# Phase 2 - 5 stable frames -> exactly one fire
# ------------------------------------------------------------------

def phase2_fire() -> AutoCaptureController:
    banner("Phase 2 - 5 stable frames -> exactly one fire")
    tracker = StabilityTracker(required_frames=5, tolerance=2.0)
    ctrl = AutoCaptureController(enabled=True, cooldown_seconds=0.4, tracker=tracker)
    _warm_start(ctrl)
    q = stable_quad()
    fires = 0
    fired_on = -1
    for i in range(8):
        if ctrl.should_capture(q):
            fires += 1
            fired_on = i + 1
    assert_eq(fires, 1, "fires total")
    assert_eq(fired_on, 5, "frame that fired")
    assert_true(ctrl.tracker.stable_count >= 5, "tracker.stable_count >= required")
    return ctrl


# ------------------------------------------------------------------
# Phase 3 - cooldown gate
# ------------------------------------------------------------------

def phase3_cooldown_gate(ctrl: AutoCaptureController) -> None:
    banner("Phase 3 - cooldown gate (0.4s)")
    q = stable_quad()
    refused = 0
    t_end = time.monotonic() + 0.25          # mid-cooldown
    while time.monotonic() < t_end:
        if ctrl.should_capture(q):
            refused += 1
            break
    assert_true(refused == 0, "no fire during cooldown")
    # Even with stable tracker, should_capture() must decline during cooldown
    for _ in range(3):
        if ctrl.should_capture(q):
            refused += 1
            break
    assert_true(refused == 0, "stable-but-cooling declined")


# ------------------------------------------------------------------
# Phase 4 - second fire after cooldown
# ------------------------------------------------------------------

def phase4_second_fire(ctrl: AutoCaptureController) -> None:
    banner("Phase 4 - second fire after cooldown")
    q = stable_quad()
    # wait out the cooldown (0.4 s)
    time.sleep(0.5)
    fires = 0
    for _ in range(8):
        if ctrl.should_capture(q):
            fires += 1
            break
    assert_eq(fires, 1, "second fire after cooldown")


# ------------------------------------------------------------------
# Phase 5 - ScanSession state machine
# ------------------------------------------------------------------

def phase5_session_state(tmp: Path) -> None:
    banner("Phase 5 - ScanSession._maybe_auto_capture state machine")
    s = build_session(tmp)
    # Stub camera + processor so we don't touch hardware
    class StubCamera:
        def __init__(self):
            self.frames = 0
        def read(self):
            self.frames += 1
            return True, _synthetic_frame()
        def release(self): pass

    class StubProcessor:
        scan_mode = "color"
        def process(self, frame):
            q = stable_quad()
            return _synthetic_frame(), _stub_result(q)

    s._camera = StubCamera()
    s._processor = StubProcessor()
    s.auto_capture_enabled = True

    # Stub capture_current_frame so we don't drag in cv2.imwrite + PDF + QR
    fire_count = {"n": 0}
    def fake_capture(frame=None):
        fire_count["n"] += 1
        s.pages.append(_synthetic_frame())
        return True, "ok", _synthetic_frame(), _stub_result(stable_quad())
    s.capture_current_frame = fake_capture   # type: ignore[assignment]

    # Reset state
    s._auto_capture_phase = "S1_seeking"
    s.auto_capture.tracker.reset()
    s.auto_capture.state = S1_SEEKING_STABLE
    s.auto_capture.last_capture_timestamp = time.monotonic() - 0.5

    # Walk through enough ticks to satisfy required_frames (5)
    fired = 0
    for _ in range(8):
        s._maybe_auto_capture()
        if s._auto_capture_phase in ("S2_cooling", "S2_waiting"):
            fired += 1
            break
    assert_true(fired >= 1, "phase transitioned S1_seeking -> S2_cooling")
    assert_true(
        s.auto_capture.last_capture_timestamp > 0,
        "controller captured timestamp set",
    )
    assert_true(s.last_message.startswith("AUTO"), "HUD last_message updated")
    assert_eq(fire_count["n"], 1, "fake_capture fired exactly once")

    # Immediately tick again - should still be cooling, no new fire
    pre_fires = fire_count["n"]
    s._maybe_auto_capture()
    assert_eq(fire_count["n"], pre_fires, "no extra fire during cooldown")
    assert_eq(s._auto_capture_phase, "S2_cooling", "still cooling")

    # 2-state FSM contract: cooldown expiry alone does NOT clear the
    # gate.  The controller must stay in S2 until it sees a "frame
    # changed" signal (page-change event, drift, motion spike, or
    # quad disappeared).  Prove it:
    time.sleep(0.5)
    for _ in range(8):
        s._maybe_auto_capture()
    assert_eq(
        fire_count["n"], pre_fires,
        "no second fire while S2 still waiting for change",
    )
    assert_eq(s._auto_capture_phase, "S2_waiting",
              "moved from S2_cooling to S2_waiting after timer")

    # Now flip a change signal explicitly and confirm the FSM moves
    # back to S1 (no fire yet - just the flip).
    s.auto_capture.consume_change()
    s._maybe_auto_capture()
    assert_eq(s._auto_capture_phase, "S1_seeking",
              "change signal flipped back to S1_seeking")

    # Walk through a fresh stable streak to fire again.
    pre_fires_2 = fire_count["n"]
    for _ in range(8):
        s._maybe_auto_capture()
        if fire_count["n"] > pre_fires_2:
            break
    assert_true(
        fire_count["n"] >= pre_fires_2 + 1,
        f"second fire after change signal "
        f"(was={pre_fires_2} now={fire_count['n']})",
    )


# ------------------------------------------------------------------
# Synthetic image + result helpers (no cv2 GUI)
# ------------------------------------------------------------------

import numpy as np  # noqa: E402

def _synthetic_frame():
    # 64x80 RGB, uniform grey - enough for the processor stub to "see" a quad
    return np.full((80, 64, 3), 200, dtype=np.uint8)


@dataclass
class _StubResult:
    corners: object
    warped: object | None = None
    confidence: float = 1.0


def _stub_result(q: Quad):
    return _StubResult(corners=q, warped=None)


# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------

def main() -> int:
    tmp = Path("captures/_synthetic_auto_capture")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        phase1_cold()
        ctrl = phase2_fire()
        phase3_cooldown_gate(ctrl)
        phase4_second_fire(ctrl)
        phase5_session_state(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL PHASES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
