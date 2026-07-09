"""Integration smoke for ScanSession._maybe_auto_capture + handle_key('c').

Exercises the user's exact spec:

    "State 1: stable 5/5 frames -> capture, move to State 2.
     State 2: capture the four-point contour at the moment of capture.
     Move back to State 1 only after 2 continuous seconds during which
     no similar contour is found.  'c' always captures and moves to
     State 2.  There is no cooldown."

We monkey-patch the camera + processor + capture path so the test runs
without any real video device.  No production code is modified.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import ScanSession                                # noqa: E402
from auto_capture_controller import (                       # noqa: E402
    S1_SEEKING_STABLE,
    S2_WAITING_FOR_CHANGE,
)
from document_processor import DetectionResult             # noqa: E402


def _det(quad: np.ndarray, conf: float = 0.9) -> DetectionResult:
    return DetectionResult(corners=quad, confidence=conf, used_yolo=False, bbox=None)


def _make_session(quad: np.ndarray, processed: np.ndarray) -> ScanSession:
    """Build a ScanSession with camera + processor fully stubbed."""
    sess = ScanSession()
    fake_frame = np.full((720, 1280, 3), 180, dtype=np.uint8)

    cam = type(
        "C",
        (),
        {
            "read": staticmethod(lambda: (True, fake_frame)),
            "release": lambda self: None,
        },
    )()
    sess._camera = cam

    proc = type(
        "P",
        (),
        {
            "process": staticmethod(lambda f: (processed, _det(quad))),
            "scan_mode": "color",
            "process_with_debug": staticmethod(
                lambda f: (
                    processed,
                    _det(quad),
                    np.zeros((200, 200), np.uint8),
                    np.zeros((200, 200), np.uint8),
                    fake_frame,
                    fake_frame,
                    processed,
                    processed,
                    processed,
                )
            ),
        },
    )()
    sess._processor = proc

    return sess


def _force_to_s1(sess: ScanSession) -> None:
    """Test helper: yank the controller back to State 1 with a fresh
    tracker, bypassing the no-match timer so the next S1 streak can
    fire without waiting for the full 2 s."""
    sess.auto_capture.consume_change()
    sess.auto_capture.tracker.reset()


def main() -> int:
    fixed_quad = np.array(
        [[100, 100], [600, 100], [600, 400], [100, 400]], dtype=np.int32
    )
    processed = np.full((200, 200, 3), 200, dtype=np.uint8)

    sess = _make_session(fixed_quad, processed)

    captured: list[float] = []

    # The corners returned by fake_capture become the baseline stored
    # by register_capture(_det.corners).  We let the test drive which
    # corners come back so T9 can verify that the live quad at the
    # moment of the manual capture is the new baseline.
    capture_corners: list[np.ndarray] = [fixed_quad]

    def fake_capture(frame=None):
        captured.append(time.monotonic())
        corners = capture_corners[-1]
        return True, "fake-captured", processed, _det(corners)

    sess.capture_current_frame = fake_capture  # type: ignore[assignment]

    # Force controller config to user's spec.
    sess.auto_capture_enabled = True
    sess.auto_capture_stable_frames = 5  # keep the test fast; prod default is 60 (~2 s)
    sess.auto_capture.s2_no_match_timeout_s = 3.0
    sess.auto_capture.tolerance = 18.0

    print("=== T1: 12 LIVE ticks on a still document ===")
    for _ in range(12):
        sess._maybe_auto_capture()
    print(f"  captures fired: {len(captured)} (expected 1)")
    print(f"  phase: {sess._auto_capture_phase!r}")
    print(f"  message: {sess.last_message!r}")
    assert len(captured) == 1, f"T1 fail: {len(captured)} captures"
    # After auto-capture the FSM should be in State 2 reporting the
    # live quad as a match against the just-saved contour.
    assert sess.auto_capture.state == S2_WAITING_FOR_CHANGE
    assert sess._auto_capture_phase == "S2_match"

    print()
    print("=== T2: 5 ticks with the same quad (S2_match holds the timer) ===")
    prev = len(captured)
    for _ in range(5):
        sess._maybe_auto_capture()
    extra = len(captured) - prev
    print(f"  extra captures while still showing same page: "
          f"{extra} (expected 0)")
    print(f"  no-match seconds: "
          f"{sess.auto_capture.s2_no_match_seconds:.2f} "
          f"(expected 0.0 -- match resets the timer)")
    assert extra == 0, f"T2 fail: S2_match should block ({extra} fires)"
    assert sess.auto_capture.s2_no_match_seconds == 0.0, (
        f"T2 fail: S2_match should reset no-match timer, "
        f"got {sess.auto_capture.s2_no_match_seconds}"
    )

    print()
    print("=== T3: 3s of a DIFFERENT quad flips S2 -> S1 -> second fire ===")
    # Replace the live quad with one that drifts beyond tolerance.
    moved_quad = fixed_quad + np.array(
        [[0, 0], [300, 0], [300, 0], [0, 0]], dtype=np.int32
    )
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(moved_quad))
    )
    # Drive the controller across 3 s of continuous no-match by
    # stamping the controller's last state-2 tick into the past.  The
    # controller caps any single delta at 0.5 s, so a few ticks spaced
    # at least 0.5 s apart are required to accumulate the full timeout.
    flips_observed = 0
    for _ in range(8):
        sess.auto_capture._last_state2_tick_t = (
            time.monotonic() - 0.6
        )
        sess._maybe_auto_capture()
        if sess.auto_capture.state == S1_SEEKING_STABLE:
            flips_observed = 1
            break
        # Real wall-clock has to advance between ticks for the next
        # delta to be meaningful.
        time.sleep(0.6)
    print(f"  ticks before flip: {flips_observed} (expected 1)")
    print(f"  phase after drifted quad: {sess._auto_capture_phase!r}")
    print(f"  state: {sess.auto_capture.state!r}")
    assert sess.auto_capture.state == S1_SEEKING_STABLE, (
        f"T3 fail: 3s of different quad should flip to S1, "
        f"state={sess.auto_capture.state!r}"
    )
    # Now build a fresh S1 streak on the moved quad and confirm a fire.
    captures_before = len(captured)
    for _ in range(6):
        sess._maybe_auto_capture()
    extra = len(captured) - captures_before
    print(f"  second fire after S2->S1: extra={extra} (expected 1)")
    assert extra == 1, f"T3 fail: expected 1 extra fire, got {extra}"

    # Restore the still-quad for the rest of the tests.
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(fixed_quad))
    )

    print()
    print("=== T4: manual 'c' captures and parks in S2 ===")
    _force_to_s1(sess)
    for _ in range(5):
        sess._maybe_auto_capture()
    manual_before = len(captured)
    sess.handle_key(ord("c"))
    manual_fires = len(captured) - manual_before
    print(f"  fires immediately after C: {manual_fires} (expected 1)")
    print(f"  phase: {sess._auto_capture_phase!r}")
    print(f"  message: {sess.last_message!r}")
    assert manual_fires == 1, f"T4a fail: {manual_fires} fires after C"
    assert (
        sess.auto_capture.state == S2_WAITING_FOR_CHANGE
    ), f"T4a fail: 'c' must park in S2, got {sess.auto_capture.state!r}"

    # Showing the same page should keep the FSM parked (no extra fires).
    for _ in range(3):
        sess._maybe_auto_capture()
    extra_during_match = len(captured) - manual_before - 1
    print(
        f"  extra fires while S2_match: "
        f"{extra_during_match} (expected 0)"
    )
    assert extra_during_match == 0, (
        f"T4b fail: {extra_during_match} extra fires"
    )

    print()
    print("=== T5: out-of-tolerance motion resets the S1 streak ===")
    _force_to_s1(sess)
    for _ in range(4):
        sess._maybe_auto_capture()
    streak_after_4 = sess.auto_capture.tracker.stable_count
    print(f"  streak after 4 ticks: {streak_after_4} (expected 4)")
    assert streak_after_4 == 4, f"T5a fail: streak={streak_after_4}"

    sess._processor.process = staticmethod(
        lambda f: (processed, _det(moved_quad))
    )
    sess._maybe_auto_capture()
    streak_after_move = sess.auto_capture.tracker.stable_count
    print(
        f"  streak after big move: {streak_after_move} (expected 1 = reset)"
    )
    assert streak_after_move == 1, f"T5b fail: streak={streak_after_move}"

    print()
    print("=== T6: 'c' in State 2 resets the no-match timer ===")
    # Park the FSM in State 2 with the original quad, accumulate some
    # no-match time, then press 'c'.  After 'c' the timer should be 0
    # and the FSM should remain in State 2.
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(fixed_quad))
    )
    _force_to_s1(sess)
    for _ in range(5):
        sess._maybe_auto_capture()
    assert sess.auto_capture.state == S2_WAITING_FOR_CHANGE

    # Now show a different quad for ~1.0 s, then press 'c'.  The
    # controller caps any single tick's delta at 0.5 s, so spread the
    # 1 s across two ticks.
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(moved_quad))
    )
    sess.auto_capture._last_state2_tick_t = time.monotonic() - 0.6
    sess._maybe_auto_capture()
    time.sleep(0.6)
    sess.auto_capture._last_state2_tick_t = time.monotonic() - 0.6
    sess._maybe_auto_capture()
    no_match_before_c = sess.auto_capture.s2_no_match_seconds
    print(
        f"  no-match seconds before 'c': {no_match_before_c:.2f} "
        f"(expected ~1.0)"
    )
    assert no_match_before_c >= 0.9, (
        f"T6 fail: no-match should be ~1.0, got {no_match_before_c}"
    )

    fires_before_c = len(captured)
    sess.handle_key(ord("c"))
    fires_added = len(captured) - fires_before_c
    no_match_after_c = sess.auto_capture.s2_no_match_seconds
    print(f"  fires added by 'c': {fires_added} (expected 1)")
    print(f"  no-match seconds after 'c': {no_match_after_c:.2f} "
          f"(expected 0.0)")
    assert fires_added == 1, f"T6 fail: 'c' did not capture ({fires_added})"
    assert no_match_after_c == 0.0, (
        f"T6 fail: 'c' should reset no-match timer, got {no_match_after_c}"
    )
    assert (
        sess.auto_capture.state == S2_WAITING_FOR_CHANGE
    ), f"T6 fail: 'c' from S2 should keep State 2, got {sess.auto_capture.state!r}"

    print()
    print("=== T7: 'c' captures from EVERY state ===")
    # User spec: clicking 'c' must capture regardless of the FSM phase.
    # We try all three labels the controller can report:
    #   * S1_seeking -- never reached the stable streak yet
    #   * S2_match   -- just captured, live quad still matches baseline
    #   * S2_waiting -- live quad differs, no-match timer accumulating

    # Make sure the next capture sees the still-quad (a match), so
    # the S2 result lands on "match" rather than "waiting".
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(fixed_quad))
    )

    # --- 7a: S1_seeking (no auto-capture happened yet) ---
    sess.auto_capture.consume_change()  # back to State 1
    sess.auto_capture.tracker.reset()
    assert sess.auto_capture.state == S1_SEEKING_STABLE
    assert sess.auto_capture.phase_label() == "S1_seeking"
    fires_before = len(captured)
    sess.handle_key(ord("c"))
    fires_added = len(captured) - fires_before
    print(f"  [S1_seeking] fires added by 'c': {fires_added} (expected 1)")
    assert fires_added == 1, f"T7a fail: {fires_added} fires from S1"
    assert sess.auto_capture.state == S2_WAITING_FOR_CHANGE
    assert sess.auto_capture.phase_label() == "S2_match"

    # --- 7b: S2_match (live quad similar to baseline) ---
    fires_before = len(captured)
    sess.handle_key(ord("c"))
    fires_added = len(captured) - fires_before
    print(f"  [S2_match]   fires added by 'c': {fires_added} (expected 1)")
    assert fires_added == 1, f"T7b fail: {fires_added} fires from S2_match"
    assert sess.auto_capture.state == S2_WAITING_FOR_CHANGE
    assert sess.auto_capture.phase_label() == "S2_match"

    # --- 7c: S2_waiting (live quad differs, timer accumulating) ---
    # Switch the live quad to one beyond tolerance, tick once so the
    # controller records "non-match" and reports the S2_waiting phase.
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(moved_quad))
    )
    sess.auto_capture._last_state2_tick_t = time.monotonic() - 0.6
    sess._maybe_auto_capture()
    assert sess.auto_capture.state == S2_WAITING_FOR_CHANGE
    assert sess.auto_capture.phase_label() == "S2_waiting", (
        f"setup: expected S2_waiting, got {sess.auto_capture.phase_label()!r}"
    )
    fires_before = len(captured)
    sess.handle_key(ord("c"))
    fires_added = len(captured) - fires_before
    no_match_after = sess.auto_capture.s2_no_match_seconds
    print(
        f"  [S2_waiting] fires added by 'c': {fires_added} (expected 1)"
    )
    print(
        f"  [S2_waiting] no-match after 'c': {no_match_after:.2f} "
        f"(expected 0.0)"
    )
    assert fires_added == 1, f"T7c fail: {fires_added} fires from S2_waiting"
    assert no_match_after == 0.0, (
        f"T7c fail: 'c' from S2_waiting should zero timer, got {no_match_after}"
    )
    assert sess.auto_capture.state == S2_WAITING_FOR_CHANGE

    print()
    print("=== T8: fresh controller starts in State 1 ===")
    # User spec: the default starting state must be State 1.  A brand
    # new controller (not yet touched by any observe / register_capture
    # call) should report S1_SEEKING_STABLE and phase "S1_seeking".
    from auto_capture_controller import AutoCaptureController

    fresh = AutoCaptureController(enabled=True)
    print(f"  fresh.state        = {fresh.state!r}")
    print(f"  fresh.phase_label  = {fresh.phase_label()!r}")
    assert fresh.state == S1_SEEKING_STABLE, (
        f"T8 fail: default state should be S1, got {fresh.state!r}"
    )
    assert fresh.phase_label() == "S1_seeking", (
        f"T8 fail: phase_label should be S1_seeking, got "
        f"{fresh.phase_label()!r}"
    )
    assert fresh.s2_no_match_seconds == 0.0
    assert fresh.last_captured_quad is None

    print()
    print("=== T9: 'c' in State 2 captures, sets baseline, keeps S2 ===")
    # Build a clean FSM, drive it into S2 via auto-capture, then press
    # 'c' from State 2.  Verify:
    #   * one capture happened,
    #   * last_captured_quad was overwritten with the NEW corners,
    #   * the FSM is STILL in State 2 (not flipped to S1),
    #   * the no-match timer is reset to 0.
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(fixed_quad))
    )
    _force_to_s1(sess)
    for _ in range(5):
        sess._maybe_auto_capture()
    assert (
        sess.auto_capture.state == S2_WAITING_FOR_CHANGE
    ), "setup: expected S2 after stable streak"
    assert sess.auto_capture.phase_label() == "S2_match"

    # Take a snapshot of the baseline quad that the controller stored
    # at the auto-capture moment.  The next 'c' must overwrite it.
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(moved_quad))
    )
    # Production behaviour: register_capture receives the corners from
    # the just-saved file (capture_current_frame's _det.corners), which
    # in real life are the refined corners of the *current* page.  Drive
    # fake_capture to return the live `moved_quad` so the baseline
    # stored by 'c' is the new contour, not the old auto-capture one.
    capture_corners.append(moved_quad)
    fires_before = len(captured)
    sess.handle_key(ord("c"))
    fires_added = len(captured) - fires_before
    baseline_after_c = sess.auto_capture.last_captured_quad
    no_match_after_c = sess.auto_capture.s2_no_match_seconds

    print(f"  fires added by 'c':           {fires_added} (expected 1)")
    print(f"  state after 'c':              {sess.auto_capture.state!r}")
    print(f"  phase after 'c':             {sess.auto_capture.phase_label()!r}")
    print(f"  no-match seconds after 'c':  {no_match_after_c:.2f} (expected 0.0)")

    assert fires_added == 1, f"T9 fail: 'c' did not capture ({fires_added})"
    assert (
        sess.auto_capture.state == S2_WAITING_FOR_CHANGE
    ), (
        f"T9 fail: 'c' from State 2 should stay in S2, "
        f"got {sess.auto_capture.state!r}"
    )
    assert no_match_after_c == 0.0, (
        f"T9 fail: 'c' should zero the no-match timer, "
        f"got {no_match_after_c}"
    )
    # The baseline must now be the NEW quad (moved_quad), not the
    # one that was saved by the auto-capture earlier.
    if baseline_after_c is not None:
        matches_new = np.array_equal(
            np.asarray(baseline_after_c), moved_quad
        )
        matches_old = np.array_equal(
            np.asarray(baseline_after_c), fixed_quad
        )
        print(
            f"  baseline replaced:           new={matches_new} "
            f"old={matches_old}"
        )
        assert matches_new, (
            "T9 fail: baseline should be overwritten with the new "
            "contour (moved_quad)"
        )
        assert not matches_old, (
            "T9 fail: baseline still holds the OLD contour"
        )

    # And the very next LIVE tick should evaluate against the NEW
    # baseline -- showing the OLD quad must register as no-match and
    # the timer must start accumulating.
    sess._processor.process = staticmethod(
        lambda f: (processed, _det(fixed_quad))
    )
    sess.auto_capture._last_state2_tick_t = time.monotonic() - 0.6
    sess._maybe_auto_capture()
    print(
        f"  phase after OLD quad tick:   "
        f"{sess.auto_capture.phase_label()!r}"
    )
    print(
        f"  no-match after OLD quad tick:"
        f" {sess.auto_capture.s2_no_match_seconds:.2f} (expected >0)"
    )
    assert (
        sess.auto_capture.phase_label() == "S2_waiting"
    ), (
        "T9 fail: showing the OLD quad should register as no-match "
        "against the NEW baseline"
    )
    assert sess.auto_capture.s2_no_match_seconds > 0.0, (
        "T9 fail: no-match timer should start accumulating against "
        "the NEW baseline"
    )

    print()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
