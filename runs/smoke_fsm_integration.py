"""Integration smoke for ScanSession._maybe_auto_capture + handle_key('c').

Exercises the user's exact spec:

    "automatic capture if 5/5 frames found and then 5 second delay after"

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


def main() -> int:
    fixed_quad = np.array(
        [[100, 100], [600, 100], [600, 400], [100, 400]], dtype=np.int32
    )
    processed = np.full((200, 200, 3), 200, dtype=np.uint8)

    sess = _make_session(fixed_quad, processed)

    captured: list[float] = []

    def fake_capture(frame=None):
        captured.append(time.monotonic())
        return True, "fake-captured", processed, _det(fixed_quad)

    sess.capture_current_frame = fake_capture  # type: ignore[assignment]

    # Force controller config to user's spec.
    sess.auto_capture_enabled = True
    sess.auto_capture_stable_frames = 5
    sess.auto_capture_cooldown_s = 5.0
    sess.auto_capture_tolerance_px = 18.0

    print("=== T1: 12 LIVE ticks on a still document ===")
    for _ in range(12):
        sess._maybe_auto_capture()
    print(f"  captures fired: {len(captured)} (expected 1)")
    print(f"  phase: {sess._auto_capture_phase!r}")
    print(f"  message: {sess.last_message!r}")
    assert len(captured) == 1, f"T1 fail: {len(captured)} captures"

    print()
    print("=== T2: 5 ticks DURING cooldown (should NOT fire) ===")
    prev = len(captured)
    for _ in range(5):
        sess._maybe_auto_capture()
    extra = len(captured) - prev
    print(f"  extra captures during cooldown: {extra} (expected 0)")
    assert extra == 0, f"T2 fail: cooldown did not block ({extra} fires)"

    print()
    print("=== T3: wait out the 5 s cooldown -> should fire AGAIN ===")
    # Bypass both cooldown gates so the test runs in <1 s instead of 5+ s.
    # The ScanSession uses two cooldowns in series (LIVE loop + controller),
    # both must be reset to re-arm.
    sess._auto_capture_cooldown_until = 0.0
    sess.auto_capture.last_capture_timestamp = 0.0
    sess.auto_capture.tracker.reset()
    for _ in range(6):
        sess._maybe_auto_capture()
    print(f"  total captures: {len(captured)} (expected 2)")
    print(f"  phase: {sess._auto_capture_phase!r}")
    assert len(captured) == 2, f"T3 fail: {len(captured)} captures"

    print()
    print("=== T4: manual C key arms the same 5s cooldown ===")
    sess.auto_capture.tracker.reset()
    sess._auto_capture_cooldown_until = 0.0
    for _ in range(5):
        sess._maybe_auto_capture()
    manual_before = len(captured)
    sess.handle_key(ord("c"))
    manual_fires = len(captured) - manual_before
    print(f"  fires immediately after C: {manual_fires} (expected 1)")
    print(f"  phase: {sess._auto_capture_phase!r}")
    print(f"  message: {sess.last_message!r}")
    assert manual_fires == 1, f"T4a fail: {manual_fires} fires after C"

    for _ in range(3):
        sess._maybe_auto_capture()
    extra_during_manual_cd = len(captured) - manual_before - 1
    print(
        f"  extra fires during manual cooldown: "
        f"{extra_during_manual_cd} (expected 0)"
    )
    assert extra_during_manual_cd == 0, (
        f"T4b fail: {extra_during_manual_cd} extra fires"
    )

    print()
    print("=== T5: out-of-tolerance motion resets the streak ===")
    sess._auto_capture_cooldown_until = 0.0
    sess.auto_capture.tracker.reset()
    for _ in range(4):
        sess._maybe_auto_capture()
    streak_after_4 = sess.auto_capture.tracker.stable_count
    print(f"  streak after 4 ticks: {streak_after_4} (expected 4)")
    assert streak_after_4 == 4, f"T5a fail: streak={streak_after_4}"

    moved_quad = fixed_quad + np.array(
        [[0, 0], [300, 0], [300, 0], [0, 0]], dtype=np.int32
    )
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
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())