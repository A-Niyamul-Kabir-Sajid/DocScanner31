"""
Smoke test for `mp3_player`.

Mirrors the phase structure of `runs/smoke_sound.py`:

    Phase 1 - module imports, backend probes
    Phase 2 - disabled MP3Player is a no-op
    Phase 3 - missing-file handling (one-shot warning dedup)
    Phase 4 - play_clip argument shape (Linux Pi path)
    Phase 5 - concurrent dispatch is non-blocking + coalesces bursts
    Phase 6 - per-event lock is released after each run

Exit 0 == ALL PHASES PASS.
"""
from __future__ import annotations

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mp3_player                       # noqa: E402


def _banner(t: str) -> None:
    print()
    print("=" * 72)
    print(f"  {t}")
    print("=" * 72)


def _ok(t: str) -> None:
    print(f"  OK  {t}")


def _fail(t: str, exc: BaseException | None = None) -> None:
    print(f"  FAIL {t}")
    if exc is not None:
        print(f"       -> {type(exc).__name__}: {exc}")
    raise SystemExit(1)


# ------------------------------------------------------------------ #
# Phase 1 - module imports + backend probes
# ------------------------------------------------------------------ #
def phase_1() -> None:
    _banner("Phase 1 - imports + backend probes")
    try:
        pydub_ok = mp3_player._pydub_available()
        alsa_ok = mp3_player._alsaaudio_available()
        ffmpeg_ok = mp3_player._ffmpeg_available()
        linux_pi_ok = mp3_player._linux_pi_audio_available()
        _ok(f"_pydub_available()      = {pydub_ok}")
        _ok(f"_alsaaudio_available()  = {alsa_ok}")
        _ok(f"_ffmpeg_available()     = {ffmpeg_ok}")
        _ok(f"_linux_pi_audio_available() = {linux_pi_ok}")
    except Exception as e:
        _fail("backend probe raised", e)


# ------------------------------------------------------------------ #
# Phase 2 - disabled player is a silent no-op
# ------------------------------------------------------------------ #
def phase_2() -> None:
    _banner("Phase 2 - disabled MP3Player is a no-op")
    try:
        p = mp3_player.MP3Player(
            enabled=False,
            captured_file="captured.mp3",
            deleted_file="deleted.mp3",
            device="plughw:2,0",
            volume_db=8.0,
            project_root=ROOT,
        )
        r1 = p.play_event("captured")
        r2 = p.play_event("page_deleted")
        r3 = p.play_event("bogus")
        if r1 is False and r2 is False and r3 is False:
            _ok("disabled player returns False for captured / page_deleted / bogus")
        else:
            _fail(f"disabled player returned non-False: {r1}, {r2}, {r3}")
    except Exception as e:
        _fail("disabled player raised", e)


# ------------------------------------------------------------------ #
# Phase 3 - missing-file handling (one-shot warning dedup)
# ------------------------------------------------------------------ #
def phase_3() -> None:
    _banner("Phase 3 - missing-file handling")
    try:
        # Both clips point at non-existent files.
        p = mp3_player.MP3Player(
            enabled=True,
            captured_file="definitely_does_not_exist.mp3",
            deleted_file="also_missing.mp3",
            device="plughw:2,0",
            volume_db=8.0,
            project_root=os.path.join(ROOT, "captures"),  # wrong dir
        )
        p._missing_warned.clear()
        r1 = p.play_event("captured")
        r2 = p.play_event("captured")     # second call -> dedup, still False
        r3 = p.play_event("page_deleted")
        if not (r1 is False and r2 is False and r3 is False):
            _fail(f"missing-file dispatch returned non-False: {r1}, {r2}, {r3}")
        _ok("missing-file dispatch returns False without raising")
    except Exception as e:
        _fail("missing-file handling raised", e)


# ------------------------------------------------------------------ #
# Phase 4 - play_clip argument shape (Linux Pi path)
# ------------------------------------------------------------------ #
def phase_4() -> None:
    _banner("Phase 4 - play_clip argument shape")
    try:
        captured_args: dict = {}

        def fake_play_clip(filename, *, device, volume_db, sample_rate, channels, chunk_bytes):
            captured_args["filename"] = filename
            captured_args["device"] = device
            captured_args["volume_db"] = volume_db
            captured_args["sample_rate"] = sample_rate
            captured_args["channels"] = channels
            captured_args["chunk_bytes"] = chunk_bytes
            return True

        original = mp3_player.play_clip
        mp3_player.play_clip = fake_play_clip
        try:
            real_clip = os.path.join(ROOT, "captured.mp3")
            created_stub = False
            if not os.path.exists(real_clip):
                # 1-byte stub is fine; the fake ignores contents.
                with open(real_clip, "wb") as fh:
                    fh.write(b"\x00")
                created_stub = True

            p = mp3_player.MP3Player(
                enabled=True,
                captured_file="captured.mp3",
                deleted_file="deleted.mp3",
                device="plughw:9,9",       # arbitrary - we are not opening hw
                volume_db=12.5,
                project_root=ROOT,
            )
            p._missing_warned.clear()
            p.play_event("captured")
            time.sleep(0.3)               # let the daemon thread call the fake

            expected = dict(
                filename=real_clip,
                device="plughw:9,9",
                volume_db=12.5,
                sample_rate=mp3_player.DEFAULT_SAMPLE_RATE,
                channels=mp3_player.DEFAULT_CHANNELS,
                chunk_bytes=mp3_player.DEFAULT_CHUNK_BYTES,
            )
            if captured_args != expected:
                _fail(f"play_clip args mismatch: got={captured_args}, want={expected}")
            _ok(f"play_clip received expected args: {captured_args}")
        finally:
            mp3_player.play_clip = original
            if created_stub:
                try:
                    os.remove(real_clip)
                except OSError:
                    pass
    except Exception as e:
        _fail("play_clip argument shape failed", e)


# ------------------------------------------------------------------ #
# Phase 5 - concurrent dispatch is non-blocking + coalesces bursts
# ------------------------------------------------------------------ #
def phase_5() -> None:
    _banner("Phase 5 - concurrent dispatch is non-blocking + coalesces bursts")
    try:
        # Event-gated fake: simulates a clip that "plays" until we release.
        started = threading.Event()
        release = threading.Event()

        def slow_play(filename, **_):
            started.set()
            release.wait(timeout=2.0)
            return True

        # Ensure clip files exist so the existence check passes.
        for name in ("captured.mp3", "deleted.mp3"):
            path = os.path.join(ROOT, name)
            if not os.path.exists(path):
                with open(path, "wb") as fh:
                    fh.write(b"\x00")

        p = mp3_player.MP3Player(
            enabled=True,
            captured_file="captured.mp3",
            deleted_file="deleted.mp3",
            device="plughw:9,9",
            volume_db=8.0,
            project_root=ROOT,
        )
        p._missing_warned.clear()

        original = mp3_player.play_clip
        mp3_player.play_clip = slow_play
        try:
            # (a) Burst-coalesce: while a clip is playing, new requests
            #     must return False instantly without blocking.
            p.play_event("captured")
            if not started.wait(timeout=2.0):
                _fail("daemon thread never entered slow_play")
            started.clear()

            t0 = time.perf_counter()
            burst_returns = [p.play_event("captured") for _ in range(50)]
            elapsed = time.perf_counter() - t0
            if any(burst_returns):
                _fail(f"in-flight burst should have been dropped, got: {burst_returns}")
            if elapsed > 0.05:
                _fail(f"burst enqueue too slow: {elapsed*1000:.1f} ms")
            _ok(f"50 in-flight play_event() calls coalesced+returned in {elapsed*1000:.2f} ms")

            # (b) After the clip finishes the lock is released and a fresh
            #     dispatch starts a new daemon thread.
            release.set()
            time.sleep(0.3)               # let the worker exit
            p.play_event("captured")
            if not started.wait(timeout=2.0):
                _fail("daemon thread did not start on second dispatch")
            release.set()
            time.sleep(0.3)
            _ok("post-completion dispatch starts a fresh daemon thread")
        finally:
            mp3_player.play_clip = original
    except Exception as e:
        _fail("concurrent dispatch test raised", e)


# ------------------------------------------------------------------ #
# Phase 6 - per-event lock is released after each run
# ------------------------------------------------------------------ #
def phase_6() -> None:
    _banner("Phase 6 - lock is released after each run")
    try:
        def fast_play(filename, **_):
            return True

        # Ensure the clip exists.
        real_clip = os.path.join(ROOT, "captured.mp3")
        created_stub = False
        if not os.path.exists(real_clip):
            with open(real_clip, "wb") as fh:
                fh.write(b"\x00")
            created_stub = True

        original = mp3_player.play_clip
        mp3_player.play_clip = fast_play
        try:
            p = mp3_player.MP3Player(
                enabled=True,
                captured_file="captured.mp3",
                deleted_file="deleted.mp3",
                device="plughw:9,9",
                volume_db=8.0,
                project_root=ROOT,
            )
            p._missing_warned.clear()

            p.play_event("captured")
            time.sleep(0.3)
            # If the lock wasn't released this acquire would block forever.
            acquired = p._play_locks["captured"].acquire(blocking=False)
            if not acquired:
                _fail("per-event lock was NOT released after first run")
            p._play_locks["captured"].release()
            _ok("per-event lock is released after run")
        finally:
            mp3_player.play_clip = original
            if created_stub:
                try:
                    os.remove(real_clip)
                except OSError:
                    pass
    except Exception as e:
        _fail("lock-release test raised", e)


# ------------------------------------------------------------------ #
# Driver
# ------------------------------------------------------------------ #
def main() -> int:
    print("runs/smoke_mp3.py - mp3_player phase harness")
    phase_1()
    phase_2()
    phase_3()
    phase_4()
    phase_5()
    phase_6()
    print()
    print("ALL PHASES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
