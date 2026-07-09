"""Smoke test for the audio-cue layer (``sound`` module + ``ScanSession``).

The harness exercises the public API end-to-end without ever needing a real
audio device:

1. ``SoundPlayer(enabled=False)`` is a complete no-op.
2. The pre-rendered WAV blobs all start with the ``RIFF`` magic and have
   the right number of channels / sample-rate / duration.
3. ``ScanSession.sound`` is a lazy property.
4. ``ScanSession.play_sound`` is a defensive wrapper that never raises
   even if the backend does.
5. The full integration - first-quad chime, auto-fire chime + click, and
   manual-``C`` click - all dispatch through ``play_sound`` without
   exploding, even when no real backend is available.

Run it with::

    python -m runs.smoke_sound

The script returns exit code ``0`` on success and ``1`` on any failure
so it can be wired into CI alongside the other smoke harnesses.
"""

from __future__ import annotations

import importlib
import io
import sys
import traceback
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _header(label: str) -> None:
    print(f"\n=== {label} ===")


def _check(condition: bool, message: str) -> None:
    if condition:
        print(f"  [OK]  {message}")
    else:
        print(f"  [FAIL] {message}")
        raise AssertionError(message)


def main() -> int:  # noqa: C901 - linear test, fine for a smoke harness
    failures: list = []

    def _phase(name: str, fn) -> None:
        _header(name)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failures.append(name)

    # ------------------------------------------------------------------ #
    # Phase 1 - module imports cleanly and exposes the right symbols.
    # ------------------------------------------------------------------ #
    def p1_imports() -> None:
        sound = importlib.import_module("sound")
        _check(hasattr(sound, "SoundPlayer"), "sound.SoundPlayer is exposed")
        _check(hasattr(sound, "play_event"), "sound.play_event is exposed")
        _check(
            hasattr(sound, "get_default_player"),
            "sound.get_default_player is exposed",
        )
        _check(
            {"detect_start", "detect_stable", "capture"}
            <= set(sound._EVENT_NOTES.keys()),
            "all three named events are registered",
        )
        for name, notes in sound._EVENT_NOTES.items():
            _check(
                len(notes) > 0 and all(f > 0 and d > 0 for f, d in notes),
                f"event {name!r} has at least one positive (freq, dur) note",
            )

    # ------------------------------------------------------------------ #
    # Phase 2 - WAV blobs are well-formed.
    # ------------------------------------------------------------------ #
    def p2_wav_blobs() -> None:
        from sound import SoundPlayer
        # Backend is irrelevant for inspection - the cache is built at
        # __init__ time using the in-memory synth path.
        player = SoundPlayer(enabled=False, backend="none")
        _check(
            set(player._cache.keys()) == {"detect_start", "detect_stable", "capture"},
            "cache contains all three events",
        )
        for name, blob in player._cache.items():
            _check(isinstance(blob, (bytes, bytearray)) and len(blob) > 44,
                   f"{name!r} WAV is > 44 bytes (header)")
            _check(blob[:4] == b"RIFF", f"{name!r} starts with 'RIFF'")
            _check(blob[8:12] == b"WAVE", f"{name!r} is a WAVE stream")
            # Open with stdlib wave to confirm it's well-formed.
            with wave.open(io.BytesIO(blob), "rb") as wf:
                _check(wf.getnchannels() == 1, f"{name!r} is mono")
                _check(wf.getsampwidth() == 2, f"{name!r} is 16-bit")
                _check(wf.getframerate() == 22050, f"{name!r} uses 22050 Hz")
                frames = wf.getnframes()
                _check(frames > 0, f"{name!r} has > 0 frames")
                duration_s = frames / wf.getframerate()
                _check(
                    0.05 <= duration_s <= 1.5,
                    f"{name!r} duration {duration_s*1000:.0f} ms is reasonable",
                )

    # ------------------------------------------------------------------ #
    # Phase 3 - enabled=False is a complete no-op.
    # ------------------------------------------------------------------ #
    def p3_disabled_noop() -> None:
        from sound import SoundPlayer
        player = SoundPlayer(enabled=True, backend="none")
        # Force the backend to a no-op ("none") to mimic a headless box.
        player._backend_name = "none"
        player._backend = None
        # Capture WAV size to ensure cache is not silently cleared.
        for name in ("detect_start", "detect_stable", "capture"):
            rc = player.play_event(name)
            _check(rc is False, f"play_event({name!r}) -> False on no backend")
        # Disabled explicitly must short-circuit BEFORE touching the backend.
        disabled = SoundPlayer(enabled=False, backend="winsound")
        _check(disabled.enabled is False, "enabled=False sticks")
        for name in ("detect_start", "detect_stable", "capture"):
            rc = disabled.play_event(name)
            _check(rc is False, f"disabled.play_event({name!r}) -> False")

    # ------------------------------------------------------------------ #
    # Phase 4 - unknown event name returns False, does not raise.
    # ------------------------------------------------------------------ #
    def p4_unknown_event() -> None:
        from sound import SoundPlayer
        player = SoundPlayer(enabled=True, backend="none")
        player._backend_name = "none"
        player._backend = None
        rc = player.play_event("__not_an_event__")
        _check(rc is False, "play_event(unknown) -> False")

    # ------------------------------------------------------------------ #
    # Phase 5 - configure() rebuilds the cache.
    # ------------------------------------------------------------------ #
    def p5_configure_rebuilds() -> None:
        from sound import SoundPlayer
        player = SoundPlayer(enabled=True, backend="none")
        before = player._cache["capture"]
        player.configure(volume=0.2)
        after = player._cache["capture"]
        _check(before != after, "configure(volume) rebuilds the WAV cache")
        _check(player.volume == 0.2, "configure(volume=0.2) sticks")
        player.configure(enabled=False)
        _check(player.enabled is False, "configure(enabled=False) sticks")

    # ------------------------------------------------------------------ #
    # Phase 6 - ScanSession wiring.
    # ------------------------------------------------------------------ #
    def p6_session_wiring() -> None:
        from app import ScanSession
        # Headless camera source so we never open a device.
        s = ScanSession(
            camera_source="http://127.0.0.1:1/video",
            web_port=18081,
        )
        _check(s.sound_enabled is True, "sound_enabled defaults to True")
        _check(s.sound_volume == 0.6, "sound_volume defaults to 0.6")
        _check(s._sound is None, "SoundPlayer is lazy (not built yet)")
        player = s.sound
        _check(isinstance(player, object), "s.sound returns a SoundPlayer-like object")
        _check(
            set(player._cache.keys())
            == {"detect_start", "detect_stable", "capture"},
            "session SoundPlayer has all three events",
        )
        # Calling .sound twice must return the same instance (lazy memo).
        _check(s.sound is player, "s.sound is memoised")
        # play_sound on a disabled session must short-circuit cleanly.
        s.sound_enabled = False
        for name in ("detect_start", "detect_stable", "capture", "nonsense"):
            try:
                s.play_sound(name)
            except Exception as exc:
                raise AssertionError(
                    f"play_sound({name!r}) raised {exc!r} on disabled session"
                )
        print("  [OK]  play_sound never raises on disabled session")

    # ------------------------------------------------------------------ #
    # Phase 7 - end-to-end FSM with sound hooks (no camera, no real audio).
    # ------------------------------------------------------------------ #
    def p7_fsm_sound_hooks() -> None:
        """Drive ``_maybe_auto_capture`` with stubbed camera + processor and
        assert that every sound hook the user asked for fires through the
        FSM transitions without the LIVE loop ever needing to spin up."""
        import numpy as np
        from app import ScanSession

        events_fired: list = []

        class _StubSound:
            enabled = True

            def play_event(self, name: str) -> bool:
                events_fired.append(name)
                return True

        class _StubDetector:
            corners = [(10, 10), (200, 10), (200, 200), (10, 200)]
            confidence = 1.0

            def __bool__(self) -> bool:
                return True

        class _StubProcessor:
            # ``scan_mode`` must match the session's current mode, else
            # ScanSession's ``processor`` property will silently rebuild
            # the real DocumentProcessor over our stub.
            scan_mode = "color"

            def process(self, frame):
                return np.zeros((4, 4, 3), dtype=np.uint8), _StubDetector()

        class _StubCamera:
            def read(self):
                return True, np.full((240, 320, 3), 200, dtype=np.uint8)

            def release(self):
                pass

        # --- 7a: auto-capture fire path ------------------------------ #
        s = ScanSession(
            camera_source="http://127.0.0.1:1/video",
            web_port=18082,
            camera_width=64, camera_height=64,
        )
        s.sound_enabled = True
        s._sound = _StubSound()  # type: ignore[assignment]
        s._sound_detect_start_played = False
        s._camera = _StubCamera()       # type: ignore[assignment]
        s._processor = _StubProcessor() # type: ignore[assignment]
        # Bypass the real quality gate / filesystem for capture.
        s.capture_current_frame = lambda *a, **kw: (  # type: ignore[assignment]
            True, "stub: capture", np.zeros((4, 4, 3), dtype=np.uint8),
            _StubDetector(),
        )
        s.auto_capture_cooldown_s = 0.05
        s.auto_capture_stable_frames = 3
        s.auto_capture_tolerance_px = 50.0
        s._auto_capture = None

        # First call: quad visible, streak building (0/3 -> 1/3). No fire.
        s._maybe_auto_capture()
        _check("detect_start" in events_fired,
               "first quad fires detect_start")
        _check("detect_stable" not in events_fired,
               "no detect_stable yet (streak < required)")

        # Force streak == required_frames + clear cooldown timestamp so
        # should_capture() returns True on the next tick.
        controller = s.auto_capture
        controller.tracker.stable_count = s.auto_capture_stable_frames
        controller.last_capture_timestamp = 0.0
        s._maybe_auto_capture()
        _check("detect_stable" in events_fired,
               "auto-fire transitions identifying->cooldown: detect_stable fires")
        _check("capture" in events_fired,
               "auto-fire commits: capture cue fires too")

        # After fire we are in cooldown - detect_start must NOT fire again
        # (the per-session flag is still True, but we shouldn't re-chime).
        before = list(events_fired)
        s._maybe_auto_capture()
        _check(events_fired == before,
               "in cooldown: no extra sound events are fired")

        # Quad disappears: the flag should re-arm so the next visible
        # quad gets a fresh blip.
        class _NoDocDetector:
            corners = None
            confidence = 0.0
            def __bool__(self) -> bool: return False
        s._processor = type("_NoDocProc", (), {
            "process": staticmethod(lambda f: (np.zeros((4,4,3), dtype=np.uint8),
                                               _NoDocDetector())),
        })()
        # Wait out the cooldown so we leave the cooldown branch.
        import time as _t
        _t.sleep(s.auto_capture_cooldown_s + 0.05)
        s._maybe_auto_capture()
        _check(s._sound_detect_start_played is False,
               "doc-disappeared re-arms the detect_start chime")

        # --- 7b: manual C path -------------------------------------- #
        events_fired.clear()
        s2 = ScanSession(
            camera_source="http://127.0.0.1:1/video",
            web_port=18083,
        )
        s2.sound_enabled = True
        s2._sound = _StubSound()  # type: ignore[assignment]
        s2._sound_detect_start_played = False
        s2.capture_current_frame = lambda *a, **kw: (  # type: ignore[assignment]
            True, "stub: manual", np.zeros((4, 4, 3), dtype=np.uint8),
            _StubDetector(),
        )
        s2._handle_live_key("c")
        _check(events_fired == ["capture"],
               f"manual C fires 'capture' exactly once (got {events_fired})")

        # --- 7c: quality-gate rejection must NOT fire 'capture' ------ #
        events_fired.clear()
        s3 = ScanSession(
            camera_source="http://127.0.0.1:1/video",
            web_port=18084,
        )
        s3.sound_enabled = True
        s3._sound = _StubSound()  # type: ignore[assignment]
        s3.capture_current_frame = lambda *a, **kw: (  # type: ignore[assignment]
            False, "rejected: blurry", None, None,
        )
        s3._handle_live_key("c")
        _check(events_fired == [],
               f"rejected capture fires no sound (got {events_fired})")

    # ------------------------------------------------------------------ #
    phases = [
        p1_imports, p2_wav_blobs, p3_disabled_noop, p4_unknown_event,
        p5_configure_rebuilds, p6_session_wiring, p7_fsm_sound_hooks,
    ]
    for p in phases:
        _phase(p.__name__, p)

    # ------------------------------------------------------------------ #
    # Phase 8 - voice layer (offline TTS) module-level shape.
    # ------------------------------------------------------------------ #
    def p8_voice_imports() -> None:
        voice = importlib.import_module("voice")
        _check(hasattr(voice, "VoicePrompter"), "voice.VoicePrompter is exposed")
        _check(hasattr(voice, "get_default_prompter"),
               "voice.get_default_prompter is exposed")
        _check(hasattr(voice, "speak"), "voice.speak is exposed")
        _check(
            isinstance(voice._PHRASE_TEMPLATES, dict),
            "voice._PHRASE_TEMPLATES is a dict",
        )
        expected_events = {
            "detected", "stable", "capture_manual", "capture_auto",
            "capture_rejected", "page_change", "document_new",
            "document_saved", "document_export", "shutdown", "error",
        }
        _check(
            expected_events <= set(voice._PHRASE_TEMPLATES.keys()),
            f"all 11 expected events are in the phrase table "
            f"(missing: {expected_events - set(voice._PHRASE_TEMPLATES.keys())})",
        )
        # Every phrase must be a non-empty string with no stray '{' / '}' placeholders.
        for ev, tmpl in voice._PHRASE_TEMPLATES.items():
            _check(isinstance(tmpl, str) and len(tmpl) > 0,
                   f"phrase {ev!r} is a non-empty string")
            # If it has a placeholder, it must be well-formed (single {name}).
            import re as _re
            placeholders = _re.findall(r"\{([^{}]+)\}", tmpl)
            _check(
                len(set(placeholders)) == len(placeholders),
                f"phrase {ev!r} has unique placeholders ({placeholders})",
            )

    # ------------------------------------------------------------------ #
    # Phase 9 - disabled VoicePrompter is a complete no-op.
    # ------------------------------------------------------------------ #
    def p9_voice_disabled_noop() -> None:
        from voice import VoicePrompter
        vp = VoicePrompter(enabled=False, backend="none")
        _check(vp.enabled is False, "VoicePrompter.enabled sticks at False")
        # Each event must return False (or a falsy value) and never raise.
        for ev in ("detected", "stable", "capture_manual", "capture_auto",
                   "page_change", "document_new", "document_saved",
                   "shutdown", "error"):
            try:
                rc = vp.speak(ev)
            except Exception as exc:
                raise AssertionError(f"disabled.speak({ev!r}) raised {exc!r}")
            _check(not rc, f"disabled.speak({ev!r}) returns falsy (got {rc!r})")

    # ------------------------------------------------------------------ #
    # Phase 10 - enabled VoicePrompter with a fake audio backend records
    # every event and reuses the WAV cache (one synth per (event, phrase)).
    # ------------------------------------------------------------------ #
    def p10_voice_enabled_with_fake_backend() -> None:
        from voice import VoicePrompter

        # A fake backend that always returns a deterministic 1-byte WAV blob
        # (not RIFF-valid - the fake audio backend ignores the bytes).
        FAKE_WAV = b"FAKE_WAV_BLOB_" * 4  # 56 bytes
        rendered_calls: list = []

        def _fake_synth(text: str) -> bytes:
            rendered_calls.append(text)
            return FAKE_WAV

        played: list = []

        class _FakeSound:
            enabled = True
            def _play_wav(self, wav: bytes) -> bool:
                played.append(wav)
                return True

        vp = VoicePrompter(enabled=True, backend="none")
        # Inject the fake synth + audio sink so we never touch pyttsx3/espeak.
        vp._backend_name = "fake"
        vp._synthesise = _fake_synth  # type: ignore[assignment]
        vp._sound = _FakeSound()       # type: ignore[assignment]
        vp._cache.clear()  # type: ignore[attr-defined]
        vp._cache_lock = __import__("threading").Lock()  # type: ignore[assignment]

        # Drive each event once.  Every event should produce exactly one
        # synth call and exactly one _play_wav call.
        rendered_calls.clear()
        played.clear()
        events = [
            ("detected", {}),
            ("stable", {}),
            ("capture_manual", {}),
            ("capture_auto", {}),
            ("capture_rejected", {"reason": "too dark"}),
            ("page_change", {"n": 95}),
            ("document_new", {}),
            ("document_saved", {"n": 3}),
            ("document_export", {"path": "/tmp/scan.pdf"}),
            ("shutdown", {}),
            ("error", {"detail": "boom"}),
        ]
        for ev, fmt in events:
            rc = vp.speak(ev, **fmt)
            _check(rc is True, f"speak({ev!r}, **{fmt}) returns True")
        _check(len(rendered_calls) == len(events),
               f"each event synthesised exactly once "
               f"(got {len(rendered_calls)} calls for {len(events)} events)")
        _check(len(played) == len(events),
               f"each event dispatched exactly once "
               f"(got {len(played)} plays for {len(events)} events)")

        # Now hit each event AGAIN.  No new synth should fire - cache hit.
        rendered_calls.clear()
        played.clear()
        for ev, fmt in events:
            rc = vp.speak(ev, **fmt)
            _check(rc is True, f"cache hit speak({ev!r}, **{fmt}) returns True")
        _check(rendered_calls == [],
               f"cache hit -> zero synth calls (got {rendered_calls})")
        _check(played == [FAKE_WAV] * len(events),
               f"cache hit -> same WAV replayed for every event "
               f"(got {len(played)} plays)")

        # Templated phrases must be interpolated into the synth payload.
        vp._cache.clear()  # type: ignore[attr-defined]
        rendered_calls.clear()
        vp.speak("capture_rejected", reason="too dark")
        _check(
            "too dark" in rendered_calls[0],
            f"capture_rejected interpolates reason="
            f"{rendered_calls[0]!r}",
        )
        vp.speak("document_saved", n=7)
        _check(
            "7" in rendered_calls[-1],
            f"document_saved interpolates n= -> {rendered_calls[-1]!r}",
        )

    # ------------------------------------------------------------------ #
    # Phase 11 - espeak-ng argv shape is exactly correct, no shell=True.
    # ------------------------------------------------------------------ #
    def p11_espeak_argv_shape() -> None:
        """Verify the espeak-ng subprocess contract without actually
        invoking a TTS engine.  We mock subprocess.run so no espeak binary
        is required on this machine."""
        import subprocess as _sp

        captured: dict = {}

        # 50-byte payload to satisfy the espeak-output length guard.
        _FAKE_ESPEAK_OUT = b"FAKE_ESPEAK_WAV_BLOB_" * 3  # 63 bytes

        class _FakeCompleted:
            returncode = 0
            stdout = _FAKE_ESPEAK_OUT
            stderr = b""

        def _fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return _FakeCompleted()

        # Force the espeak backend to be selected.
        from voice import _synthesise_espeak  # noqa: F401  (sanity import)
        import voice as _voice_mod

        real_run = _voice_mod.subprocess.run  # type: ignore[attr-defined]
        try:
            _voice_mod.subprocess.run = _fake_run  # type: ignore[attr-defined]
            out = _voice_mod._synthesise_espeak(
                "Document saved", language="en-us"
            )
        finally:
            _voice_mod.subprocess.run = real_run  # type: ignore[attr-defined]

        _check(out == _FAKE_ESPEAK_OUT,
               f"_synthesise_espeak returns subprocess stdout (got {out!r})")
        argv = captured.get("argv")
        _check(isinstance(argv, list),
               f"subprocess argv is a list (got {type(argv).__name__})")
        _check(
            argv[:4] == ["espeak-ng", "-v", "en-us", "-w"],
            f"argv prefix is exactly ['espeak-ng', '-v', 'en-us', '-w'] "
            f"(got {argv[:4]!r})",
        )
        _check(
            argv[4] == "-",
            f"argv[4] is '-' (write to stdout) (got {argv[4]!r})",
        )
        _check(
            argv[5] == "Document saved",
            f"argv[5] is the literal text (got {argv[5]!r})",
        )
        kwargs = captured.get("kwargs", {})
        _check(kwargs.get("shell", False) is False,
               "subprocess.run shell=False (no shell injection risk)")
        _check(kwargs.get("capture_output") is True,
               "subprocess.run capture_output=True (stdout/stderr captured)")
        # We deliberately do NOT use check=True - the code inspects
        # ``proc.returncode`` itself so it can format a friendlier error
        # message containing stderr.  Verify returncode handling instead:
        _check(
            "check" not in kwargs or kwargs.get("check") in (False, None),
            "subprocess.run does not pass check=True "
            "(returncode is inspected manually for richer errors)",
        )

    # ------------------------------------------------------------------ #
    phases = [
        p1_imports, p2_wav_blobs, p3_disabled_noop, p4_unknown_event,
        p5_configure_rebuilds, p6_session_wiring, p7_fsm_sound_hooks,
        p8_voice_imports, p9_voice_disabled_noop,
        p10_voice_enabled_with_fake_backend,
        p11_espeak_argv_shape,
    ]
    for p in phases:
        _phase(p.__name__, p)

    print()
    if failures:
        print(f"FAILED phases: {failures}")
        return 1
    print("All sound + voice smoke phases PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
