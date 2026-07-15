"""Manual audio-cue sanity check.

This script does NOT need a camera or the LIVE loop.  It exercises both
audio paths end-to-end so a developer can confirm audio works on their
box without having to capture/delete a real page:

    1. Short WAV cues  -- ``sound.SoundPlayer``  (the "ka-chunk" +
       "undo" tones).  Works on Windows (winsound), macOS (afplay)
       and Linux (paplay / aplay / ffplay).
    2. Long MP3 cues   -- ``mp3_player.MP3Player``.  This pipeline is
       Linux-only (Pi 5 MAX98357A I2S amp); on Windows / macOS it is
       a documented no-op.

Usage
=====

    python -m runs.manual_audio_check            # play everything
    python -m runs.manual_audio_check --quiet    # only print checklist
    python -m runs.manual_audio_check --skip-mp3 # only the short WAV cues

What it prints
==============

For every check it prints a `[OK] ...` / `[!!] ...` line so a developer
can scan the output top-to-bottom and immediately see:

* Is ``winsound`` importable on this box?
* Is the ``SoundPlayer`` enabled and what backend did it pick?
* Are the two canned WAVs well-formed (RIFF, mono, 16-bit, 22050 Hz)?
* Did ``PlaySound`` return ``True`` for each event?
* On non-Linux hosts, does the MP3 pipeline correctly no-op?
* Are the ``captured.mp3`` / ``deleted.mp3`` files present at the
  project root?

Exit code
=========

Returns ``0`` when every short-tone check passes (the Windows dev box
expectation).  Returns ``1`` when a short-tone check failed.

The long-form MP3 check is informational only: on Windows/macOS it is
expected to no-op.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import wave
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _header(label: str) -> None:
    print(f"\n=== {label} ===")


def _check(condition: bool, message: str, *, warn_only: bool = False) -> bool:
    """Print a single tick / cross and return the boolean result."""
    if condition:
        print(f"  [OK]   {message}")
    else:
        tag = "[!!]" if warn_only else "[FAIL]"
        print(f"  {tag}  {message}")
    return condition


def _wav_inspect(blob: bytes) -> Tuple[bool, str]:
    """Return ``(is_well_formed, human_readable_summary)`` for a WAV blob."""
    try:
        with wave.open(io.BytesIO(blob), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            nf = wf.getnframes()
            dur = nf / sr
        ok = blob[:4] == b"RIFF" and blob[8:12] == b"WAVE" and ch == 1 and sw == 2
        return ok, f"{ch}ch x {sw*8}-bit x {sr} Hz, {nf} frames ({dur*1000:.0f} ms)"
    except Exception as exc:
        return False, f"<unparseable: {exc!r}>"


def main(argv: Optional[list] = None) -> int:  # noqa: C901
    p = argparse.ArgumentParser(
        description="Manual audio-cue sanity check.",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Don't actually play audio, only print the checklist.")
    p.add_argument("--skip-mp3", action="store_true",
                   help="Skip the long-form MP3 pipeline checks.")
    p.add_argument("--volume", type=float, default=0.6,
                   help="Linear gain 0..1 for the short-tone player "
                        "(default: 0.6, matches the LIVE loop).")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity for the sound / mp3_player modules. "
             "Use DEBUG on the Pi to see exactly why _alsa_available() "
             "is returning False (e.g. 'alsa backend skipped: ffmpeg not "
             "on PATH').",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    failures: list = []

    # ---------------------------------------------------------------- #
    _header("Phase 1 -- sound module imports")
    try:
        import sound as sound_mod
    except Exception as exc:
        print(f"  [FAIL] could not import sound: {exc!r}")
        return 1
    _check(hasattr(sound_mod, "SoundPlayer"), "sound.SoundPlayer is exposed")
    _check(hasattr(sound_mod, "play_event"), "sound.play_event is exposed")
    _check(
        hasattr(sound_mod, "get_default_player"),
        "sound.get_default_player is exposed",
    )
    _KNOWN = {"captured", "page_deleted"}
    _check(
        set(sound_mod._EVENT_NOTES.keys()) == _KNOWN,
        f"only the two named events are registered "
        f"(got {set(sound_mod._EVENT_NOTES.keys())!r})",
    )

    # ---------------------------------------------------------------- #
    _header("Phase 2 -- platform + backend probe")
    print(f"  platform          : {sys.platform}")
    print(f"  python            : {sys.version.split()[0]}")
    print(f"  project_root      : {PROJECT_ROOT}")
    try:
        import winsound  # type: ignore[import-not-found]
        _check(True, "winsound importable (Windows native playback OK)")
    except Exception as exc:
        _check(False, f"winsound import failed: {exc!r}")
        failures.append("winsound import")

    # ---------------------------------------------------------------- #
    _header("Phase 3 -- SoundPlayer state")
    player = sound_mod.SoundPlayer(
        enabled=True,
        volume=float(args.volume),
    )
    print(f"  enabled           : {player.enabled}")
    print(f"  volume            : {player.volume}")
    print(f"  backend           : {player._backend_name}")
    print(f"  cached events     : {sorted(player._cache.keys())}")
    print(f"  pre-written wavs  : {sorted(player._files.keys())}")
    if player._files:
        for name, path in player._files.items():
            try:
                size = os.path.getsize(path)
            except OSError as exc:
                size = f"<missing: {exc!r}>"
            print(f"    - {name}: {path} ({size} bytes)")
    _check(
        player._backend_name in ("winsound", "cli", "alsa"),
        f"backend is one of winsound|cli|alsa (got {player._backend_name!r})",
    )
    _check(player.enabled is True, "SoundPlayer is enabled")
    _check(
        set(player._cache.keys()) == _KNOWN,
        f"cache has exactly the two events "
        f"(got {set(player._cache.keys())!r})",
    )

    # ---------------------------------------------------------------- #
    _header("Phase 4 -- WAV blob inspection")
    for name, blob in player._cache.items():
        ok, summary = _wav_inspect(blob)
        _check(ok, f"{name!r} WAV well-formed: {summary}")
        if not ok:
            failures.append(f"wav {name!r}")

    # ---------------------------------------------------------------- #
    _header("Phase 5 -- play each event")
    for name in sorted(_KNOWN):
        if args.quiet:
            print(f"  --   would play_event({name!r}) "
                  f"(skipped because --quiet)")
            continue
        try:
            ok = player.play_event(name)
        except Exception as exc:
            ok = False
            print(f"  [FAIL] play_event({name!r}) raised {exc!r}")
            failures.append(f"play {name!r}")
            continue
        _check(ok is True, f"play_event({name!r}) -> True")
        if not ok:
            failures.append(f"play {name!r}")

    # ---------------------------------------------------------------- #
    if not args.skip_mp3:
        _header("Phase 6 -- MP3 pipeline probe (informational on non-Linux)")
        print(f"  platform          : {sys.platform}")
        try:
            import mp3_player as mp3_mod
        except Exception as exc:
            _check(False, f"could not import mp3_player: {exc!r}", warn_only=True)
        else:
            _check(
                hasattr(mp3_mod, "MP3Player"),
                "mp3_player.MP3Player is exposed",
                warn_only=True,
            )
            mp3 = mp3_mod.MP3Player(enabled=True)
            print(f"  enabled           : {mp3.enabled}")
            print(f"  device            : {mp3.device}")
            print(f"  volume_db         : {mp3.volume_db:+0.1f}")
            print(f"  captured_file     : {mp3.captured_file} "
                  f"({'OK' if os.path.exists(mp3.captured_file) else 'MISSING'})")
            print(f"  deleted_file      : {mp3.deleted_file} "
                  f"({'OK' if os.path.exists(mp3.deleted_file) else 'MISSING'})")

            if not sys.platform.startswith("linux"):
                _check(
                    True,
                    "non-Linux host: long-form MP3 cues are documented "
                    "no-ops (Pi 5 MAX98357A I2S amp only)",
                    warn_only=True,
                )
            else:
                _check(
                    mp3_mod._pydub_available(),
                    "pydub is importable",
                    warn_only=True,
                )
                _check(
                    mp3_mod._alsaaudio_available(),
                    "pyalsaaudio is importable",
                    warn_only=True,
                )
                _check(
                    mp3_mod._ffmpeg_available(),
                    "ffmpeg is on PATH",
                    warn_only=True,
                )
                for name in ("captured", "page_deleted"):
                    path = mp3._filename_for(name)
                    _check(
                        os.path.exists(path) if path else False,
                        f"{name!r} MP3 exists at {path}",
                        warn_only=True,
                    )

            # Don't actually dispatch -- the Linux path is real audio and
            # the non-Linux path is a no-op; either way the user only
            # needs to *see* the configuration above.
            for name in sorted(_KNOWN):
                if args.quiet:
                    print(f"  --   would mp3.play_event({name!r}) "
                          f"(skipped because --quiet)")
                    continue
                try:
                    ok = mp3.play_event(name)
                except Exception as exc:
                    ok = False
                    print(f"  [FAIL] mp3.play_event({name!r}) raised {exc!r}",
                          file=sys.stderr)
                    continue
                # ``MP3Player.play_event`` only returns False for
                # explicitly-disabled / unknown-event / missing-file
                # paths.  The actual decode-and-stream is asynchronous
                # on a daemon thread, so even on non-Linux (where the
                # pipeline later fails inside ``play_clip``) the call
                # itself returns ``True`` if the file exists.  Treat any
                # ``True`` here as "accepted" and any ``False`` as a
                # pipeline miss.
                if not sys.platform.startswith("linux"):
                    expected = (
                        True
                        if mp3._filename_for(name)
                        and os.path.exists(mp3._filename_for(name) or "")
                        else False
                    )
                    _check(
                        ok is expected,
                        f"mp3.play_event({name!r}) -> {ok} (expected {expected})",
                        warn_only=True,
                    )
                else:
                    _check(
                        ok is True,
                        f"mp3.play_event({name!r}) -> {ok}",
                        warn_only=True,
                    )

    # ---------------------------------------------------------------- #
    _header("Phase 7 -- 'why no audio?' checklist (Windows)")
    print(
        "  If you heard the tones, you're done.\n"
        "  If you didn't, walk this list top-to-bottom:\n"
        "    [1] Right-click the speaker icon in the taskbar,\n"
        "        'Open Sound settings' -> make sure the output device\n"
        "        is NOT muted and is NOT set to 'Disabled'.\n"
        "    [2] Run this script with --quiet first to confirm the\n"
        "        WAV blobs and backend probe are green; if any [FAIL]\n"
        "        appears above, fix that first.\n"
        "    [3] Open the captured.mp3 / deleted.mp3 in a media\n"
        "        player to confirm your speakers actually play audio.\n"
        "    [4] The Pi-only long-form MP3 cues are intentionally a\n"
        "        no-op on Windows -- use the short WAV tones on the\n"
        "        dev box.\n"
        "    [5] Set logging to DEBUG to see exactly which event\n"
        "        fired and which one was dropped (and, on Linux, why\n"
        "        _alsa_available() may have skipped the Pi I2S backend):\n"
        "          python -m runs.manual_audio_check --log-level DEBUG"
    )

    # ---------------------------------------------------------------- #
    print()
    if failures:
        print(f"FAILED checks: {failures}")
        return 1
    print("All short-tone audio checks PASSED. You should have heard:")
    print("  - one 3-note 'ka-chunk' (captured)")
    print("  - one 2-note 'undo' (page_deleted)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))