"""Probe the post-fix voice pipeline by counting _play_wav invocations.

Run with::  .venv\\Scripts\\python.exe runs\\probe_voice_regression.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import voice  # noqa: E402
import sound  # noqa: E402

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")

# Wrap SoundPlayer._play_wav with a counter so we can see whether the worker
# actually dispatched the WAV bytes.
play_count = {"n": 0}
original = sound.SoundPlayer._play_wav

def _wrapped(self, wav_bytes):
    play_count["n"] += 1
    print(f"    [PROBE] _play_wav called  -> total={play_count['n']}", flush=True)
    return original(self, wav_bytes)

sound.SoundPlayer._play_wav = _wrapped

vp = voice.VoicePrompter(enabled=True, rate_wpm=165)

events = [
    ("capture_manual",   {}),
    ("capture_auto",     {}),
    ("page_change",      {"n": 2}),
    ("document_saved",   {"n": 3}),
    ("document_saved",   {"n": 1}),
]

for evt, fmt in events:
    phrase = vp.phrase(evt, **fmt)
    print(f"[PROBE] speak({evt!r}, {fmt}) -> phrase={phrase!r}", flush=True)
    vp.speak(evt, **fmt)

# Wait for the worker to drain.
time.sleep(15.0)
vp.shutdown()

print(f"\nTOTAL PLAYS: {play_count['n']}", flush=True)
sys.exit(0 if play_count["n"] == len(events) else 1)