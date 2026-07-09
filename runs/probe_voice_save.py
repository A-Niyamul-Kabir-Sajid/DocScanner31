"""Time each phrase so we can see if document_saved is too slow / silenced.

Captures timestamps:
  enqueue      = when speak() returned True (phrase pushed to worker)
  synth_done   = when worker called _dispatch(wav)
  play_done    = when _play_wav was invoked
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

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

# Wrap _play_wav to timestamp every dispatch.
play_log = []
original = sound.SoundPlayer._play_wav

def _wrapped(self, wav_bytes, *, event_name=None):
    play_log.append(time.monotonic())
    print(f"    [PLAY t+{play_log[-1]-t0:5.2f}s] bytes={len(wav_bytes):>6}  "
          f"event={event_name!r}", flush=True)
    return original(self, wav_bytes, event_name=event_name)

sound.SoundPlayer._play_wav = _wrapped

t0 = time.monotonic()
vp = voice.VoicePrompter(enabled=True, rate_wpm=165)
print(f"[t+{time.monotonic()-t0:.2f}s] VoicePrompter constructed", flush=True)

# Wait for prewarm thread to finish initialising the engine.
time.sleep(3.0)
print(f"[t+{time.monotonic()-t0:.2f}s] prewarm window done; engine = "
      f"{type(voice._pyttsx3_engine).__name__}", flush=True)

# Simulate the save flow as it happens in app.py:
#   line 986: saved = self.finish_pdf()
#   line 989: self.speak("document_saved", n=self.page_count())
print(f"[t+{time.monotonic()-t0:.2f}s] >>> speak('document_saved', n=3) <<<",
      flush=True)
vp.speak("document_saved", n=3)

# Wait long enough for the worker to drain.
time.sleep(8.0)
vp.shutdown()
print(f"\nTOTAL DISPATCHED: {len(play_log)}", flush=True)