# Audio Playback — `sound.py`

The Smart Document Scanner exposes exactly two audio cues. Both are wired through a
single `SoundPlayer` class so the LIVE FSM never blocks on the speaker.

## Cue catalogue

| Event name      | Notes (`(freq_hz, duration_s)`)              | When it fires                                  |
|-----------------|----------------------------------------------|------------------------------------------------|
| `captured`      | `(880.0, 0.07), (660.0, 0.07), (523.0, 0.14)` | Page committed to the PDF (auto or manual)     |
| `page_deleted`  | `(740.0, 0.08), (523.0, 0.14)`                | Most-recent page dropped via the `X` key       |

Defined at `sound.py:175-180` in the `_EVENT_NOTES` dict.

Earlier `detect_start` / `detect_stable` chimes have been removed; no other events
play any sound.

## Entry points

| Caller                                | Call                                      | File / line        |
|---------------------------------------|-------------------------------------------|--------------------|
| `ScanSession._on_capture_committed`   | `self._play_sound("captured")`            | `app.py`           |
| `ScanSession._on_page_deleted`        | `self._play_sound("page_deleted")`        | `app.py`           |
| Module-level convenience               | `sound.play_event(name)`                  | `sound.py:849-851` |
| Process-wide lazy instance            | `sound.get_default_player()`              | `sound.py:841-846` |

`ScanSession._play_sound` is a defensive wrapper: it delegates to
`self.sound.play_event(event)` inside a `try/except` so a broken audio backend
never breaks the LIVE loop.

## Internal flow of `play_event(name)` (`sound.py:472-509`)

1. **Enabled gate** — `if not self.enabled: return False` (`sound.py:485`).
2. **MP3-first lookup** — `_ensure_mp3_pcm(name)` (`sound.py:412-449`):
   - If `configure_mp3(event, path)` registered an MP3, decode it once via
     `pydub.AudioSegment.from_mp3(...)` (`sound.py:384-410`), cache the
     resulting `(sample_rate, pcm_bytes)`, and reuse it on every subsequent
     call. No per-press decode latency.
   - If the file is missing or decode fails, fall through to the procedural
     tone (warn-once, then DEBUG).
3. **Procedural tone fallback** — `self._cache.get(name)` (`sound.py:502`).
   WAV blobs are pre-rendered in the constructor so the LIVE hot path is
   allocation-free.
4. **Dispatch** — call `_play_wav(wav, event_name=name)` for the tone, or
   `_play_pcm(raw, sample_rate=..., event_name=name)` for the decoded MP3.
5. If neither path hits, log at DEBUG and return `False` — silent no-op,
   never a crash.

## Synthesis (`sound.py:125-167`)

- `_sine(freq, dur, ...)` — generates 16-bit mono PCM bytes with a 5 ms linear
  attack/release envelope at both ends to avoid clicks (`sound.py:131-141`).
- `_silence(dur, ...)` — zeroed PCM for rests (`freq <= 0`).
- `_build_wav(notes, ...)` — wraps the concatenated samples in a RIFF/WAVE
  header via `wave.open(...)`.

All WAV blobs are pre-rendered in `SoundPlayer.__init__`
(`sound.py:220-223`) into `self._cache`.

## Backend selection (`sound.py:292-324`)

One of four outcomes at construction time:

| Backend    | When chosen                                                          | Dispatch site                  |
|------------|----------------------------------------------------------------------|--------------------------------|
| `winsound` | Windows + `winsound` importable                                      | `_play_wav` / `_play_pcm` winsound arm |
| `cli`      | Linux/macOS with `afplay` / `paplay` / `aplay` / `ffplay` on `$PATH` | `_play_wav` / `_play_pcm` cli arm      |
| `alsa`     | Linux + `pyalsaaudio` + `pydub` + `ffmpeg` (Pi 5 I2S path)           | `_play_wav` / `_play_pcm` alsa arm     |
| `none`     | None of the above                                                    | Silent no-op                   |

`prefer="winsound"|"cli"|"alsa"|"auto"` lets tests force a specific backend.
Default is `"auto"`. On Windows, `__init__` calls `_write_wav_files()`
(`sound.py:273-289`) to materialise every cached WAV to a temp file because
`winsound.SND_MEMORY + SND_ASYNC` raises — temp files are tracked in
`self._files` for cleanup.

## Dispatch paths

### `_play_wav(wav, event_name=...)` (`sound.py:512-660`) — procedural tone

- **`winsound`** (`sound.py:525-552`) — must play from a file path. Uses the
  pre-written temp file for canned events, or writes a fresh scratch file
  (tracked in `self._scratch`) for TTS-style calls. Calls
  `winsound.PlaySound(path, SND_FILENAME|SND_ASYNC|SND_NODEFAULT)`.
- **`cli`** (`sound.py:553-574`) — temp WAV + `subprocess.Popen(argv + [path])`
  on a daemon thread; cleaned up after the OS finishes playing.
- **`alsa`** (`sound.py:575-655`) — `pydub.AudioSegment.from_wav(...)` →
  `alsaaudio.PCM(device="plughw:2,0", mode=NORMAL)` → chunked `pcm.write(...)`
  on a daemon thread named `sound-alsa-<event>`.

### `_play_pcm(raw, sample_rate, event_name=...)` (`sound.py:663-832`) — MP3

Same three arms, but the bytes are already decoded so the `pydub` round-trip
is skipped:

- **`winsound`** (`sound.py:701-732`) — wraps raw PCM in a fresh WAV header in
  a temp file, then plays.
- **`cli`** (`sound.py:733-763`) — wraps + `Popen` on a daemon thread.
- **`alsa`** (`sound.py:764-827`) — opens `alsaaudio.PCM(...)` directly,
  writes in `alsa_chunk_bytes` (default 4096) slices.

All six arms are **non-blocking**; the LIVE FSM never waits on audio.

## Concurrency — per-event ALSA locks (`sound.py:235-237`)

`plughw:2,0` is **exclusive** on Linux. Two simultaneous
`alsaaudio.PCM(device=...)` calls raise `Device or resource busy`.
`SoundPlayer` solves this exactly the same way `MP3Player` does:

- One `threading.Lock` per event name, created in `__init__`.
- Each ALSA branch does `lock.acquire(blocking=False)`. If a previous clip is
  still streaming → drop the new one (`return True` to keep the no-op
  indistinguishable on the outside).
- The lock is released in the daemon-thread `finally` block **after
  `pcm.close()`** (`sound.py:641-649` and `sound.py:813-821`), so the next
  `play_event()` always finds the device free.

For TTS-style calls (no `event_name`), a shared `_scratch_lock` is created
lazily on first use.

## MP3 registration (`sound.py:356-382`)

```python
sound_player.configure_mp3("captured", "/path/to/captured.mp3")
sound_player.configure_mp3("page_deleted", "/path/to/deleted.mp3")
```

`path=None` clears the registration; empty string raises `ValueError`.
A new path invalidates the cached PCM so the next press re-decodes.

## Runtime configuration (`sound.py:327-354`)

- `configure(enabled=...)` — global kill-switch (also wired to `--no-sound`).
- `configure(volume=...)` — re-renders **every** cached WAV blob, wipes
  on-disk temp files, and re-writes them.
- `configure(alsa_device=..., alsa_chunk_bytes=...)` — I2S path tuning.
- `configure(mp3_volume_db=...)` — invalidates `_pcm_cache` because the old
  gain is baked into the cached bytes.

## Cleanup (`sound.py:451-469`)

- `_cleanup_files()` deletes every temp WAV and scratch PCM.
- `close()` calls `_cleanup_files()` and is safe to invoke multiple times.

## Where audio is silent

- **No chime on stability detection** — the FSM only plays sound on a
  successful capture or a successful delete.
- **No chime on `no_match_timer` expiry (S2 → S1 transition)** — the timer
  is silent.
- **No chime on manual `c` keystroke vs auto-capture** — both paths play
  the same `captured` cue.

To add a new cue (e.g. an "S2 released" chime), the surgical change is:

1. Add a row to `_EVENT_NOTES` at `sound.py:175-180`.
2. Call `self._play_sound("new_event")` from the relevant site — for a
   State-2-release cue, that would be `auto_capture_controller.py:_flip_to_state1`
   (`auto_capture_controller.py:333-347`).
