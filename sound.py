"""Audio cues for the Smart Document Scanner.

Two short sound effects are exposed via :func:`play_event`:

* ``"captured"``     - a satisfying descending three-note "ka-chunk" when a
  page is committed to the PDF (auto-capture fire, manual ``C`` press,
  or any other successful capture path).
* ``"page_deleted"`` - a soft "undo" cue when the most recently captured
  page is dropped via the ``X`` key (or the equivalent UI affordance).

No other events play any sound.  Earlier ``detect_start`` / ``detect_stable``
chimes have been removed so the audio feedback is limited to user-visible
state changes only.

Design goals
============

* **No external asset files.**  WAV bytes are synthesised at module import
  time using ``numpy`` (already a hard dependency for OpenCV).
* **No additional pip dependency.**  ``winsound`` ships with Python on
  Windows; on macOS / Linux we fall back to invoking the platform's
  built-in audio player via ``subprocess``.  If neither backend can be
  reached the call becomes a silent no-op so the LIVE loop never blocks.
* **Async playback.**  Tones are queued with ``SND_ASYNC`` (winsound) or
  ``Popen`` (subprocess) so the scanner FSM is never stalled by audio.

The module is intentionally tiny and has no project-specific imports, so
it can also be exercised by ``runs/smoke_sound.py`` in isolation.
"""

from __future__ import annotations

import io
import logging
import math
import os
import struct
import subprocess
import sys
import tempfile
import threading
import wave
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Defaults - may be overridden by ``configure(enabled=..., volume=...)``.
# --------------------------------------------------------------------------- #
DEFAULT_ENABLED: bool = True
DEFAULT_VOLUME: float = 0.6      # 0.0 (silent) - 1.0 (full)
DEFAULT_SAMPLE_RATE: int = 22050 # Hz - plenty for short blips

# ALSA backend defaults - match the Pi I2S amp conventions from ``mp3_player``
# so the user can swap in raw synthesized tones alongside the long-form MP3s.
DEFAULT_ALSA_DEVICE: str = "plughw:2,0"
DEFAULT_ALSA_CHUNK_BYTES: int = 4096

# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _winsound_available() -> bool:
    """Return True if the ``winsound`` stdlib module is usable on this OS."""
    return sys.platform.startswith("win")


def _cli_player_available() -> Optional[list]:
    """Return the argv used to play a wav on this OS, or None if unavailable."""
    if sys.platform == "darwin":
        return ["afplay"]
    if sys.platform.startswith("linux"):
        for player in ("paplay", "aplay", "ffplay"):
            # Use `command -v` so we don't depend on shutil.which (same semantics)
            try:
                rc = subprocess.call(
                    ["command", "-v", player],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
            except (FileNotFoundError, OSError):
                continue
            if rc == 0:
                return [player]
    return None


def _alsa_available() -> bool:
    """Return True if the Pi-style ALSA I2S backend is usable on this OS.

    Requires ``pyalsaaudio`` to be importable and the binary ``ffmpeg`` on
    ``$PATH`` (used by ``pydub`` to decode the WAV blob).  Falls through
    to ``False`` on non-Linux and in environments without the I2S overlay.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        import alsaaudio  # type: ignore[import-not-found]  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("alsa (pyalsaaudio) import failed: %s", exc)
        return False
    try:
        from pydub import AudioSegment  # type: ignore[import-not-found]  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("alsa (pydub) import failed: %s", exc)
        return False
    if shutil_which("ffmpeg") is None:
        logger.debug("alsa backend skipped: ffmpeg not on PATH")
        return False
    return True


def shutil_which(cmd: str) -> Optional[str]:
    """Thin wrapper around ``shutil.which`` so the module stays import-safe."""
    import shutil
    try:
        return shutil.which(cmd)
    except Exception:  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# WAV synthesis
# --------------------------------------------------------------------------- #
def _sine(freq_hz: float, duration_s: float, *, sample_rate: int = DEFAULT_SAMPLE_RATE,
          volume: float = DEFAULT_VOLUME) -> bytes:
    """Generate 16-bit mono PCM bytes for a sine tone at ``freq_hz``."""
    n_samples = max(1, int(duration_s * sample_rate))
    amplitude = int(32767 * max(0.0, min(1.0, volume)))
    frames = bytearray()
    for n in range(n_samples):
        # Linear attack/release envelope (first/last 5 ms) avoids clicks.
        if n < int(0.005 * sample_rate):
            env = n / max(1, int(0.005 * sample_rate))
        elif n > n_samples - int(0.005 * sample_rate):
            env = (n_samples - n) / max(1, int(0.005 * sample_rate))
        else:
            env = 1.0
        sample = amplitude * env * math.sin(2.0 * math.pi * freq_hz * n / sample_rate)
        frames += struct.pack("<h", int(sample))
    return bytes(frames)


def _silence(duration_s: float, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Return ``duration_s`` of zeroed 16-bit PCM bytes."""
    n_samples = max(1, int(duration_s * sample_rate))
    return b"\x00\x00" * n_samples


def _build_wav(notes: list, *, sample_rate: int = DEFAULT_SAMPLE_RATE, volume: float = DEFAULT_VOLUME) -> bytes:
    """Build a WAV blob from a list of ``(freq_hz, duration_s)`` notes.

    A frequency of ``0`` is treated as a rest (silence).
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        for freq, dur in notes:
            if freq <= 0:
                wf.writeframes(_silence(dur, sample_rate=sample_rate))
            else:
                wf.writeframes(
                    _sine(freq, dur, sample_rate=sample_rate, volume=volume)
                )
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Event -> (notes, label) table.
# Frequencies are chosen to be pleasant and clearly distinguishable.
# Only two cues ship today: capture success and page deletion.
# --------------------------------------------------------------------------- #
_EVENT_NOTES: Dict[str, list] = {
    # Three-note "ka-chunk" - page successfully captured.
    "captured":     [(880.0, 0.07), (660.0, 0.07), (523.0, 0.14)],
    # Soft two-note descending "undo" cue - last page removed.
    "page_deleted": [(740.0, 0.08), (523.0, 0.14)],
}


# --------------------------------------------------------------------------- #
# SoundPlayer
# --------------------------------------------------------------------------- #
class SoundPlayer:
    """Tiny audio-cue manager used by ``ScanSession``.

    Parameters
    ----------
    enabled:
        When ``False``, every call to :meth:`play_event` becomes a no-op.
        This is also the global kill-switch used by ``--no-sound``.
    volume:
        Linear gain ``[0.0, 1.0]`` applied to every tone.  Stored at
        construction time so changing it later does not retroactively
        affect already-cached WAV blobs.
    backend:
        Force ``"winsound"`` / ``"cli"`` / ``"auto"``.  Defaults to
        ``"auto"`` which picks the first backend that is callable on
        this OS.  Useful for tests.
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_ENABLED,
        volume: float = DEFAULT_VOLUME,
        backend: str = "auto",
        *,
        alsa_device: str = DEFAULT_ALSA_DEVICE,
        alsa_chunk_bytes: int = DEFAULT_ALSA_CHUNK_BYTES,
    ) -> None:
        self.enabled = bool(enabled)
        self.volume = float(max(0.0, min(1.0, volume)))
        self.alsa_device = str(alsa_device)
        self.alsa_chunk_bytes = int(alsa_chunk_bytes)
        self._backend_name, self._backend = self._select_backend(backend)
        # Pre-render every WAV so playback is allocation-free on the
        # LIVE hot path.
        self._cache: Dict[str, bytes] = {
            name: _build_wav(notes, volume=self.volume)
            for name, notes in _EVENT_NOTES.items()
        }
        # On Windows we also pre-write each WAV to a temp file because
        # ``winsound.SND_MEMORY`` cannot be combined with ``SND_ASYNC``
        # (raises "Cannot play asynchronously from memory").  Pre-writing
        # means the LIVE hot path is allocation-free on play_event.
        self._files: Dict[str, str] = {}
        # Per-event "currently playing" lock for the ALSA backend.
        # ``plughw:<card>,<dev>`` is exclusive -- two simultaneous
        # ``alsaaudio.PCM(device=...)`` calls raise "Device or resource
        # busy".  We use the same drop-the-burst pattern as
        # :class:`mp3_player.MP3Player` so spamming the capture / delete
        # keys never leaves a dangling handle blocking the next play.
        self._play_locks: Dict[str, threading.Lock] = {
            name: threading.Lock() for name in _EVENT_NOTES
        }
        # Scratch files written on demand for non-canned WAVs (e.g. TTS
        # phrases from ``voice.py``).  Cleaned up in ``close``.
        self._scratch: list = []
        # MP3 file registration + lazy PCM cache.
        # ``_mp3_files`` maps event_name -> absolute path.  When a file
        # is registered for an event, ``play_event`` will decode it to
        # raw S16_LE/mono PCM bytes on the first press and stash the
        # result in ``_pcm_cache``.  Subsequent presses reuse the
        # cached bytes -- no re-decode latency on the LIVE hot path.
        # The cache stores ``(sample_rate, pcm_bytes)`` tuples because
        # the ALSA dispatcher needs to know the sample rate to open the
        # PCM handle (``pcm.setrate(...)``).
        self._mp3_files: Dict[str, str] = {}
        self._pcm_cache: Dict[str, Tuple[int, bytes]] = {}
        self._pcm_cache_lock = threading.Lock()
        # MP3 decode tuning.  Mirrors the canonical Pi 5 pipeline from
        # ``mp3_player.play_clip`` so the user's existing captured.mp3 /
        # deleted.mp3 files play unchanged.
        self._mp3_sample_rate: int = 48000  # Hz, MAX98357A DAC rate
        self._mp3_channels: int = 1         # mono amplifier
        self._mp3_volume_db: float = 0.0    # no boost by default
        # Per-event "warned once" set for missing MP3 files.  Mirrors
        # ``MP3Player._missing_warned`` so the LIVE log doesn't fill up.
        self._mp3_missing_warned: set = set()
        if self._backend_name == "winsound" and self.enabled:
            self._write_wav_files()
        if self.enabled:
            extra = ""
            if self._backend_name == "alsa":
                extra = f", device={self.alsa_device}"
            logger.info("SoundPlayer enabled (backend=%s, volume=%.2f%s)",
                        self._backend_name, self.volume, extra)
        else:
            logger.info("SoundPlayer disabled")

    def _write_wav_files(self) -> None:
        """Materialise every cached WAV to a temp file (Windows/winsound).

        ``winsound.PlaySound`` only supports async playback from a *file*
        path (``SND_FILENAME | SND_ASYNC``); passing the WAV bytes with
        ``SND_MEMORY`` works synchronously but raises ``RuntimeError``
        when combined with ``SND_ASYNC``.  We pay the disk-write cost
        once at construction so the LIVE loop never blocks.
        """
        for name, wav in self._cache.items():
            try:
                fd, path = tempfile.mkstemp(prefix=f"docscan_{name}_", suffix=".wav")
                with os.fdopen(fd, "wb") as f:
                    f.write(wav)
                self._files[name] = path
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("could not pre-write %s wav: %s", name, exc)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_backend(prefer: str) -> tuple:
        """Return ``(name, callable_or_None)`` for the chosen backend."""
        prefer = (prefer or "auto").lower()

        winsound = None
        if _winsound_available():
            try:
                import winsound as _winsound  # type: ignore[import-not-found]
                winsound = _winsound
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("winsound import failed: %s", exc)

        cli_argv = _cli_player_available()
        alsa_ok = _alsa_available()

        if prefer == "winsound" and winsound is not None:
            return "winsound", winsound
        if prefer == "cli" and cli_argv is not None:
            return "cli", cli_argv
        if prefer == "alsa" and alsa_ok:
            # Sentinel: the actual device/format constants are read in
            # ``_play_wav_alsa`` at dispatch time so they can be overridden
            # by ``configure(alsa_device=..., alsa_chunk_bytes=...)``.
            return "alsa", "alsa"
        # "auto"
        if winsound is not None:
            return "winsound", winsound
        if cli_argv is not None:
            return "cli", cli_argv
        if alsa_ok:
            return "alsa", "alsa"
        return "none", None

    # ------------------------------------------------------------------ #
    def configure(self, *, enabled: Optional[bool] = None,
                  volume: Optional[float] = None,
                  alsa_device: Optional[str] = None,
                  alsa_chunk_bytes: Optional[int] = None,
                  mp3_volume_db: Optional[float] = None) -> None:
        """Toggle enable/volume at runtime. Re-renders WAVs on volume change."""
        if enabled is not None:
            self.enabled = bool(enabled)
        if volume is not None:
            self.volume = float(max(0.0, min(1.0, volume)))
            self._cache = {
                name: _build_wav(notes, volume=self.volume)
                for name, notes in _EVENT_NOTES.items()
            }
            # Volume changed -> the on-disk WAVs must be regenerated too.
            self._cleanup_files()
            if self._backend_name == "winsound" and self.enabled:
                self._write_wav_files()
        if alsa_device is not None:
            self.alsa_device = str(alsa_device)
        if alsa_chunk_bytes is not None:
            self.alsa_chunk_bytes = int(alsa_chunk_bytes)
        if mp3_volume_db is not None:
            # Volume change invalidates the cached PCM bytes (they
            # already have the old gain baked in).
            self._mp3_volume_db = float(mp3_volume_db)
            with self._pcm_cache_lock:
                self._pcm_cache.clear()

    def configure_mp3(self, event: str, path: Optional[str]) -> None:
        """Register (or clear) an MP3 file path for ``event``.

        Setting ``path=None`` clears the registration.  An empty
        string is rejected as a programming error.

        Once a path is registered, :meth:`play_event` will:

        1. On the first call, decode the MP3 to raw S16_LE/mono PCM
           bytes (one-shot ``pydub`` decode) and cache the result.
        2. On every subsequent call, reuse the cached PCM directly --
           no per-press decode latency, no ffmpeg subprocess.
        3. If the file is missing, pydub/ffmpeg is unavailable, or the
           decode errors out, fall through to the procedural tone so
           the LIVE loop never goes silent.
        """
        if path is None:
            self._mp3_files.pop(event, None)
            with self._pcm_cache_lock:
                self._pcm_cache.pop(event, None)
            return
        if not path:
            raise ValueError("configure_mp3: path must be a non-empty string")
        self._mp3_files[event] = str(path)
        # New file -> invalidate any cached PCM for this event.
        with self._pcm_cache_lock:
            self._pcm_cache.pop(event, None)

    def _decode_mp3_to_pcm(self, event: str, path: str) -> Optional[Tuple[int, bytes]]:
        """Decode ``path`` to ``(sample_rate, pcm_bytes)`` S16_LE/mono.

        Returns ``None`` on any failure -- callers must fall back to
        the procedural tone in that case.
        """
        try:
            from pydub import AudioSegment  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("mp3 decode for %r skipped (pydub unavailable): %s",
                         event, exc)
            return None
        try:
            seg = AudioSegment.from_mp3(path)
        except Exception as exc:
            logger.debug("mp3 decode for %r failed (%s): %s",
                         event, path, exc)
            return None
        try:
            seg = seg.set_channels(self._mp3_channels)
            seg = seg.set_sample_width(2)  # 16-bit = S16_LE
            if self._mp3_volume_db:
                seg = seg + self._mp3_volume_db
            return (seg.frame_rate, bytes(seg.raw_data))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("mp3 reformat for %r failed: %s", event, exc)
            return None

    def _ensure_mp3_pcm(self, event: str) -> Optional[Tuple[int, bytes]]:
        """Return cached PCM for ``event``, decoding on first miss.

        Returns ``None`` if no MP3 is registered, the file is missing,
        or the decode failed.  Thread-safe -- a single decode runs even
        if two callers race for the cache slot.
        """
        with self._pcm_cache_lock:
            cached = self._pcm_cache.get(event)
        if cached is not None:
            return cached
        path = self._mp3_files.get(event)
        if not path:
            return None
        if not os.path.exists(path):
            # Same "warn once, stay quiet after" pattern as
            # ``MP3Player.play_event`` so the LIVE log doesn't fill up.
            if event not in self._mp3_missing_warned:
                self._mp3_missing_warned.add(event)
                logger.warning(
                    "mp3 clip for %r not found at %s; "
                    "falling back to procedural tone (further missing-file "
                    "logs will stay at DEBUG)", event, path,
                )
            else:
                logger.debug("mp3 clip for %r missing: %s", event, path)
            return None
        decoded = self._decode_mp3_to_pcm(event, path)
        if decoded is None:
            return None
        # Re-check the cache under the lock -- another thread may have
        # raced ahead and filled it while our decode was running.
        with self._pcm_cache_lock:
            existing = self._pcm_cache.get(event)
            if existing is not None:
                return existing
            self._pcm_cache[event] = decoded
            return decoded

    def _cleanup_files(self) -> None:
        """Delete any temp WAVs previously written for winsound playback."""
        for path in self._files.values():
            try:
                os.unlink(path)
            except OSError:
                pass
        self._files.clear()
        # Scratch files written by ``_play_wav`` for non-canned WAVs.
        for path in self._scratch:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._scratch.clear()

    def close(self) -> None:
        """Release temp files. Safe to call multiple times."""
        self._cleanup_files()

    # ------------------------------------------------------------------ #
    def play_event(self, name: str) -> bool:
        """Play the named event. Returns True if dispatched.

        Lookup order:

        1. **MP3 cache** -- if an MP3 file was registered via
           :meth:`configure_mp3` for ``name`` and the decode produced
           cached PCM, play that.  This is the user-supplied long-form
           audio path (e.g. ``captured.mp3`` on the Pi 5 deployment).
        2. **Procedural tone** -- the built-in short WAV previously
           synthesised at construction time.  This is the fallback so
           audio is always heard even when MP3 is misconfigured.
        """
        if not self.enabled:
            logger.debug("play_event(%r): skipped (SoundPlayer disabled)", name)
            return False
        # MP3-first lookup.  If a file is registered and decodes
        # successfully, play it.  The dispatch goes through the same
        # ALSA backend + per-event lock + pcm.close() machinery as the
        # procedural tone path, so the two layers never race for
        # ``plughw:2,0``.
        pcm = self._ensure_mp3_pcm(name)
        if pcm is not None:
            sample_rate, raw = pcm
            rc = self._play_pcm(raw, sample_rate=sample_rate,
                                event_name=name)
            logger.info("play_event(%r) -> %s (backend=%s, source=mp3, "
                        "rate=%d, bytes=%d)",
                        name, rc, self._backend_name, sample_rate, len(raw))
            return rc
        wav = self._cache.get(name)
        if wav is None:
            logger.debug("play_event(%r): unknown event, ignoring", name)
            return False
        rc = self._play_wav(wav, event_name=name)
        logger.info("play_event(%r) -> %s (backend=%s, source=tone, volume=%.2f)",
                    name, rc, self._backend_name, self.volume)
        return rc

    # ------------------------------------------------------------------ #
    def _play_wav(self, wav: bytes, *, event_name: Optional[str] = None) -> bool:
        """Dispatch ``wav`` to the chosen backend. Always non-blocking.

        ``event_name`` is only used to look up a pre-rendered temp file
        for the three canned tone events.  When it is ``None`` (e.g.
        :mod:`voice` synthesized a TTS phrase on the fly) we materialise
        the bytes to a fresh temp file so the winsound backend can play
        them asynchronously; that file is deleted on :meth:`close`.
        """
        name, backend = self._backend_name, self._backend
        # Scratch file for non-canned WAVs (TTS phrases etc.).
        scratch_path: Optional[str] = None
        try:
            if name == "winsound" and backend is not None:
                # ``winsound.SND_MEMORY`` cannot be combined with
                # ``SND_ASYNC`` ("Cannot play asynchronously from memory"),
                # so we always play from a real file path.
                path = ""
                if event_name is not None:
                    path = self._files.get(event_name, "")
                if not path:
                    # Materialise ``wav`` to a fresh temp file.  ``close``
                    # will sweep it up via ``_scratch``.
                    fd, scratch_path = tempfile.mkstemp(
                        prefix="docscan_wav_", suffix=".wav",
                    )
                    try:
                        with os.fdopen(fd, "wb") as f:
                            f.write(wav)
                    except Exception:
                        scratch_path = None
                        raise
                    self._scratch.append(scratch_path)
                    path = scratch_path
                flags = (
                    backend.SND_FILENAME
                    | backend.SND_ASYNC
                    | getattr(backend, "SND_NODEFAULT", 0)
                )
                backend.PlaySound(path, flags)
                return True
            if name == "cli" and backend is not None:
                argv = list(backend)
                # Write to a temp file (CLI players can't read from stdin
                # portably) and clean it up once afplay/aplay is done.
                def _run_and_cleanup() -> None:
                    fd, path = tempfile.mkstemp(suffix=".wav")
                    try:
                        with os.fdopen(fd, "wb") as f:
                            f.write(wav)
                        subprocess.Popen(
                            argv + [path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("cli sound playback failed: %s", exc)
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                threading.Thread(target=_run_and_cleanup, daemon=True).start()
                return True
            if name == "alsa" and backend == "alsa":
                # Linux / Pi 5 I2S path: write the synth WAV through
                # pydub -> pyalsaaudio -> plughw:<card>,<dev>.  Same
                # pipeline shape as ``mp3_player.play_clip`` but the
                # source bytes come from ``SoundPlayer._cache`` instead
                # of an MP3 file, and we always emit S16_LE / mono at the
                # synthesised sample rate so the DAC gets exactly what
                # we built.  Runs on a daemon thread so the LIVE loop
                # never blocks on the speaker-write.
                #
                # ``plughw:<card>,<dev>`` is an *exclusive* ALSA device.
                # Two simultaneous ``alsaaudio.PCM(device=...)`` calls
                # raise "Device or resource busy", so we serialise
                # back-to-back plays per event via ``self._play_locks``.
                # If a previous request is still streaming we drop the
                # new one (matches :class:`mp3_player.MP3Player`'s
                # "drop the burst" design).
                lock = self._play_locks.get(event_name or "") if event_name else None
                if lock is None:
                    # TTS / on-the-fly WAV (no canned event name): use a
                    # shared lock so we don't trip "Device busy" either.
                    lock = getattr(self, "_scratch_lock", None)
                    if lock is None:
                        self._scratch_lock = threading.Lock()
                        lock = self._scratch_lock
                if not lock.acquire(blocking=False):
                    logger.debug(
                        "alsa sound playback for %r: previous clip still "
                        "playing, dropping (matches MP3Player policy)",
                        event_name,
                    )
                    return True
                device = self.alsa_device
                chunk_bytes = self.alsa_chunk_bytes
                captured_lock = lock

                def _run_alsa(wav_bytes: bytes = wav,
                              dev: str = device,
                              chunk: int = chunk_bytes,
                              lk: threading.Lock = captured_lock) -> None:
                    pcm = None
                    try:
                        import alsaaudio  # type: ignore[import-not-found]
                        from pydub import AudioSegment  # type: ignore[import-not-found]
                        seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
                        seg = seg.set_channels(1)
                        seg = seg.set_sample_width(2)  # S16_LE
                        # Keep the synthesised sample rate so the
                        # attack/release envelope we wrote in
                        # ``_sine`` survives the plughw resampler.
                        pcm = alsaaudio.PCM(device=dev,
                                            mode=alsaaudio.PCM_NORMAL)
                        pcm.setchannels(1)
                        pcm.setrate(seg.frame_rate)
                        pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
                        pcm.setperiodsize(chunk)
                        raw = seg.raw_data
                        for off in range(0, len(raw), chunk):
                            pcm.write(raw[off:off + chunk])
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("alsa sound playback failed: %s", exc)
                    finally:
                        # Release the PCM handle BEFORE the lock so the
                        # next play_event() can grab plughw:2,0 without
                        # "Device or resource busy".  Closing an
                        # already-closed PCM is a no-op in pyalsaaudio.
                        if pcm is not None:
                            try:
                                pcm.close()
                            except Exception:
                                pass
                        try:
                            lk.release()
                        except Exception:
                            pass

                threading.Thread(
                    target=_run_alsa, daemon=True,
                    name=f"sound-alsa-{event_name or 'wav'}",
                ).start()
                return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("sound backend %s failed: %s", name, exc)
            return False
        # No backend available - silent no-op.
        return False

    # ------------------------------------------------------------------ #
    def _play_pcm(self, raw: bytes, *, sample_rate: int,
                  event_name: Optional[str] = None) -> bool:
        """Play pre-decoded S16_LE mono PCM ``raw`` to the active backend.

        This is the consolidation point that lets ``play_event`` play
        either the procedural tone or a user-supplied MP3 clip from a
        single dispatch path.  The byte layout is identical regardless
        of source -- the existing ``_play_wav`` ALSA branch already
        downmixes the synth WAV to S16_LE / mono, so reusing its lock
        + pcm.close() plumbing keeps the two layers from racing for
        ``plughw:2,0``.

        Parameters
        ----------
        raw:
            Raw PCM bytes (16-bit signed little-endian, mono).
        sample_rate:
            Frame rate of ``raw`` in Hz.  Required so the ALSA backend
            can call ``pcm.setrate(...)`` correctly; ignored by the
            winsound backend (which rebuilds a WAV header around the
            bytes via :func:`wave`).
        event_name:
            Used to look up the per-event playback lock so back-to-back
            MP3 + tone plays serialise instead of colliding on the
            exclusive ALSA device.  ``None`` falls back to a shared
            scratch lock (TTS-style calls).

        Returns
        -------
        bool
            ``True`` if dispatch was accepted (``False`` means no
            usable backend or the call was a silent no-op).
        """
        if not raw:
            logger.debug("_play_pcm(%r): empty buffer, nothing to play", event_name)
            return False
        name, backend = self._backend_name, self._backend
        try:
            if name == "winsound" and backend is not None:
                # Wrap the raw PCM in a WAV header so winsound (which
                # only accepts WAV files) can play it.  The temp file
                # is tracked via ``self._scratch`` and swept in
                # ``close``.
                fd, path = tempfile.mkstemp(
                    prefix="docscan_pcm_", suffix=".wav",
                )
                scratch_path = path
                try:
                    with os.fdopen(fd, "wb") as f:
                        # 1ch, 2 bytes/sample, 48 kHz (or whatever the
                        # decoder produced) -> 44-byte RIFF/WAVE header
                        # followed by the raw PCM bytes.
                        nchannels = 1
                        sampwidth = 2
                        with wave.open(f, "wb") as wf:
                            wf.setnchannels(nchannels)
                            wf.setsampwidth(sampwidth)
                            wf.setframerate(int(sample_rate))
                            wf.writeframes(raw)
                except Exception:
                    scratch_path = None
                    raise
                self._scratch.append(scratch_path)
                flags = (
                    backend.SND_FILENAME
                    | backend.SND_ASYNC
                    | getattr(backend, "SND_NODEFAULT", 0)
                )
                backend.PlaySound(scratch_path, flags)
                return True
            if name == "cli" and backend is not None:
                argv = list(backend)
                # Same shape as the WAV CLI path: write a real file
                # (CLI players don't read raw PCM from stdin), fire
                # ``Popen``, and let the OS clean up the temp file
                # when the player exits.
                def _run_and_cleanup(pcm_data: bytes = raw,
                                     rate: int = sample_rate) -> None:
                    fd, path = tempfile.mkstemp(suffix=".wav")
                    try:
                        with os.fdopen(fd, "wb") as f:
                            with wave.open(f, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(int(rate))
                                wf.writeframes(pcm_data)
                        subprocess.Popen(
                            argv + [path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception as exc:  # pragma: no cover - def.
                        logger.debug("cli pcm playback failed: %s", exc)
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                threading.Thread(
                    target=_run_and_cleanup, daemon=True,
                ).start()
                return True
            if name == "alsa" and backend == "alsa":
                # Same per-event lock + pcm.close() guarantee as the
                # WAV path, but the bytes are already decoded so we
                # skip the pydub round-trip.  This is the fix for the
                # "Device or resource busy" race that existed when
                # ``SoundPlayer`` and ``MP3Player`` each opened their
                # own ``alsaaudio.PCM`` handle for the same
                # ``plughw:2,0`` device.
                lock = self._play_locks.get(event_name or "") if event_name else None
                if lock is None:
                    lock = getattr(self, "_scratch_lock", None)
                    if lock is None:
                        self._scratch_lock = threading.Lock()
                        lock = self._scratch_lock
                if not lock.acquire(blocking=False):
                    logger.debug(
                        "alsa pcm playback for %r: previous clip still "
                        "playing, dropping (matches tone/MP3 policy)",
                        event_name,
                    )
                    return True
                device = self.alsa_device
                chunk_bytes = self.alsa_chunk_bytes
                captured_lock = lock
                rate = int(sample_rate)

                def _run_alsa_pcm(pcm_data: bytes = raw,
                                  dev: str = device,
                                  chunk: int = chunk_bytes,
                                  lk: threading.Lock = captured_lock,
                                  rate_hz: int = rate) -> None:
                    pcm = None
                    try:
                        import alsaaudio  # type: ignore[import-not-found]
                        pcm = alsaaudio.PCM(device=dev,
                                            mode=alsaaudio.PCM_NORMAL)
                        pcm.setchannels(1)
                        pcm.setrate(rate_hz)
                        pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
                        pcm.setperiodsize(chunk)
                        for off in range(0, len(pcm_data), chunk):
                            pcm.write(pcm_data[off:off + chunk])
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("alsa pcm playback failed: %s", exc)
                    finally:
                        # Release the PCM handle BEFORE the lock so the
                        # next play_event() can grab plughw:2,0 without
                        # "Device or resource busy".  Closing an
                        # already-closed PCM is a no-op in pyalsaaudio.
                        if pcm is not None:
                            try:
                                pcm.close()
                            except Exception:
                                pass
                        try:
                            lk.release()
                        except Exception:
                            pass

                threading.Thread(
                    target=_run_alsa_pcm, daemon=True,
                    name=f"sound-alsa-pcm-{event_name or 'pcm'}",
                ).start()
                return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("pcm backend %s failed: %s", name, exc)
            return False
        # No backend available - silent no-op.
        return False


# --------------------------------------------------------------------------- #
# Module-level convenience (used when ScanSession doesn't need its own instance)
# --------------------------------------------------------------------------- #
_default_player: Optional[SoundPlayer] = None


def get_default_player() -> SoundPlayer:
    """Return the process-wide ``SoundPlayer`` (lazy-initialised)."""
    global _default_player
    if _default_player is None:
        _default_player = SoundPlayer()
    return _default_player


def play_event(name: str) -> bool:
    """Convenience: dispatch a named event through the default player."""
    return get_default_player().play_event(name)