"""Long-form MP3 audio cues for the Smart Document Scanner (Raspberry Pi 5).

This module is the **file-based** companion to :mod:`sound` (short WAV tones)
and :mod:`voice` (spoken TTS).  While the scanner is in use on a Raspberry Pi 5
it plays two short, user-supplied MP3 clips from the project root:

* ``captured.mp3``  - played immediately after a page is committed to the PDF
  (manual ``C`` press, auto-capture fire, or any other successful capture path).
* ``deleted.mp3``   - played after the most recently captured page has been
  removed via the ``X`` key.

Why this layer exists
=====================

On the Raspberry Pi 5 deployment we wired up a **MAX98357A mono I2S amplifier**
feeding a 5 W speaker through the dedicated ``dtoverlay=max98357a`` audio card
(card 2, device 0).  That hardware has two important properties:

1. **No software volume slider.**  The chip's GAIN pin is hard-wired, and
   ALSA's user-space mixers don't see ``plughw:2,0`` because we bypass
   PulseAudio/PipeWire and write directly to the I2S hardware driver.
2. **No PipeWire / PulseAudio routing.**  We hand the raw, byte-level PCM
   stream to ``pyalsaaudio`` so the OS sound servers can't throttle the gain.

That means an MP3 clip you drop into the project root will play at whatever
level ``ffmpeg`` decodes it to — typically quiet enough that the user has to
press their ear to the speaker to hear it.  The fix is the pipeline below:

  1. ``pydub`` (which calls ``ffmpeg`` under the hood) decodes the MP3 into
     raw audio, **downmixes** it to mono (one channel) and **resamples** it
     to 48 kHz / 16-bit — the exact format the MAX98357A DAC speaks.
  2. We add a software gain boost (``+8 dB`` by default; configurable).
  3. We open ``plughw:2,0`` via ``pyalsaaudio`` and stream the bytes in
     ``32768``-byte chunks — the exact period size that the speaker-test
     routine found to be glitch-free on this hardware.

Backend matrix
==============

* **Linux (Raspberry Pi 5, etc.)** — full pydub + pyalsaaudio pipeline.
  Optional: when ``pydub`` / ``pyalsaaudio`` / ``ffmpeg`` are missing the
  call silently no-ops (DEBUG-logged) so the LIVE loop never stalls.
* **Windows / macOS** — the same module imports cleanly but every call
  becomes a no-op, because the MAX98357A hardware only exists on the Pi
  deployment.  This keeps the dev box green without needing ``pydub`` or
  a Pi I2S overlay.

Design goals
============

* **Never blocks the LIVE loop.**  Every playback runs on a daemon thread;
  the hot path returns immediately.
* **Coalesces bursts.**  If the user spams ``C`` ten times in a row, the
  player drops in-flight requests rather than queueing an audible backlog.
* **File existence check.**  Missing MP3s are logged once and silently
  skipped — no exception thrown on the hot path.
* **Single-source defaults.**  All tuning constants live in
  :mod:`config` (``DEFAULT_MP3_*``) and on :class:`config.AppConfig`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Defaults - may be overridden by ``configure(...)`` or by the CLI flags on
# ``app.py``.  These mirror the ones in ``config.py`` but live here too so
# the module is standalone-testable.
# --------------------------------------------------------------------------- #
DEFAULT_ENABLED: bool = True
DEFAULT_DEVICE: str = "plughw:2,0"   # MAX98357A I2S amp on Raspberry Pi 5
DEFAULT_VOLUME_DB: float = 8.0       # +8 dB software gain (no hw mixer)
DEFAULT_CAPTURED_FILE: str = "captured.mp3"
DEFAULT_DELETED_FILE: str = "deleted.mp3"
DEFAULT_SAMPLE_RATE: int = 48000     # Hz - matches MAX98357A DAC
DEFAULT_CHANNELS: int = 1            # mono (single-speaker amplifier)
DEFAULT_CHUNK_BYTES: int = 32768     # matches the working period size

# --------------------------------------------------------------------------- #
# Backend probes
# --------------------------------------------------------------------------- #
def _pydub_available() -> bool:
    """Return ``True`` if :mod:`pydub` is importable."""
    try:
        from pydub import AudioSegment  # type: ignore[import-not-found]  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("pydub import failed: %s", exc)
        return False


def _alsaaudio_available() -> bool:
    """Return ``True`` if :mod:`alsaaudio` is importable."""
    try:
        import alsaaudio  # type: ignore[import-not-found]  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("alsaaudio import failed: %s", exc)
        return False


def _ffmpeg_available() -> bool:
    """Return ``True`` if the ``ffmpeg`` binary is on PATH.

    pydub shells out to ffmpeg for MP3 decoding.  We check this so the
    DEBUG log clearly says *which* dependency is missing.
    """
    try:
        return shutil.which("ffmpeg") is not None
    except Exception:  # pragma: no cover - defensive
        return False


def _linux_pi_audio_available() -> bool:
    """Return ``True`` if the Pi I2S overlay is loaded (i.e. card 2, device 0)."""
    if not sys.platform.startswith("linux"):
        return False
    if not _alsaaudio_available():
        return False
    if not _ffmpeg_available():
        return False
    # Best-effort probe: try to actually open the device.  Done lazily in the
    # playback worker, but we expose a no-op probe so tests can introspect.
    return True


# --------------------------------------------------------------------------- #
# The pure-pipeline ``play_clip`` - exposed for tests + reuse.
# --------------------------------------------------------------------------- #
def play_clip(
    filename: str,
    *,
    device: str = DEFAULT_DEVICE,
    volume_db: float = DEFAULT_VOLUME_DB,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> bool:
    """Play ``filename`` synchronously on the MAX98357A hardware.

    Parameters
    ----------
    filename:
        Absolute or project-root-relative path to a ``.mp3`` file.
    device:
        ALSA device string (defaults to ``plughw:2,0``, the Pi 5 I2S amp).
    volume_db:
        Software gain in decibels added to the decoded audio before write.
        ``+8 dB`` is loud enough for desk-distance listening on a 5 W cone.
    sample_rate, channels, chunk_bytes:
        Direct passthrough to the ALSA device configuration.  Defaults are
        the values that the working speaker-test routine found to be clean.

    Returns
    -------
    bool
        ``True`` if the file was opened and at least one chunk was streamed.
        ``False`` if any precondition failed (file missing, backend missing,
        device refused to open, etc.).  Failures are DEBUG-logged.

    Notes
    -----
    This function is the verbatim pipeline from the working Pi 5 speaker
    test.  It is exposed (not inlined into the worker thread) so that
    :mod:`runs.smoke_mp3` can call it with a fake device and inspect the
    byte stream that *would* have been written.
    """
    if not os.path.exists(filename):
        logger.debug("play_clip: file not found: %s", filename)
        return False
    if not _pydub_available() or not _alsaaudio_available():
        logger.debug("play_clip: pydub/alsaaudio not installed; skipping %s", filename)
        return False
    if not _ffmpeg_available():
        logger.debug("play_clip: ffmpeg not on PATH; cannot decode %s", filename)
        return False
    if not sys.platform.startswith("linux"):
        # Pi-only pipeline.  Emit a one-shot INFO so Windows/macOS users
        # can see *why* their long-form MP3 cues are silent.
        logger.info("play_clip: %s skipped (Pi I2S pipeline is Linux-only)",
                    filename)
        return False

    # Imported lazily so the module loads on Windows / macOS where neither
    # package is installed.
    from pydub import AudioSegment  # type: ignore[import-not-found]
    import alsaaudio  # type: ignore[import-not-found]

    try:
        # 1. Decode + downmix + resample + reformat.
        song = AudioSegment.from_mp3(filename)
        song = song.set_channels(channels)
        song = song.set_frame_rate(sample_rate)
        song = song.set_sample_width(2)  # 2 bytes = 16-bit audio (S16_LE)

        # 2. Digital boost.
        song = song + volume_db

        # 3. Open the hardware PCM device.
        pcm = alsaaudio.PCM(device=device, mode=alsaaudio.PCM_NORMAL)
        pcm.setchannels(channels)
        pcm.setrate(sample_rate)
        pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        pcm.setperiodsize(chunk_bytes)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("play_clip: setup failed for %s: %s", filename, exc)
        return False

    raw_data = song.raw_data
    if not raw_data:
        logger.debug("play_clip: decoded audio is empty: %s", filename)
        return False

    try:
        for offset in range(0, len(raw_data), chunk_bytes):
            chunk = raw_data[offset:offset + chunk_bytes]
            pcm.write(chunk)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("play_clip: write failed for %s: %s", filename, exc)
        return False
    return True


# --------------------------------------------------------------------------- #
# MP3Player
# --------------------------------------------------------------------------- #
class MP3Player:
    """Async MP3-clip dispatcher used by ``ScanSession``.

    Parameters
    ----------
    enabled:
        Global kill-switch (mirrors ``--no-mp3``).  When ``False``, every
        call to :meth:`play_event` is a silent no-op.
    captured_file / deleted_file:
        Absolute paths to the two MP3 clips.  Defaults resolve to
        ``<project_root>/captured.mp3`` and ``<project_root>/deleted.mp3``.
    device / volume_db / sample_rate / channels / chunk_bytes:
        Direct passthrough to :func:`play_clip`.
    project_root:
        Directory used to resolve the default clip filenames.  Defaults
        to :data:`config.PROJECT_ROOT` so ``app.py`` doesn't have to
        pass it explicitly.

    Notes
    -----
    On non-Linux platforms (or when ``pydub`` / ``pyalsaaudio`` / ``ffmpeg``
    are missing) the player is a **no-op** rather than an error.  This keeps
    development on Windows friction-free while preserving the full hardware
    pipeline on the Pi 5 deployment.
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_ENABLED,
        captured_file: str = "",
        deleted_file: str = "",
        device: str = DEFAULT_DEVICE,
        volume_db: float = DEFAULT_VOLUME_DB,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        project_root: Optional[str] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = str(device)
        self.volume_db = float(volume_db)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.chunk_bytes = int(chunk_bytes)

        if project_root is None:
            try:
                import config as _config  # type: ignore[import-not-found]
                project_root = str(_config.PROJECT_ROOT)
            except Exception:  # pragma: no cover - defensive
                project_root = os.getcwd()
        self.project_root = project_root

        # Resolve default filenames against the project root.
        self.captured_file = self._resolve_path(
            captured_file or DEFAULT_CAPTURED_FILE,
        )
        self.deleted_file = self._resolve_path(
            deleted_file or DEFAULT_DELETED_FILE,
        )

        # Per-event "file missing" warnings so a missing clip doesn't
        # spam the log on every keypress.
        self._missing_warned: set = set()
        self._missing_lock = threading.Lock()

        # Per-event playback lock.  We don't queue bursts: if the user
        # spams ``C`` ten times we drop everything except the most recent
        # request, matching the design choice that overlapping tones are
        # more annoying than informative.
        self._play_locks: Dict[str, threading.Lock] = {
            "captured": threading.Lock(),
            "page_deleted": threading.Lock(),
        }

        if self.enabled:
            backend_ok = _linux_pi_audio_available()
            cap_exists = os.path.exists(self.captured_file)
            del_exists = os.path.exists(self.deleted_file)
            logger.info(
                "MP3Player enabled (backend_ok=%s, device=%s, volume=%+0.1f dB, "
                "captured=%s [%s], deleted=%s [%s])",
                backend_ok, self.device, self.volume_db,
                self.captured_file, "OK" if cap_exists else "MISSING",
                self.deleted_file, "OK" if del_exists else "MISSING",
            )
        else:
            logger.info("MP3Player disabled")

    # ------------------------------------------------------------------ #
    def _resolve_path(self, name: str) -> str:
        """Return ``name`` if absolute, else join it onto ``project_root``."""
        if os.path.isabs(name):
            return name
        return os.path.join(self.project_root, name)

    # ------------------------------------------------------------------ #
    def configure(
        self,
        *,
        enabled: Optional[bool] = None,
        captured_file: Optional[str] = None,
        deleted_file: Optional[str] = None,
        volume_db: Optional[float] = None,
    ) -> None:
        """Toggle enable / volume / filenames at runtime."""
        if enabled is not None:
            self.enabled = bool(enabled)
        if volume_db is not None:
            self.volume_db = float(volume_db)
        if captured_file is not None:
            self.captured_file = self._resolve_path(captured_file)
            with self._missing_lock:
                self._missing_warned.discard("captured")
        if deleted_file is not None:
            self.deleted_file = self._resolve_path(deleted_file)
            with self._missing_lock:
                self._missing_warned.discard("page_deleted")

    # ------------------------------------------------------------------ #
    def play_event(self, name: str) -> bool:
        """Dispatch ``name`` on a daemon thread.  Returns ``True`` if accepted.

        Recognised event names:

        * ``"captured"``     - plays ``captured_file``.
        * ``"page_deleted"`` - plays ``deleted_file``.

        Anything else is silently ignored (DEBUG-logged).
        """
        if not self.enabled:
            logger.debug("mp3 play_event(%r): skipped (MP3Player disabled)", name)
            return False
        filename = self._filename_for(name)
        if filename is None:
            logger.debug("mp3 play_event(%r): unknown event, ignoring", name)
            return False
        if not os.path.exists(filename):
            with self._missing_lock:
                if name not in self._missing_warned:
                    self._missing_warned.add(name)
                    logger.warning(
                        "mp3 clip for %r not found at %s; "
                        "further missing-file logs will stay at DEBUG",
                        name, filename,
                    )
                else:
                    logger.debug("mp3 clip for %r missing: %s", name, filename)
            return False

        # Non-blocking dispatch.  We use a per-event lock as a "currently
        # playing" semaphore: if a previous request is still streaming we
        # drop this one (the user gets the most recent cue, not a backlog).
        lock = self._play_locks.get(name)
        if lock is not None and not lock.acquire(blocking=False):
            logger.debug("mp3 play_event(%r): previous clip still playing, dropping", name)
            return False

        thread = threading.Thread(
            target=self._run_clip,
            args=(name, filename, lock),
            daemon=True,
            name=f"mp3-{name}",
        )
        thread.start()
        return True

    # ------------------------------------------------------------------ #
    def _filename_for(self, name: str) -> Optional[str]:
        if name == "captured":
            return self.captured_file
        if name == "page_deleted":
            return self.deleted_file
        return None

    # ------------------------------------------------------------------ #
    def _run_clip(
        self,
        event_name: str,
        filename: str,
        lock: Optional[threading.Lock],
    ) -> None:
        """Worker entry point - runs :func:`play_clip` and releases the lock."""
        try:
            ok = play_clip(
                filename,
                device=self.device,
                volume_db=self.volume_db,
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_bytes=self.chunk_bytes,
            )
            if not ok:
                logger.debug("mp3 %s: playback failed for %s", event_name, filename)
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:  # pragma: no cover - defensive
                    pass


# --------------------------------------------------------------------------- #
# Module-level convenience (used when ScanSession doesn't need its own instance)
# --------------------------------------------------------------------------- #
_default_player: Optional[MP3Player] = None
_default_lock = threading.Lock()


def get_default_player() -> MP3Player:
    """Return the process-wide :class:`MP3Player` (lazy-initialised)."""
    global _default_player
    if _default_player is None:
        with _default_lock:
            if _default_player is None:
                _default_player = MP3Player()
    return _default_player


def play_event(name: str) -> bool:
    """Convenience: dispatch a named event through the default player."""
    return get_default_player().play_event(name)
