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
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Defaults - may be overridden by ``configure(enabled=..., volume=...)``.
# --------------------------------------------------------------------------- #
DEFAULT_ENABLED: bool = True
DEFAULT_VOLUME: float = 0.6      # 0.0 (silent) - 1.0 (full)
DEFAULT_SAMPLE_RATE: int = 22050 # Hz - plenty for short blips

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
    ) -> None:
        self.enabled = bool(enabled)
        self.volume = float(max(0.0, min(1.0, volume)))
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
        # Scratch files written on demand for non-canned WAVs (e.g. TTS
        # phrases from ``voice.py``).  Cleaned up in ``close``.
        self._scratch: list = []
        if self._backend_name == "winsound" and self.enabled:
            self._write_wav_files()
        if self.enabled:
            logger.info("SoundPlayer enabled (backend=%s, volume=%.2f)",
                        self._backend_name, self.volume)
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

        if prefer == "winsound" and winsound is not None:
            return "winsound", winsound
        if prefer == "cli" and cli_argv is not None:
            return "cli", cli_argv
        # "auto"
        if winsound is not None:
            return "winsound", winsound
        if cli_argv is not None:
            return "cli", cli_argv
        return "none", None

    # ------------------------------------------------------------------ #
    def configure(self, *, enabled: Optional[bool] = None,
                  volume: Optional[float] = None) -> None:
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
        """Play the named event's WAV. Returns True if dispatched."""
        if not self.enabled:
            logger.debug("play_event(%r): skipped (SoundPlayer disabled)", name)
            return False
        wav = self._cache.get(name)
        if wav is None:
            logger.debug("play_event(%r): unknown event, ignoring", name)
            return False
        rc = self._play_wav(wav, event_name=name)
        logger.info("play_event(%r) -> %s (backend=%s, volume=%.2f)",
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("sound backend %s failed: %s", name, exc)
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