"""Spoken-voice prompts for the Smart Document Scanner.

This module is the voice-layer companion to :mod:`sound`.  Where ``sound.py``
plays short tone cues (3 fixed events), ``voice.py`` plays short spoken
phrases for ~10 lifecycle events:

* ``"detected"``           - a quad has just been seen in the LIVE frame.
* ``"stable"``            - the FSM has confirmed 5/5 stable frames.
* ``"capture_manual"``    - ``C`` was pressed and the page committed.
* ``"capture_auto"``      - the FSM auto-fired and the page committed.
* ``"capture_rejected"``  - the quality gate rejected the capture.
* ``"page_change"``       - the page-change detector noticed a flip.
* ``"document_new"``      - ``N`` was pressed and a new session started.
* ``"document_saved"``    - ``D`` was pressed and the PDF was written.
* ``"document_export"``   - the scanner finished and wrote outputs.
* ``"shutdown"``          - main loop is about to exit cleanly.
* ``"error"``             - something went wrong on the LIVE hot path.

Design goals
============

* **Cross-platform, fully offline.**

  - On Windows we use :mod:`pyttsx3` (SAPI5 wrapper, no internet needed,
    no extra runtime needed beyond ``pip install pyttsx3``).
  - On Linux / Raspberry Pi we shell out to ``espeak-ng`` (the
    ``espeak-ng`` apt package, available on Raspberry Pi OS without any
    extra Python dependency).

* **Single audio pipeline.**

  Both backends return raw WAV bytes which are then handed to
  :class:`sound.SoundPlayer._play_wav`.  We never call ``winsound`` /
  ``afplay`` / ``aplay`` ourselves - we reuse whatever backend ``sound``
  has already selected.  This keeps tone + voice perfectly synchronised
  on the same audio device.

* **Never blocks the LIVE loop.**

  Synthesis (both ``pyttsx3.save_to_file`` and ``espeak-ng`` subprocess)
  happens inside a daemon thread.  The hot path returns immediately.

* **Caches phrases in memory.**

  The first time a phrase is requested we render the WAV and stash the
  bytes in a ``dict`` so subsequent calls are allocation-free on the hot
  path.  Phrases that contain dynamic values (``{reason}``, ``{n}``,
  ``{path}`` ...) are keyed by ``(event, formatted_phrase)``.

* **Defensive everywhere.**

  If the backend cannot be imported / invoked, or synthesis fails, the
  call is silently dropped (``return False``) and logged at DEBUG level.
  The LIVE loop is never stalled by audio.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Defaults - may be overridden by ``configure(enabled=..., language=...)``
# or by the CLI flags on ``app.py``.
# --------------------------------------------------------------------------- #
DEFAULT_ENABLED: bool = True
DEFAULT_LANGUAGE: str = "en"      # espeak-ng voice id (e.g. "en", "en-us")
DEFAULT_RATE_WPM: int = 165       # speaking rate (pyttsx3: words/min)
DEFAULT_BACKEND: str = "auto"     # "pyttsx3" | "espeak" | "auto"


# --------------------------------------------------------------------------- #
# Phrase table.
#
# Each entry is a template.  ``{reason}`` / ``{n}`` / ``{path}`` /
# ``{detail}`` are filled by ``speak(event, reason=..., n=..., ...)``.
# --------------------------------------------------------------------------- #
_PHRASE_TEMPLATES: Dict[str, str] = {
    # FSM
    "detected":         "Document detected",
    "stable":           "Stable, capturing",

    # Capture outcomes
    "capture_manual":   "Page added manually",
    "capture_auto":     "Page added automatically",
    "capture_rejected": "Capture rejected, {reason}",

    # Page change detector
    "page_change":      "Page {n}",

    # Session lifecycle
    "document_new":     "New document opened",
    "document_saved":   "Document saved, {n} pages",
    "document_export":  "Exported to {path}",

    # Shutdown / error
    "shutdown":         "Scanner shutting down",
    "error":            "Error, {detail}",
}


# --------------------------------------------------------------------------- #
# Backend probes
# --------------------------------------------------------------------------- #
def _pyttsx3_available() -> bool:
    """Return True if :mod:`pyttsx3` can be imported on this OS."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import pyttsx3  # type: ignore[import-not-found]  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("pyttsx3 import failed: %s", exc)
        return False


def _espeak_available() -> bool:
    """Return True if the ``espeak-ng`` binary is on PATH."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        rc = subprocess.call(
            ["command", "-v", "espeak-ng"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
        return rc == 0
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover
        logger.debug("espeak-ng probe failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def _synthesise_pyttsx3(text: str, *, rate_wpm: int) -> bytes:
    """Render ``text`` to a WAV blob via :mod:`pyttsx3` (Windows)."""
    import pyttsx3  # type: ignore[import-not-found]

    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", int(rate_wpm))
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(fd)
            engine.save_to_file(text, path)
            engine.runAndWait()
            with open(path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    finally:
        try:
            engine.stop()
        except Exception:  # pragma: no cover - defensive
            pass


def _synthesise_espeak(text: str, *, language: str) -> bytes:
    """Render ``text`` to a WAV blob via the ``espeak-ng`` subprocess."""
    # ``-w -`` writes a WAV file to stdout.  We capture it.
    proc = subprocess.run(
        ["espeak-ng", "-v", language, "-w", "-", text],
        capture_output=True,
        shell=False,
    )
    if proc.returncode != 0:  # pragma: no cover - defensive
        raise RuntimeError(
            "espeak-ng exited with rc=%s (stderr=%r)"
            % (proc.returncode, proc.stderr[:200])
        )
    if not proc.stdout or len(proc.stdout) < 44:  # WAV header is 44 bytes
        raise RuntimeError("espeak-ng returned empty/short output")
    return proc.stdout


# --------------------------------------------------------------------------- #
# VoicePrompter
# --------------------------------------------------------------------------- #
class VoicePrompter:
    """Spoken-voice prompt manager used by ``ScanSession``.

    Parameters
    ----------
    enabled:
        When ``False``, every call to :meth:`speak` becomes a no-op.
        This is also the global kill-switch used by ``--no-voice``.
    language:
        espeak-ng voice id (e.g. ``"en"``, ``"en-us"``, ``"de"``).  On
        Windows the language is forwarded to pyttsx3 as a best-effort
        hint (most SAPI5 voices ignore it).
    rate_wpm:
        Speaking rate in words per minute.  Defaults to 165.
    backend:
        Force ``"pyttsx3"`` / ``"espeak"`` / ``"auto"``.  Defaults to
        ``"auto"`` which picks the first backend that is callable on
        this OS.  Useful for tests.

    Notes
    -----
    The VoicePrompter does NOT own the audio device.  Once a phrase has
    been synthesised into WAV bytes, the bytes are forwarded to
    :class:`sound.SoundPlayer._play_wav` so that tone and voice share the
    same backend (winsound on Windows, CLI on Linux/macOS).
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_ENABLED,
        language: str = DEFAULT_LANGUAGE,
        rate_wpm: int = DEFAULT_RATE_WPM,
        backend: str = DEFAULT_BACKEND,
        sound_player: Optional[object] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.language = str(language)
        self.rate_wpm = int(rate_wpm)

        self._backend_name, self._backend_callable = self._select_backend(backend)
        # Use the caller's SoundPlayer (e.g. ScanSession._sound) so that
        # volume / enable / backend are unified.  Fall back to a private
        # instance only when the caller doesn't supply one.
        self._sound = sound_player if sound_player is not None else self._make_sound_player()
        self._cache: Dict[Tuple[str, str], bytes] = {}
        self._cache_lock = threading.Lock()

        if self.enabled:
            logger.info(
                "VoicePrompter enabled (backend=%s, language=%s, rate=%d wpm)",
                self._backend_name, self.language, self.rate_wpm,
            )
        else:
            logger.info("VoicePrompter disabled")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_backend(prefer: str) -> Tuple[str, Optional[object]]:
        """Return ``(name, callable_or_None)`` for the chosen backend.

        The ``callable`` is a function ``(text: str) -> bytes`` that
        returns WAV bytes for ``text``.  ``None`` means "no backend".
        """
        prefer = (prefer or "auto").lower()

        pyttsx3_ok = _pyttsx3_available()
        espeak_ok = _espeak_available()

        if prefer == "pyttsx3" and pyttsx3_ok:
            return "pyttsx3", _synthesise_pyttsx3
        if prefer == "espeak" and espeak_ok:
            return "espeak", _synthesise_espeak

        # "auto"
        if pyttsx3_ok:
            return "pyttsx3", _synthesise_pyttsx3
        if espeak_ok:
            return "espeak", _synthesise_espeak
        return "none", None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_sound_player() -> Optional[object]:
        """Return a :class:`sound.SoundPlayer` we can borrow ``_play_wav`` from.

        Returns ``None`` if :mod:`sound` cannot be imported (should never
        happen in production but keeps the module standalone-testable).
        """
        try:
            import sound as _sound  # type: ignore[import-not-found]
            return _sound.SoundPlayer(enabled=True, volume=0.8)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("could not import sound.SoundPlayer: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    def configure(self, *, enabled: Optional[bool] = None,
                  language: Optional[str] = None,
                  rate_wpm: Optional[int] = None) -> None:
        """Toggle enable/language/rate at runtime."""
        if enabled is not None:
            self.enabled = bool(enabled)
        if language is not None:
            self.language = str(language)
        if rate_wpm is not None:
            self.rate_wpm = int(rate_wpm)

    # ------------------------------------------------------------------ #
    def phrase(self, event: str, **fmt) -> Optional[str]:
        """Return the rendered phrase for ``event`` (or ``None`` if unknown)."""
        template = _PHRASE_TEMPLATES.get(event)
        if template is None:
            logger.debug("speak(%r): unknown event, ignoring", event)
            return None
        try:
            return template.format(**fmt)
        except KeyError as exc:
            logger.debug(
                "speak(%r): missing format key %s, ignoring", event, exc,
            )
            return None

    # ------------------------------------------------------------------ #
    def _synthesise(self, text: str) -> Optional[bytes]:
        """Render ``text`` -> WAV bytes using the selected backend."""
        if self._backend_callable is None:
            return None
        try:
            if self._backend_name == "pyttsx3":
                return self._backend_callable(text, rate_wpm=self.rate_wpm)
            if self._backend_name == "espeak":
                return self._backend_callable(text, language=self.language)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("voice synthesis failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    def _get_or_render(self, event: str, phrase: str) -> Optional[bytes]:
        """Return cached WAV for ``(event, phrase)`` or render + cache it."""
        key = (event, phrase)
        with self._cache_lock:
            wav = self._cache.get(key)
            if wav is not None:
                return wav
        wav = self._synthesise(phrase)
        if wav is None:
            return None
        with self._cache_lock:
            # Double-checked; another thread may have raced us.
            self._cache.setdefault(key, wav)
        return wav

    # ------------------------------------------------------------------ #
    def speak(self, event: str, **fmt) -> bool:
        """Speak the phrase for ``event``.  Returns True if dispatched.

        Parameters
        ----------
        event:
            One of the keys in ``_PHRASE_TEMPLATES``.
        **fmt:
            Placeholder values for the phrase template, e.g.
            ``speak("capture_rejected", reason="too dark")`` or
            ``speak("document_saved", n=3)``.
        """
        if not self.enabled:
            return False

        phrase = self.phrase(event, **fmt)
        if phrase is None:
            return False

        wav = self._get_or_render(event, phrase)
        if wav is None:
            return False

        return self._dispatch(wav)

    # ------------------------------------------------------------------ #
    def _dispatch(self, wav: bytes) -> bool:
        """Hand ``wav`` to the shared audio backend (daemon thread)."""
        if self._sound is None:
            logger.debug("voice dispatch: no SoundPlayer available")
            return False

        def _runner() -> None:
            try:
                self._sound._play_wav(wav)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("voice dispatch failed: %s", exc)

        threading.Thread(target=_runner, daemon=True).start()
        return True


# --------------------------------------------------------------------------- #
# Module-level convenience (used when ScanSession doesn't need its own instance)
# --------------------------------------------------------------------------- #
_default_prompter: Optional[VoicePrompter] = None


def get_default_prompter() -> VoicePrompter:
    """Return the process-wide ``VoicePrompter`` (lazy-initialised)."""
    global _default_prompter
    if _default_prompter is None:
        _default_prompter = VoicePrompter()
    return _default_prompter


def speak(event: str, **fmt) -> bool:
    """Convenience: dispatch a named event through the default prompter."""
    return get_default_prompter().speak(event, **fmt)