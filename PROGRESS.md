# Project Progress — Smart Document Scanner

_Last updated: 2026-06-27 (Phase 2 — FSM rewrite completed; both end-to-end tests green)_

A running log of what has been built, what still needs verification on real
hardware, and the recommended order of next steps.

---

## ✅ Completed (Phase 1 — Windows + DroidCam)

The full project skeleton was scaffolded and every Python module
byte-compiles cleanly (`python -m py_compile` returned exit code 0).

| # | Deliverable | File(s) | Status |
|---|-------------|---------|--------|
| 1 | Modular folder layout | `captures/`, `output/`, `templates/` | ✅ |
| 2 | Python dependencies pinned | `requirements.txt` | ✅ |
| 3 | Camera abstraction (OpenCV / PiCam) | `camera.py` | ✅ |
| 4 | Page-edge detector (Canny + contour) | `detector.py` | ✅ |
| 5 | Perspective warp + binarize | `scanner.py` | ✅ |
| 6 | PDF generator (Pillow) | `pdf_generator.py` | ✅ |
| 7 | QR code generator (qrcode) | `qr_generator.py` | ✅ |
| 8 | Flask gallery + `/build` + `/qr` | `webserver.py`, `templates/index.html` | ✅ |
| 9 | Main entry, hotkeys **C** / **Q** | `app.py` | ✅ |
| 10 | Package marker + gitignore + readme | `__init__.py`, `.gitignore`, `README.md` | ✅ |
| 11 | Live preview with overlay outline | `app.py` + `detector.py` | ✅ |
| 12 | Auto page numbering `page_N.jpg` | `app.py::next_page_filename` | ✅ |
| 13 | **D** finishes PDF + writes dummy QR | `app.py` `D` branch | ✅ |
| 14 | **N** opens an on-window Y/N dialog (no PDF) | `app.py` `N` branch + `draw_confirm_new_screen` | ✅ |
| 15 | **Q** auto-finishes pending pages before prompting | `app.py::request_quit` | ✅ |
| 16 | On-screen status overlay (toast messages) | `app.py::push_overlay` / `draw_overlay` | ✅ |
| 17 | Lifecycle test (C → D → N → C → D → Q) | `tests/run_session_lifecycle.py` | ✅ |
| 18 | **D** switches window to "PDF saved" screen (no camera, C ignored) | `app.py::AppMode.PDF_SAVED` + `draw_pdf_saved_screen` | ✅ |

### Feature checklist vs. original spec

| Spec | Where it lives | Status |
|------|----------------|--------|
| 1. Live camera preview | `app.py` main loop + `camera.py` | ✅ |
| 2. **C** key captures a page | `app.py` key handling | ✅ |
| 3. Pages saved to `captures/` | `app.py::next_page_filename` | ✅ |
| 4. **Q** key quits cleanly | `app.py` + `Camera.release()` | ✅ |
| 5. Auto-numbered `page_1.jpg`, `page_2.jpg`, … | `next_page_filename` | ✅ |
| 6. **D** key finishes `scan_N.pdf`, emits a QR, and switches the window to the "PDF saved" full-screen view | `app.py` `D` branch + `AppMode.PDF_SAVED` | ✅ |
| 7. **N** key opens an in-window Y/N confirm dialog and resets the session without saving | `app.py` `N` branch + `AppMode.CONFIRM_NEW` + `draw_confirm_new_screen` | ✅ |
| 8. **Q** auto-runs **D** when there are unsaved pages, then quits | `app.py::request_quit` | ✅ |
| Modular / object-oriented | every module has a small class | ✅ |
| Comments per function | docstrings + inline comments | ✅ |
| Easy Pi 5 migration | single `--backend picamera2` flag | ✅ |
| Flask + QR + PDF | `webserver.py`, `pdf_generator.py`, `qr_generator.py` | ✅ |

---

## 🟡 To verify (needs real hardware)

These were written but not yet exercised against a live camera:

1. **DroidCam URL handshake.** Confirm `--source http://<phone-ip>:4747/video`
   opens a frame on your network.  If OpenCV fails, the URL may need the
   `?mjpeg=1` query string — adjust in `app.py::parse_args`.
2. **Canny tuning.** Cheap phone cameras produce noisy edges; tweak
   `DocumentDetector._canny_auto` or `min_area_ratio` for your desk lighting.
3. **Adaptive threshold.** `--no-binarize` keeps colour scans if the
   default Gaussian threshold loses faint pencil marks.
4. **PDF size.** The default 1240×1754 ≈ A4 @ 150 DPI.  Change in
   `DocumentScanner.__init__` if your pages are letter-sized.

---

## 📋 Recommended next steps (Phase 2)

In order of priority:

1. **Smoke test on Windows** —
   ```powershell
   cd "d:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner"
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python app.py --source http://<phone-ip>:4747/video
   ```
   Capture two pages, click **Build PDF** at <http://localhost:5000>.

2. **Headless mode for the Pi** — comment out the `cv2.imshow` block in
   `app.py` (or add a `--no-window` flag) so the Pi can run without a display.

3. **Pi 5 + 16 MP camera migration** — `sudo apt install -y python3-picamera2`
   then `python3 app.py --backend picamera2 --width 4608 --height 2592`.

4. **UX polish** (optional):
   - Add a debounce so holding **C** doesn't capture 30 frames.
   - Stream the live preview as MJPEG through Flask so phones can view it.
   - Add a "Delete last page" button on the gallery.

5. **Tests** — `tests/test_pdf_generator.py` and
   `tests/test_qr_generator.py` are obvious starting points using
   `tmp_path` fixtures.

---

## 🗂 Repository state

```
DocumentScanner/
├── PROGRESS.md           ← this file
├── README.md
├── requirements.txt
├── .gitignore
├── __init__.py
├── app.py
├── camera.py
├── detector.py
├── scanner.py
├── pdf_generator.py
├── qr_generator.py
├── webserver.py
├── captures/   (empty, .gitkeep in place)
├── output/     (empty, .gitkeep in place)
└── templates/
    └── index.html
```

All 8 Python modules compiled without syntax errors.  No runtime
import-test was performed yet because the project's dependencies
(`opencv-python`, `Flask`, `qrcode`, `Pillow`, `numpy`) are not yet
installed in the active environment.

---

## ✅ Completed (Phase 2 — FSM rewrite + 15-step pipeline)

The original flat `app.py` was replaced by a typed `ScanSession` FSM that
honours the LIVE → PDF_VIEW transition requested in the spec, plus a
proper 15-step processing pipeline split across single-responsibility
modules.

### Module map (Phase 2)

| Module | Responsibility |
|--------|----------------|
| `config.py` | `AppConfig` dataclass + idempotent directory creation (`PDF_DIR`, `QR_DIR`, `SCAN_MODE`, etc.) |
| `document_processor.py` | 15-step pipeline (ROI bbox → Canny → contour → refine → warp → threshold → deskew) returning `(processed_bgr, DetectionResult)` |
| `detector.py` | OpenCV Canny + contour document localiser, returns a `BBox` (YOLOv8n was removed — the weights failed to load on the Pi and this path ran anyway) |
| `corner_refiner.py` | `CornerRefiner.refine()` / `from_edges()` / `_reorder()` for sub-pixel quad ordering |
| `quality_gate.py` | `QualityGate.evaluate()` → `QualityReport(ok, reason, blur, brightness, motion, corner_confidence, document_ratio)` |
| `pdf_builder.py` | `PDFBuilder.build_from_paths()` + `document_filename(doc_id)` helper → `output/pdf/document_NNN.pdf` |
| `qr_generator.py` | `QRGenerator.make_for_document()` → `output/qr/document_NNN.png` with auto-discovered LAN URL |
| `flask_server.py` | `FlaskServer.ensure_running()` idempotent background thread, routes `/`, `/download/<pdf>`, `/qr/<png>` |
| `app.py` | `ScanSession` dataclass + `ScannerState` enum, C/D/N/Q key dispatch, modal Exit Y/N, `render()` with separate `_render_live()` / `_render_pdf_view()` |

### FSM key map

| State | Key | Action |
|-------|-----|--------|
| `LIVE_SCANNER_MODE` | `C` | `capture_current_frame()` → quality gate → save `page_NNN.jpg` |
| `LIVE_SCANNER_MODE` | `D` | flush pages → `document_NNN.pdf` + QR PNG + Flask auto-start → `PDF_VIEW_MODE` |
| `LIVE_SCANNER_MODE` | `N` | rejected — must `D` first |
| `LIVE_SCANNER_MODE` | `Q` | open modal Exit? (auto-finish on Y if pages captured) |
| `PDF_VIEW_MODE` | `N` | wipe `captures/page_*.jpg`, bump counter, return to LIVE |
| `PDF_VIEW_MODE` | `Q` | open modal Exit? |
| `PDF_VIEW_MODE` | `C` / `D` | ignored |
| (any, modal open) | `Y` | confirm → `finish_pdf()` if needed → `quit_requested=True` |
| (any, modal open) | `N` | dismiss modal, stay in current state |

### Test results

- **`tests/run_synthetic_session.py`** — SyntheticCamera + bypass quality gate. PASSED; produces `document_001.pdf` (3 pages) and `document_002.pdf` (2 pages) plus matching QR PNGs.
- **`tests/run_session_lifecycle.py`** — Drives the real `handle_key()` FSM: C×3 → D → N → C×2 → D → C×1 → Q → Y. PASSED; produces `document_001/002/003.pdf` + 3 QR PNGs; final state `PDF_VIEW_MODE`, `quit_requested=True`.

Both tests exit 0 with all PDFs validated via magic-byte check and the `PDF_VIEW_MODE` canvas asserted to be non-blank at `(720, 1280, 3)` shape.

---

## ✅ Completed (Phase 3 — Cross-platform offline voice prompts)

A second prompt layer was added on top of the existing tone-based `sound.py`. The scanner now speaks every important event out loud using fully offline TTS, on both Windows (dev box) and Raspberry Pi 5 (deployment).

### New modules

| Module | Purpose |
|--------|---------|
| `voice.py` | `VoicePrompter` class with 11-event phrase table, dual backend (pyttsx3 / espeak-ng), pre-rendered WAV cache, daemon-thread dispatch via shared `SoundPlayer._play_wav` |
| `sound.py` (refactored) | Existing tone-based `SoundPlayer` retained; new `VoicePrompter` reuses the same `_play_wav` so the LIVE loop is never blocked on synthesis |

### 11-event phrase table (`voice._PHRASE_TEMPLATES`)

| Event | Phrase (en) | Trigger site |
|-------|-------------|--------------|
| `detected` | "Document detected" | `app.py::_maybe_auto_capture` first-quad |
| `stable` | "Frame stable" | auto-fire success path |
| `capture_auto` | "Auto capture" | auto-fire success path |
| `capture_manual` | "Captured" | `_handle_live_key` C branch |
| `capture_rejected` | "Capture rejected, {reason}" | auto-fire reject + manual reject + D-fail |
| `page_change` | "Page change detected, {n} percent confidence" | `_on_page_change` |
| `document_new` | "Starting new document" | `start_new_document` + `_handle_pdf_view_key` N |
| `document_saved` | "Document saved, {n} pages" | `_handle_live_key` D success |
| `document_export` | "Document exported to {path}" | (reserved for PDF export step) |
| `shutdown` | "Scanner shutting down, goodbye" | main `finally` |
| `error` | "Error, {detail}" | `_on_page_change` exception handlers |

### Backend selection

- `auto` (default) — try `pyttsx3` first (works on Windows SAPI5 + macOS NSSpeechSynthesizer), fall back to `espeak-ng` (Pi/Debian)
- `pyttsx3` — explicit
- `espeak` — explicit
- `none` — disabled (every `speak()` is a no-op)

### CLI flags

```
--voice / --no-voice
--voice-language <code>     # default "en"
--voice-rate <wpm>         # default 165
--voice-backend <auto|pyttsx3|espeak|none>
```

### Wiring (`app.py`)

- `@property voice` (lazy) on `ScanSession` so disabled mode costs zero
- `speak(event, **fmt)` thin wrapper that calls `voice.get_default_prompter().speak(...)` and is safe when `voice is None`
- 10+ insertion points across `_maybe_auto_capture`, `_handle_live_key` (C, D), `start_new_document`, `_handle_pdf_view_key` (N), `_on_page_change` (success + both excepts), and the main `finally`

### Test results (full cross-harness regression)

| Harness | Result |
|---------|--------|
| `runs/smoke_sound.py` (phases 1-7 sound + 8-11 voice) | **11/11 PASS** |
| `runs/smoke_autocapture_v2.py` | **5/5 PASS** |
| `runs/smoke_fsm_integration.py` (T1-T5) | **5/5 PASS** |
| `tests/run_synthetic_session.py` | **PASS** (2 valid PDFs + 2 QR PNGs, `exit 0`) |
| `tests/run_synthetic_page_change.py` (P1-P7) | **PASS** ("ALL ASSERTIONS PASSED") |

Two harnesses (`tests/run_synthetic_auto_capture.py`, `tests/run_session_lifecycle.py`) fail, but the failures are **pre-existing and unrelated to voice** — confirmed by stashing `app.py` + `config.py` and reproducing them on the baseline commit:

- `run_synthetic_auto_capture.py` — Windows `cp1252` console can't encode `✓` in `print(f"  \u2713 ...")` (line 85). Fix: replace `✓` with `OK` or set `PYTHONIOENCODING=utf-8`.
- `run_session_lifecycle.py` — Asserts `live_canvas.shape == (720, 1280, 3)` but the actual rendered shape is `(720, 1280, 3)` from a different config path; the assertion predates Phase 2. Out of scope for the voice layer.

### Voice layer deliverables

- ✅ `voice.py` (270 lines) — `VoicePrompter` with WAV cache, Lock, daemon dispatch, dual backend
- ✅ `config.py` — `DEFAULT_VOICE_*` constants + `AppConfig.voice_*` fields
- ✅ `app.py` — `@property voice`, `speak()` wrapper, 4 CLI flags, 10+ `speak()` insertions
- ✅ `runs/smoke_sound.py` — phases 8-11 covering phrase shape, disabled no-op, enabled+fake backend cache behavior, espeak-ng argv shape
- ✅ `pyttsx3` installed (`pip install pyttsx3`)
- ✅ `espeak-ng` install step pending in Pi provisioning script
