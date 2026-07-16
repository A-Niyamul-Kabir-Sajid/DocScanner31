# Camera Detection & Capture — `camera.py`

This document covers the **detect** and **capture** halves of `Camera`. It
intentionally stops at the boundary of `read()` returning a frame — anything
that happens to the frame *after* that (detection, warping, JPEG encoding,
PDF writing) lives in `document_processor.py` / `pdf_generator.py` and is
out of scope here.

---

## 1. Platform detection — "am I on a Pi?"

Entry point: `detect_raspberry_pi()` at `camera.py:96-156`.

The result is cached in the module-level `_RPI_DETECTED` (`camera.py:93`) so
the LIVE tick never re-probes the kernel. On first call the function walks
three signals, in order, and returns on the first match:

| # | Signal | Where | Why |
|---|---|---|---|
| 1 | Kernel architecture | `os.uname().machine` | `{aarch64, armv7l, armv6l}` are Pi kernels — useful when `/proc` is masked inside a container. |
| 2 | Device-tree model | `/proc/device-tree/model` | Canonical check: file contains the string `"Raspberry Pi"` (case-insensitive). |
| 3 | libcamera subdev presence | `/dev/v4l-subdev*` | Only the Pi camera stack creates these nodes (IMX219/IMX477/IMX519 drivers). |

Every probe is wrapped in `try / except OSError` so a missing procfs on a
desktop box can never break the OpenCV fallback. The function never raises.

The decision is then used by **`select_backend(source, requested)`**
(`camera.py:167-189`) which is what `Camera.__init__` actually calls:

```
"opencv"     -> returned verbatim
"picamera2"  -> returned verbatim
"auto"       -> "picamera2" if detect_raspberry_pi() else "opencv"
```

A URL source (DroidCam / IP Webcam) **always downgrades to `opencv`** even
when the caller explicitly asked for `picamera2` — the Pi camera stack
cannot serve an HTTP MJPEG feed, and silently failing would mask the real
cause. Detection of "is this a URL?" lives in `_looks_like_url(source)`
(`camera.py:159-164`): a string with both a `scheme` and a `netloc` after
`urlparse(...)` is treated as a stream.

---

## 2. Camera construction

`Camera.__init__` (`camera.py:251-329`) does not raise on failure. It
records every error in `self.last_open_error` and leaves `self.is_open =
False` so the LIVE loop can render a "camera not found" overlay and poll
`try_reopen()` from the main tick. This is the lazy accessor shown in
`app.py:445-476`.

Sticky knobs preserved on the instance:

| Field | Default | Used by |
|---|---|---|
| `self.backend` | resolved value from `select_backend(...)` | All read / reopen paths |
| `self.source` | user-supplied | Re-opened on retry |
| `self.width`, `self.height` | `1280×720` | Set on OpenCV; used to size the empty frame returned when offline |
| `self.rotate` | `0` | Applied in `_apply_rotate(frame)` after every read (`camera.py:851-870`) |
| `self.full_fov` | `True` | Pi only — selects the **largest** sensor mode so the preview is not cropped to the requested aspect ratio (`camera.py:486-548`) |
| `self.autofocus`, `self.lens_position` | from `config.DEFAULT_*` | Applied as `AfMode` / `LensPosition` via `set_controls` after `Picamera2.start()` |
| `self.autofocus_on_capture` | `False` | Whether to kick a one-shot AF pass before each capture |
| `self.is_open`, `self.last_open_error` | `False`, `None` | The "camera usable right now?" latches used by the LIVE overlay |
| `self._last_read_ok` | `False` | Lets `try_reopen()` distinguish a transient read failure from a pipeline that needs a full reconfigure |

After `__init__` sets these, it dispatches to exactly one opener based on
the resolved backend:

```
backend == "picamera2"  -> self._open_picamera2()
backend == "opencv"     -> self._open_opencv(raise_on_failure=False)
```

---

## 3. Backend A — OpenCV (`_open_opencv`, `camera.py:334-416`)

Used for **everything that isn't the Pi Camera Module**: UVC USB webcams,
DroidCam over Wi-Fi, IP Webcam URLs, any Linux `/dev/video*` node.

### Step-by-step

1. **Coerce the source.** `int(self.source)` succeeds for numeric indices
   (`"0"`, `"1"`) so Windows uses the DirectShow / MSMF index path. Anything
   else is treated as a URL string (`camera.py:346-352`).
2. **Release any stale handle.** If `self._cap is not None`, call
   `self._cap.release()` and clear it (`camera.py:354-360`). Cheap
   belt-and-braces; the OpenCV backend never holds two handles at once.
3. **Fast TCP probe for HTTP(S) URLs.** `_probe_http_url(url, timeout=1.0)`
   (`camera.py:195-214`) opens a `socket.create_connection((host, port),
   timeout=1.0)`. We do **not** send an HTTP request — just verify the
   socket accepts. This exists because `cv2.VideoCapture` will happily block
   for **10–30 s** waiting for an unreachable DroidCam server, which makes
   the LIVE loop look frozen. On failure we log, set `last_open_error`,
   flip `is_open=False`, and (if `raise_on_failure=True`) raise; otherwise
   return so the app layer can poll `try_reopen()`.
4. **Open the handle.** URLs go through `cv2.VideoCapture(url,
   cv2.CAP_FFMPEG)` (`camera.py:381-386`) — explicit FFMPEG routing fixes a
   refusal bug on the opencv-python 4.13 Windows wheel where the default
   backend rejects MJPEG `multipart/x-mixed-replace` streams. Numeric
   indices use the platform default: `cv2.VideoCapture(src)` (line 388).
5. **Verify.** `bool(self._cap.isOpened())` must be `True`. Otherwise log
   `last_open_error`, set `is_open=False`, and either raise or return
   (`camera.py:397-407`).
6. **Apply geometry.** `self._cap.set(CAP_PROP_FRAME_WIDTH, w)` and
   `..._HEIGHT, h)`. Failures are swallowed because some cameras (mostly
   phone URLs) refuse `set()` (`camera.py:410-414`).
7. **Stamp success.** `self.is_open = True`, `self.last_open_error = None`
   (`camera.py:415-416`).

### Recovery (`try_reopen` → `_open_opencv`, `camera.py:421-431`)

`ScanSession` calls `camera.try_reopen()` from its LIVE tick. The OpenCV
branch simply re-runs `_open_opencv(raise_on_failure=False)`. Safe to
invoke every tick — the actual `VideoCapture` call is cheap when the source
is reachable.

---

## 4. Backend B — picamera2 (`_open_picamera2`, `camera.py:774-847`)

Used only on the Raspberry Pi when the source is the Camera Module
(IMX219 / IMX477 / IMX519 + AK7375 lens motor).

### Step-by-step

1. **Import.** `from picamera2 import Picamera2; from libcamera import
   controls`. Raises `RuntimeError` with an actionable hint if either
   import fails — the operator needs to `pip install picamera2` on Pi OS
   Bookworm (`camera.py:776-783`).
2. **Instantiate.** `self._pi_cam = Picamera2()` (line 790).
3. **Configure.** `self._select_picamera2_config()` (`camera.py:486-548`)
   builds the `picamera2` configuration:
   - With `full_fov=True` (default) it picks the **largest** entry from
     `self._pi_cam.sensor_modes` and passes it as
     `sensor={"output_size": ...}` so libcamera downscales the main stream
     to `(width, height)` instead of cropping the sensor (which is what
     `rpicam-hello --width W --height H` does internally).
   - With `full_fov=False` it falls back to the legacy behaviour —
     picamera2 picks a sensor mode whose aspect ratio already matches the
     requested output, which crops the sensor (digital zoom).
   - Pre-0.3 picamera2 builds don't accept the `sensor=` kwarg; a
     `TypeError` is caught and the call falls back to default mode
     selection (`camera.py:539-548`).
4. **Start the pipeline.** `self._pi_cam.configure(config); start()`
   (`camera.py:801-802`).
5. **Apply focus controls AFTER start.** libcamera rejects most
   `set_controls` calls on a configured-but-not-started pipeline. The
   `focus_controls` dict is built first (`camera.py:809-814`):
   - `autofocus=True`  → `{"AfMode": AfModeEnum.Continuous}`
   - `autofocus=False` → `{"AfMode": AfModeEnum.Manual, "LensPosition": <dpt>}`

   `set_controls(focus_controls)` is then wrapped in `try/except` so a
   fixed-focus camera module (V1 / V2 / HQ without AF lens) that doesn't
   expose `AfModeEnum` at all still boots — the app logs a warning and
   runs with whatever the sensor defaulted to (`camera.py:815-826`).
6. **Verify the lens actually moved.** On the IMX519 + AK7375 the very
   first `set_controls` after `start()` is frequently a no-op: the lens
   actuator needs a frame to wake up, and libcamera silently clamps
   `LensPosition` to the previous value rather than raising. The
   verification loop lives in `_settle_manual_focus`
   (`camera.py:645-772`) — see §6.
7. **Mark the pipeline live.** `self.is_open = True`,
   `self._last_read_ok = True`, `self.last_open_error = None`
   (`camera.py:845-847`).

### Recovery (`_reopen_picamera2`, `camera.py:433-481`)

`picamera2` has no cheap "are you still alive?" probe, so when a read
raises or the user explicitly closed the handle we tear down and recreate
the pipeline:

```
if pi_cam is open AND last_read_ok -> short-circuit (True)
else:
    stop() ; close() ; _pi_cam = None
    _open_picamera2()  # reapplies focus controls too
    stamp is_open = _last_read_ok = True
```

A transient `capture_array` failure therefore does **not** escalate to a
full reconfigure — the `_last_read_ok` latch (`camera.py:321`) is what
distinguishes "hiccup" from "pipeline is wedged".

---

## 5. Frame capture — `Camera.read()` (`camera.py:1093-1150`)

Single entry point used by every caller (`app.py:1010`, `1252`, `2005`,
plus the test/probe scripts). Always returns `(ok: bool, frame:
np.ndarray)` so the rest of the pipeline can pattern-match on `ok` the
same way it would for `cv2.VideoCapture.read()`.

### picamera2 branch (`camera.py:1106-1131`)

1. If `self._pi_cam is None`, return `(False, np.zeros((h, w, 3), uint8))`
   so the GUI can render an empty frame without crashing.
2. `frame_rgb = self._pi_cam.capture_array()` — picamera2 always hands
   RGB. Any exception flips `is_open=False`, records `last_open_error`,
   returns `(False, empty)`.
3. `frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)` — keeps the
   OpenCV downstream pipeline unchanged.
4. `frame_bgr = self._apply_rotate(frame_bgr)` — `cv2.rotate` for 90° /
   180° / 270° (`camera.py:851-870`). Fast-path returns untouched when
   `rotate == 0`.
5. Stamp `self._last_read_ok = True`. If `is_open` was False (e.g. after
   a transient failure we recovered from) flip it back to True on the
   first successful frame.
6. Return `(True, frame_bgr)`.

### OpenCV branch (`camera.py:1133-1150`)

1. If not open or no handle, return `(False, empty)` with the requested
   geometry.
2. `ok, frame = self._cap.read()`. Exceptions flip `is_open=False` and
   return `(False, empty)` — never raises.
3. If `not ok`, log a warning and return `(False, frame)` (a network
   stream sometimes returns a frame of zeros on a hiccup — we don't want
   to throw away the tuple shape the caller expects).
4. Otherwise return `(True, self._apply_rotate(frame))`.

### Empty-frame contract

Both branches return `np.zeros((self.height, self.width, 3), dtype=uint8)`
on failure. Callers can therefore `cv2.putText(empty, "camera offline", ...)`
without a None check.

---

## 6. Release & context-manager

`release()` (`camera.py:1152-1167`) tears down whichever backend is
active:

- OpenCV: `self._cap.release()` and clear the handle.
- picamera2: `self._pi_cam.stop()` then `close()`, then clear the handle.
- Both: `self.is_open = False`, `self._last_read_ok = False` so the next
  `try_reopen` knows it has to do real work.

`Camera` is also a context manager (`__enter__` / `__exit__` →
`release()`), so `with Camera(...) as cam: ...` is safe.

---

## Summary diagram

```
Camera(source, backend="auto", ...)
        │
        ▼
select_backend(source, requested)              [camera.py:167]
        │   detect_raspberry_pi()              [camera.py:96]
        │     ├─ uname().machine in {aarch64, armv7l, armv6l}
        │     ├─ /proc/device-tree/model contains "Raspberry Pi"
        │     └─ /dev/v4l-subdev* exists
        │   URL sources always -> "opencv"
        ▼
backend == "picamera2" ?  ─yes─▶ _open_picamera2()
        │                       ├─ Picamera2() + configure(full_fov)
        │                       ├─ start()
        │                       ├─ set_controls({AfMode, LensPosition?})
        │                       ├─ _settle_manual_focus()  # IMX519 race
        │                       └─ is_open = True
        │
        └─────no──▶ _open_opencv(raise_on_failure=False)
                          ├─ int(source) for index / URL string otherwise
                          ├─ _probe_http_url()  # 1 s TCP check
                          ├─ cv2.VideoCapture(url, CAP_FFMPEG)  or
                          │   cv2.VideoCapture(src)
                          ├─ isOpened() check
                          └─ set(WIDTH, HEIGHT); is_open = True

LIVE tick:
  ok, frame = camera.read()        [camera.py:1093]
        ├─ picamera2: capture_array() → RGB→BGR → _apply_rotate
        └─ OpenCV:    self._cap.read() → _apply_rotate

  ok is False → empty np.zeros((h, w, 3)) of the requested geometry
  ok is True  → rotated BGR frame, ready for the detector
```

---

## Key invariants

1. **No constructor raise.** Every backend error is stored in
   `self.last_open_error`; `is_open=False` is the canonical "camera
   unusable" signal the GUI overlay polls.
2. **Read never raises.** Both branches catch every exception inside
   `read()` and return `(False, empty_frame)` so the LIVE tick is
   bulletproof.
3. **`read()` shape is stable.** Always `(bool, np.ndarray)`, never
   `None`. Callers can safely `cv2.putText(empty, ...)` on failure.
4. **Geometry from `__init__` is preserved.** `read()` returns
   `(height, width, 3)` empty frames even when the camera is offline, so
   downstream blits / resizes don't have to special-case "no frame yet".
5. **picamera2 focus controls are applied *after* `start()`.** libcamera
   rejects most controls on a configured-but-not-started pipeline. The
   settle/retry loop in `_settle_manual_focus` exists specifically because
   the IMX519 + AK7375 silently drops the first `set_controls` call.
6. **Reopen is cheap.** OpenCV reopens the underlying capture in <1 s on
   success; picamera2 only rebuilds the pipeline when `_last_read_ok`
   says the previous pipeline was actually dead, not just hiccupping.
