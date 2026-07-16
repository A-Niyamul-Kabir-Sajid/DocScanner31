# Tuning the Auto-Capture FSM

The auto-capture loop is a two-state FSM that lives in
`auto_capture_controller.py`. Both states have timing knobs you may want
to tweak — here is exactly what each one does and where to change it.

---

## State 2 → State 1 transition (the "no-match" timer)

> **Behaviour:** after a page is captured, the FSM sits in
> `S2_WAITING_FOR_CHANGE`. While the live quad keeps *matching* the quad
> we just saved, nothing happens. The moment the live quad diverges (page
> pulled out, swapped, rotated, hand in the way), the FSM starts a
> wall-clock timer. When the timer reaches the threshold, the FSM flips
> back to `S1_SEEKING_STABLE` so the next page can auto-capture.

**Knob:** `AutoCaptureController.s2_no_match_timeout_s`
**File:** `auto_capture_controller.py`, **line 68**
**Current value:** `1.5` seconds
**Type:** float (wall-clock seconds, NOT frames)

```python
@dataclass
class AutoCaptureController:
    ...
    # Continuous seconds of "no match" required to flip State 2 -> 1.
    # The State 2 timer measures how long the live quad has been *not similar*
    # to ``last_captured_quad``; when it reaches this value the FSM returns
    # to State 1 so the next auto-capture can fire.  Currently 1.5 s.
    s2_no_match_timeout_s: float = 1.5   # ← edit this number
```

### What to change it to

| Goal | Value |
|---|---|
| Snap back to ready immediately after the page leaves the frame | `0.5` – `1.0` |
| Default (current) | `1.5` |
| Give the user a moment to settle a freshly placed page | `2.0` – `3.0` |
| Avoid spurious double-captures on shaky hands | `3.0` – `5.0` |

### How to override without editing the file

Pass it through the constructor at the call site:

```python
ctrl = AutoCaptureController(
    enabled=True,
    s2_no_match_timeout_s=2.0,   # 2-second S2 → S1 window
)
```

### What it is NOT

- **Not frame-based.** It uses `time.monotonic()` deltas in
  `_observe_state2` (`auto_capture_controller.py:316-322`), so changing
  the LIVE tick rate does **not** affect it.
- **Not a cooldown.** There is no fixed cooldown between captures any
  more — the no-match timer is the only gate.

---

## State 1 stability window (the "hold still" timer)

> **Behaviour:** while the FSM is in `S1_SEEKING_STABLE` it asks the
> `StabilityTracker` how many consecutive frames have shown a
> sufficiently-similar quad. When that count reaches `required_frames`,
> the FSM fires a capture and parks in State 2.

There are **three** cooperating knobs because "stable" has two parts:
*how many frames*, and *how close the corners must be*.

### Knob A — frame count

**File:** `config.py`, **line 143**
**Current value:** `60` (≈1.8 s at the 30 ms LIVE tick)

```python
# Stable corners required before auto-capture fires.  At the 30 ms LIVE
# tick (~33 fps) ``required_frames`` is roughly the wait time in seconds
# times 33, so the default 60 is ~1.8 s on the LIVE loop.  Bump this for a
# longer "hold still" window, drop it for snappier auto-capture.
DEFAULT_STABLE_FRAMES: int = 60   # ← edit this number
```

The LIVE loop runs at the heartbeat tick (≈30 ms in `app.py`), so:

| Seconds of stability | `DEFAULT_STABLE_FRAMES` |
|---|---|
| ~0.5 s (snappy) | `16` |
| ~1.0 s | `33` |
| ~1.8 s (current default) | `60` |
| ~2.0 s | `66` |
| ~3.0 s (conservative) | `99` |

### Knob B — drift tolerance

**File:** `config.py`, **line 149**
**Current value:** `18.0` pixels

```python
# Maximum corner drift (pixels) tolerated between consecutive frames.
# The contour + approxPolyDP pass routinely jitters 8-15 px even when the
# document is held still, so a tight 6 px threshold never lets the streak
# build.  18 px is comfortably above that noise floor but still below the
# drift you get from a slow hand swap (~40+ px).
DEFAULT_STABILITY_TOLERANCE: float = 18.0   # ← edit this number
```

Two consecutive frames are treated as "the same quad" only when the
maximum corner distance (computed in `corner_refiner.distance`,
`corner_refiner.py:81-94`) is ≤ `DEFAULT_STABILITY_TOLERANCE`.

| Goal | Value |
|---|---|
| Very strict — only fires when the page is glued down | `6.0` – `10.0` |
| Default (current) | `18.0` |
| Forgiving of detector jitter / phone-camera MJPEG | `25.0` – `35.0` |

**Important:** `DEFAULT_STABILITY_TOLERANCE` is mirrored by
`AutoCaptureController.tolerance` (`auto_capture_controller.py:66`).
They should match, otherwise State 1 and State 2 disagree about what
"similar" means.

### Knob C — jitter band (advanced)

**File:** `stability_tracker.py`, **line 21**
**Current value:** `2.5`

```python
# Drift between ``tolerance`` and ``tolerance * jitter_band`` is treated as
# detection noise: the streak is *paused* (count stays put, baseline keeps
# the previous quad) instead of being reset.  Above the band we assume the
# document genuinely moved and reset the counter to 1.
jitter_band: float = 2.5   # ← edit this number
```

Drift between `tolerance` and `tolerance * jitter_band` is treated as
**detector noise** — the streak pauses instead of resetting. If you ever
see the stable counter freeze and never reach the threshold, bump this
to `3.0` – `4.0`.

---

## Quick reference

| Knob | File:line | Effect | Default |
|---|---|---|---|
| `s2_no_match_timeout_s` | `auto_capture_controller.py:68` | Wall-clock seconds before S2 → S1 | `1.5 s` |
| `DEFAULT_STABLE_FRAMES` | `config.py:143` | Frames of stillness before S1 fires | `60` (~1.8 s) |
| `DEFAULT_STABILITY_TOLERANCE` | `config.py:149` | Max pixel drift counted as "still" | `18.0 px` |
| `jitter_band` | `stability_tracker.py:21` | Multiple of tolerance still treated as noise | `2.5` |
| `motion_trigger_px` | `auto_capture_controller.py:69` | MAD above which a frame is a "motion spike" | `25.0 px` |
| `AutoCaptureController.tolerance` | `auto_capture_controller.py:66` | Mirror of `DEFAULT_STABILITY_TOLERANCE` used in S2 | `18.0` |

---

## Recipes

**Snappy scanner** (fire fast, accept some risk of blur):
```python
DEFAULT_STABLE_FRAMES = 33          # ~1.0 s
s2_no_match_timeout_s = 1.0
DEFAULT_STABILITY_TOLERANCE = 12.0
```

**Conservative scanner** (no spurious captures, no wobbly hands):
```python
DEFAULT_STABLE_FRAMES = 99          # ~3.0 s
s2_no_match_timeout_s = 3.0
DEFAULT_STABILITY_TOLERANCE = 25.0
jitter_band = 3.5
```

**Current balance** (project default):
```python
DEFAULT_STABLE_FRAMES = 60          # ~1.8 s
s2_no_match_timeout_s = 1.5
DEFAULT_STABILITY_TOLERANCE = 18.0
jitter_band = 2.5
```
