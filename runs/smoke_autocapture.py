"""Smoke-test: build a ScanSession, touch `auto_capture`, ensure no crash."""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

from app import ScanSession
from config import DEFAULT_STABLE_FRAMES, DEFAULT_STABILITY_TOLERANCE

sess = ScanSession(
    camera_source="fake",
    camera_backend="opencv",
    camera_width=1280,
    camera_height=720,
    web_host="127.0.0.1",
    web_port=5000,
)
# web=False equivalent: disable the lazy FlaskServer by stubbing ensure_running.
sess._flask_server = type("_NoFlask", (), {"ensure_running": lambda self: None})()
sess.auto_capture_enabled = True
sess.auto_capture_cooldown_s = 1.5
sess.auto_capture_stable_frames = DEFAULT_STABLE_FRAMES
sess.auto_capture_tolerance_px = DEFAULT_STABILITY_TOLERANCE

print("controller before access:", sess._auto_capture)
ctrl = sess.auto_capture
print("after touch:",
      type(ctrl).__name__,
      "required_frames=", ctrl.tracker.required_frames,
      "tolerance=", ctrl.tracker.tolerance)
print("OK - no AttributeError on first property access")