"""Smoke-test the modernized UI: list routes + hit every endpoint via test client."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import app as app_mod
from flask_server import FlaskServer


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dscan_ui_"))
    try:
        scanned = tmp / "scanned"
        raw = tmp / "raw"
        out = tmp / "output"
        for d in (scanned, raw, out, out / "pdf", out / "qr"):
            d.mkdir(parents=True, exist_ok=True)

        sess = app_mod.ScanSession(
            captures_dir=scanned,
            output_dir=out,
            camera_source=99,           # never read
            web_host="127.0.0.1",
            web_port=0,
            scan_mode="color",
        )
        sess.pdf_dir = out / "pdf"
        sess.qr_dir = out / "qr"
        sess.scanned_dir = scanned
        sess.raw_dir = raw
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        print("session setup failed:", exc)
        return 1

    srv = FlaskServer(sess, host="127.0.0.1", port=5055)
    srv.ensure_running()
    time.sleep(0.4)

    try:
        _exercise(srv)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def _exercise(srv: FlaskServer) -> None:

    print("--- routes registered ---")
    for rule in sorted(srv._app.url_map.iter_rules(), key=lambda r: str(r)):
        methods = sorted(m for m in rule.methods if m not in {"HEAD", "OPTIONS"})
        print(f"  {str(methods):22} {rule.rule}")

    with srv._app.test_client() as c:
        print("\n--- GET / (rendered) ---")
        r = c.get("/")
        print("  status", r.status_code, "len", len(r.data))

        print("\n--- GET /api/status ---")
        r = c.get("/api/status")
        print("  status", r.status_code, "json", json.dumps(r.get_json()))

        print("\n--- POST /finish (no pages) ---")
        r = c.post("/finish")
        print("  status", r.status_code, "json", json.dumps(r.get_json()))

        print("\n--- POST /new-session ---")
        r = c.post("/new-session")
        print("  status", r.status_code, "json", json.dumps(r.get_json()))

        print("\n--- POST /delete-last-page (empty) ---")
        r = c.post("/delete-last-page")
        print("  status", r.status_code, "json", json.dumps(r.get_json()))

        print("\n--- POST /quit ---")
        r = c.post("/quit")
        print("  status", r.status_code, "json", json.dumps(r.get_json()))

        print("\n--- POST /quit?save=1 ---")
        r = c.post("/quit?save=1")
        print("  status", r.status_code, "json", json.dumps(r.get_json()))

        print("\n--- POST /quit?discard=1 ---")
        r = c.post("/quit?discard=1")
        print("  status", r.status_code, "json", json.dumps(r.get_json()))

        # /captures, /pdf, /qr should 404 cleanly when files don't exist.
        print("\n--- GET /captures/missing.jpg (expect 404) ---")
        r = c.get("/captures/missing.jpg")
        print("  status", r.status_code)

        print("\n--- GET /pdf/999 (expect 404) ---")
        r = c.get("/pdf/999")
        print("  status", r.status_code)

        print("\n--- GET /qr/999 (expect 404) ---")
        r = c.get("/qr/999")
        print("  status", r.status_code)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())